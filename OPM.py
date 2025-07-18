import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

LOT_SIZE = 100  # one option contract = 100 shares

def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == 'call':
        return S * stats.norm.cdf(d1) - K * np.exp(-r*T) * stats.norm.cdf(d2)
    else:
        return K * np.exp(-r*T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

def calculate_greeks_vec(S, K, T, r, sigma, option_type_arr):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    delta = np.where(option_type_arr=='call',
                     stats.norm.cdf(d1),
                     stats.norm.cdf(d1)-1)
    gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = ((-S*stats.norm.pdf(d1)*sigma/(2*np.sqrt(T))
              - np.where(option_type_arr=='call',
                         r*K*np.exp(-r*T)*stats.norm.cdf(d2),
                         -r*K*np.exp(-r*T)*stats.norm.cdf(-d2)))
             ) / 365
    vega = S * stats.norm.pdf(d1) * np.sqrt(T) / 100
    return delta, gamma, theta, vega

def get_stock_data(ticker):
    hist = yf.Ticker(ticker).history(period="1y")
    if hist.empty:
        return None, None, None
    S = hist['Close'].iloc[-1]
    vol = hist['Close'].pct_change().dropna().std() * np.sqrt(252)
    return S, vol, hist

def main():
    st.set_page_config(page_title="Option Chain Sell Analysis", layout="wide")
    st.title("📝 Sell-Only Option Chain Analysis")

    # Sidebar inputs
    st.sidebar.header("Parameters")
    ticker = st.sidebar.text_input("Ticker", "AAPL").upper()
    today = datetime.now().date()
    st.sidebar.write(f"Date: {today}")

    S, vol, hist = get_stock_data(ticker)
    if S is None:
        st.sidebar.error("No data for ticker.")
        return

    st.sidebar.success(f"Spot Price: ${S:.2f}")
    st.sidebar.info(f"Volatility: {vol*100:.1f}%")

    # Sell side selector
    side = st.sidebar.selectbox("Position", ["Sell Call", "Sell Put"])
    opt_type = side.split()[1].lower()  # 'call' or 'put'

    # Expiry picker
    expiries = yf.Ticker(ticker).options
    future = [d for d in expiries if datetime.strptime(d, '%Y-%m-%d').date() > today]
    expiry = st.sidebar.selectbox("Expiry", future or [(today + timedelta(days=30)).isoformat()])
    T = (datetime.strptime(expiry, '%Y-%m-%d').date() - today).days / 365.0

    # Cash and model parameters
    cash = st.sidebar.number_input("Cash Available ($)", 0, 10_000_000, 10_000, 100)
    r_pct       = st.sidebar.slider("Risk‑Free Rate (%)", 0.0, 10.0, 5.0, 0.1)
    vol_adj_pct = st.sidebar.slider("Vol Adjustment (%)", -50, 50, 0, 1)
    r       = r_pct / 100
    adj_vol = vol * (1 + vol_adj_pct / 100)

    # Fetch option chain
    chain = yf.Ticker(ticker).option_chain(expiry)
    df = chain.calls if opt_type=='call' else chain.puts

    # Market premium: midpoint(bid,ask) or fallback to lastPrice
    df['market_premium'] = np.where(
        (df.bid > 0) & (df.ask > 0),
        (df.bid + df.ask) / 2,
        df.lastPrice
    )

    # Theoretical price and diff%
    df['theo']     = black_scholes(S, df.strike, T, r, adj_vol, opt_type)
    df['diff_pct'] = (df.theo - df.market_premium) / df.market_premium * 100

    # Vectorized greeks
    delta, gamma, theta, vega = calculate_greeks_vec(
        S,
        df.strike.values,
        T,
        r,
        adj_vol,
        np.array([opt_type] * len(df))
    )
    df['delta'], df['gamma'], df['theta'], df['vega'] = delta, gamma, theta, vega


    # Select the 10 strikes closest to spot, reset their index for styling
    df['dist'] = (df.strike - S).abs()
    count = st.sidebar.slider("Number of strikes to show", 5, 50, 10, 1)

    # then build subset with that:
    subset = (
        df.nsmallest(count, 'dist')
        .sort_values('strike')
        .reset_index(drop=True)
    )
    atm_pos = subset['dist'].idxmin()  # 0–9 position of ATM

    # Max contracts based on spot * 100
    max_contracts = int(cash / (S * LOT_SIZE))
    st.sidebar.markdown(f"**Max Contracts (spot basis):** {max_contracts}")

    # Display the 10-strike table with ATM row highlighted
    st.subheader(f"{side} Chain @ {expiry}")
    display = subset[['strike','market_premium','theo','diff_pct','delta','gamma','theta','vega']].copy()
    display = display.rename(columns={
        'strike':'Strike',
        'market_premium':'Market Premium',
        'theo':'Theoretical',
        'diff_pct':'Diff (%)',
        'delta':'Δ',
        'gamma':'Γ',
        'theta':'Θ',
        'vega':'ν',
    })
    styled = (
        display.style
               .format({
                   'Market Premium':'${:,.2f}',
                   'Theoretical':'${:,.2f}',
                   'Diff (%)':'{:+.1f}%',
                   'Δ':'{:.3f}',
                   'Γ':'{:.3f}',
                   'Θ':'{:.3f}',
                   'ν':'{:.3f}'
               })
               .apply(
                   lambda row: [
                       'background-color: lightblue' if row.name == atm_pos else ''
                       for _ in row
                   ],
                   axis=1
               )
    )
    st.dataframe(styled)

    # Optional: drill into a single strike
    pick = st.selectbox("Inspect Strike", subset.strike.tolist())
    if pick is not None:
        prices = np.linspace(S*0.8, S*1.2, 50)
        opts   = [
            black_scholes(p, pick, T, r, adj_vol, opt_type)
            for p in prices
        ]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices, y=opts, mode='lines', name='Theoretical'))
        fig.add_vline(x=S, line_dash='dash', annotation_text='Spot')
        fig.add_vline(x=pick, line_dash='dash', annotation_text='Strike')
        fig.update_layout(
            title=f"{opt_type.title()} @ {pick}",
            xaxis_title='Stock Price',
            yaxis_title='Option Price'
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
