import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import plotly.express as px

warnings.filterwarnings("ignore")

LOT_SIZE = 100


def plot_series(data, title, chart_type, current_price=None):
    if chart_type == "Candlestick":
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close']
        )])
    else:
        fig = px.line(
            data.reset_index(),
            x='Date',
            y='Close',
            title=title,
            markers=True,
            hover_data={'Open':':.2f','High':':.2f','Low':':.2f','Close':':.2f'}
        )
    if current_price is not None:
        fig.add_hline(
            y=current_price,
            line_dash="dash",
            line_color="gray",   
            annotation_text=f"Current: ${current_price:.2f}",
            annotation_position="top right"
        )
    fig.update_layout(
        title=title,
        hovermode='x unified',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def get_stock_data(ticker):
    # fetch last price for display + overlay
    hist = yf.Ticker(ticker).history(period="1y")
    if hist.empty:
        st.error("No data for ticker.")
        return None
    price = hist["Close"].iloc[-1]
    st.metric(label="Current Price", value=f"{price:.2f}")

    # sidebar controls
    c1, c2, c3 = st.columns(3)
    with c1:
        hist_period = st.selectbox("Historical period", ["1mo","3mo","6mo","1y","5y","max"], index=1)
    with c2:
        hist_interval = st.selectbox("Historical interval", ["1d","1wk","1mo"], index=0)
    with c3:
        hist_type = st.selectbox("Chart type", ["Line","Candlestick"], index=0, key="historical")

    # fetch and plot
    data = yf.Ticker(ticker).history(period=hist_period, interval=hist_interval)
    if data.empty:
        st.error(f"No historical data for {ticker}")
    else:
        data.index.name = 'Date'
        fig = plot_series(
            data,
            f"{ticker} Historical ({hist_period}, {hist_interval})",
            hist_type,
            current_price=price 
        )
        st.plotly_chart(fig, use_container_width=True)

    return price

def get_earnings(ticker):
    T = yf.Ticker(ticker)
    cal = T.calendar
    next_er = None
    if hasattr(cal, "index") and "Earnings Date" in cal.index:
        next_er = pd.to_datetime(cal.loc["Earnings Date"].values[0]).date()
    elif isinstance(cal, dict) and cal.get("Earnings Date"):
        raw = cal["Earnings Date"][0] if isinstance(cal["Earnings Date"], (list, tuple, np.ndarray)) else cal["Earnings Date"]
        next_er = pd.to_datetime(raw).date()
    if next_er:
        days_to_er = (next_er - datetime.now().date()).days
        st.info(f"Next earnings: **{next_er}** (in {days_to_er} days)")
    else:
        st.info("No upcoming earnings date available.")
    today = datetime.now().date()
    expiries = T.options
    future = [d for d in expiries if datetime.strptime(d, "%Y-%m-%d").date() > today]
    choices = []
    for d in future:
        d_date = datetime.strptime(d, "%Y-%m-%d").date()
        days = (d_date - today).days
        label = f"{d} ({days}D)"
        if next_er and d_date <= next_er < d_date + timedelta(days=7):
            label += " Earnings Week"
        choices.append((d, label))
    default = (today + timedelta(days=30)).isoformat()
    expiry = st.selectbox("Expiry", [c[0] for c in choices] or [default], format_func=lambda d: dict(choices).get(d, d))
    return expiry

def formulate_df(ticker, current_price, expiry):
    chain = yf.Ticker(ticker).option_chain(expiry)
    calls, puts = chain.calls.copy(), chain.puts.copy()
    for df in (calls, puts):
        df["market_premium"] = np.where((df.bid > 0) & (df.ask > 0), (df.bid + df.ask) / 2, df.lastPrice)
    merged = pd.merge_asof(
        calls.sort_values("strike"),
        puts.sort_values("strike"),
        on="strike",
        suffixes=("_call", "_put"),
        direction="nearest"
    )
    merged["dist"] = (merged.strike - current_price).abs()
    return merged.nsmallest(30, "dist").sort_values("strike").reset_index(drop=True)

def get_max_lot(current_price):
    cash = st.number_input("Cash Available (USD)", 0, 10_000_000, 40_000, 100)
    max_contracts = int(cash / (current_price * LOT_SIZE))
    st.markdown(f"**Max Contracts:** {max_contracts}")

def display(subset, current_price, expiry):
    subset["Breakeven Call"] = subset.strike + subset.market_premium_call
    subset["Breakeven Put"] = subset.strike - subset.market_premium_put
    display_df = subset[[
        "Breakeven Call", "market_premium_call", "strike", "market_premium_put", "Breakeven Put"
    ]].rename(columns={
        "market_premium_call": "SELL Call Premium",
        "strike":              "Strike",
        "market_premium_put":  "SELL Put Premium"
    })
    for col in ["SELL Call Premium", "SELL Put Premium", "Breakeven Call", "Breakeven Put", "Strike"]:
        display_df[col] = display_df[col].map("${:,.2f}".format)
    atm_idx = (subset.strike - current_price).abs().idxmin()
    styled = display_df.style.apply(
        lambda row: ["background-color: lightblue" if row.name == atm_idx else "" for _ in row],
        axis=1
    )
    height = display_df.shape[0] * 35 + 40
    st.subheader(f"Option Premium @ {expiry}")
    st.dataframe(styled, height=height, use_container_width=True)

def main():
    st.set_page_config(page_title="Option Analysis", layout="wide")
    st.title("📝 Sell-Only Option Analysis")
    ticker = st.text_input("Ticker", "PLTR").upper()
    price = get_stock_data(ticker)
    if price is None:
        return
    expiry = get_earnings(ticker)
    subset = formulate_df(ticker, price, expiry)
    get_max_lot(price)
    display(subset, price, expiry)

if __name__ == "__main__":
    main()
