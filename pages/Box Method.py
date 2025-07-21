import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def get_tickers():
    default_tickers = ["PLTR", "NVDA", "MSFT", "TSLA", "GOOGL",
                       "AMZN", "TSM", "PANW", "SOFI", "NVO", "META"]
    extra_input = st.text_input("Add more tickers (comma‑separated)", value="")
    extra = [t.strip().upper() for t in extra_input.split(",") if t.strip()]
    tickers = list(dict.fromkeys(default_tickers + extra))
    if not tickers:
        st.info("No tickers to display.")
        return []
    return tickers

def get_box_method(tickers):
    results = []
    for t in tickers:
        try:
            daily = yf.Ticker(t).history(period="3d", interval="1d")
            y = daily.iloc[-2]
            high_y, low_y = y["High"], y["Low"]
            mid = (high_y + low_y) / 2
            intraday = yf.Ticker(t).history(period="1d", interval="5m")
            current = intraday["Close"].iloc[-1]
            band = (high_y - low_y) * 0.1
            lower_mid = mid - band/2
            upper_mid = mid + band/2
            if lower_mid <= current <= upper_mid:
                zone = "Middle"
            elif current > upper_mid:
                zone = "Top"
            else:
                zone = "Bottom"
            results.append({
                "Ticker": t,
                "Yesterday High": high_y,
                "Yesterday Low": low_y,
                "Midpoint": mid,
                "Current Price": current,
                "Zone": zone
            })
        except Exception as e:
            results.append({"Ticker": t, "Yesterday High": None,
                            "Yesterday Low": None, "Midpoint": None,
                            "Current Price": None, "Zone": "Error",
                            "Error": str(e)})
    df = pd.DataFrame(results).set_index("Ticker")
    fmt = lambda x: f"{x:,.2f}" if pd.notna(x) else "-"
    for col in ["Yesterday High", "Yesterday Low", "Midpoint", "Current Price"]:
        df[col] = df[col].apply(fmt)
    st.subheader("Box Levels, Current Price & Zone")
    height = df.shape[0] * 35 + 40
    st.dataframe(df, height=height, use_container_width=True)
    return df

def plot_series(data, title, chart_type):
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
            x=data.index,
            y='Close',
            title=title,
            markers=True,
            hover_data={'Open':':.2f','High':':.2f','Low':':.2f','Close':':.2f'}
        )
    fig.update_layout(title=title, hovermode='x unified', margin=dict(l=40,r=40,t=60,b=40))
    return fig




def historical_analysis(selected):
    c1, c2, c3 = st.columns(3)
    with c1:
        hist_period = st.selectbox("Historical period", ["1mo","3mo","6mo","1y","5y","max"], index=1)
    with c2:
        hist_interval = st.selectbox("Historical interval", ["1d","1wk","1mo"], index=0)
    with c3:
        hist_type = st.selectbox("Chart type", ["Line","Candlestick"], index=0, key="historical")
    data = yf.Ticker(selected).history(period=hist_period, interval=hist_interval)
    if data.empty:
        st.error(f"No historical data for {selected}")
    else:
        data.index.name = 'Date'
        fig = plot_series(data, f"{selected} Historical ({hist_period}, {hist_interval})", hist_type)
        fig.update_layout(
            title=f"{selected} Historical ({hist_period}, {hist_interval})",
            xaxis_title='Date', yaxis_title='Price', hovermode='x unified',
            margin=dict(l=40, r=40, t=60, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

def intra_analysis(selected, df):
    c1, c2, c3 = st.columns(3)
    with c1:
        intra_period = st.selectbox("Intraday period", ["1d","5d","7d"], index=0)
    with c2:
        intra_interval = st.selectbox("Intraday interval", ["1m","5m","15m","30m","60m"], index=1)
    with c3:
        intra_type = st.selectbox("Chart type", ["Line","Candlestick"], index=0, key="intra")
    data = yf.Ticker(selected).history(period=intra_period, interval=intra_interval)
    if data.empty:
        st.error(f"No intraday data for {selected}")
    else:
        hy = float(df.loc[selected, "Yesterday High"].replace(",", ""))
        ly = float(df.loc[selected, "Yesterday Low"].replace(",", ""))
        md = float(df.loc[selected, "Midpoint"].replace(",", ""))
        band = (hy - ly) * 0.1
        # base chart
        fig = plot_series(data, f"{selected} Intraday ({intra_period}, {intra_interval})", intra_type)
        fig.add_hline(y=hy, line=dict(color="green", dash="dash"), annotation_text="High", annotation_position="top left")
        fig.add_hline(y=ly, line=dict(color="red", dash="dash"), annotation_text="Low", annotation_position="bottom left")
        fig.add_hline(y=md, line=dict(color="gray", dash="dot"), annotation_text="Midpt", annotation_position="top right")
        fig.add_vrect(x0=data.index.min(), x1=data.index.max(), y0=md - band/2, y1=md + band/2, fillcolor="gray", opacity=0.2, line_width=0)
        fig.update_layout(hovermode='x unified', margin=dict(l=40, r=40, t=60, b=40))
        st.plotly_chart(fig, use_container_width=True)

def ticker_analysis(tickers, df):
    st.markdown("---")
    selected = st.selectbox("Select ticker to analyze", tickers)
    historical_analysis(selected)
    intra_analysis(selected, df)

def main():
    st.set_page_config(page_title="Box Method", layout="wide")
    st.title("📦 Box Method Trading Zones")
    tickers = get_tickers()
    if not tickers:
        return
    df = get_box_method(tickers)
    ticker_analysis(tickers, df)

if __name__ == "__main__":
    main()
