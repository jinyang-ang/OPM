import warnings
from datetime import datetime, timedelta, time
import pytz
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


def get_valuation_metrics(T):
    info = T.info
    pe = info.get('trailingPE') or info.get('forwardPE')
    growth = info.get('earningsQuarterlyGrowth') or info.get('earningsGrowth')
    peg = None
    if pe and growth and growth != 0:
        peg = pe / (growth * 100 if growth < 2 else growth)
    ev = info.get('enterpriseValue')
    ebitda = info.get('ebitda')
    ev_ebitda = ev / ebitda if ev and ebitda else None
    fcf = info.get('freeCashflow')
    ev_fcf = ev / fcf if ev and fcf else None

    # Display
    st.subheader("Valuation Metrics")
    cols = st.columns(5)
    cols[0].metric("P/E", f"{pe:.1f}" if pe else "N/A")
    cols[1].metric("PEG", f"{peg:.2f}" if peg else "N/A")
    cols[2].metric("EV/EBITDA", f"{ev_ebitda:.1f}" if ev_ebitda else "N/A")
    cols[3].metric("EV/FCF", f"{ev_fcf:.1f}" if ev_fcf else "N/A")


def get_stock_data(T):

    hist = T.history(period="2d", interval="1d")
    price      = hist["Close"].iloc[-1]
    prev_close = hist["Close"].iloc[-2]
    delta_pct  = (price - prev_close) / prev_close * 100

    st.metric(
        label="Current Price",
        value=f"{price:.2f}",
        delta=f"{delta_pct:.2f}%"
    )

    earnings = get_earnings(T)
    get_valuation_metrics(T)

    c1, c2, c3 = st.columns(3)
    with c1:
        hist_period = st.selectbox("Historical period", ["1mo","3mo","6mo","1y","5y","max"], index=1)
    with c2:
        hist_interval = st.selectbox("Historical interval", ["1d","1wk","1mo"], index=0)
    with c3:
        hist_type = st.selectbox("Chart type", ["Line","Candlestick"], index=0, key="historical")

    # fetch and plot
    data = T.history(period=hist_period, interval=hist_interval)
    if data.empty:
        st.error(f"No historical data for {T.ticker}")
    else:
        data.index.name = 'Date'
        fig = plot_series(
            data,
            f"{T.ticker} Historical ({hist_period}, {hist_interval})",
            hist_type,
            current_price=price 
        )
        st.plotly_chart(fig, use_container_width=True)

    vol_fig = go.Figure(
        data=[go.Bar(x=data.index, y=data['Volume'], name='Volume')]
    )
    vol_fig.update_layout(
        title=f"{T.ticker} Trading Volume ({hist_period}, {hist_interval})",
        xaxis_title='Date',
        yaxis_title='Volume',
        hovermode='x unified',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(vol_fig, use_container_width=True)

    return price, earnings


def get_news(T, first_n=3):
    raw_news = T.news or []
    if not raw_news:
        st.info("No recent news available for this ticker.")
        return

    st.subheader("📰 Latest News")
    sg_tz = pytz.timezone("Asia/Singapore")

    def render_article(article):
        content = article.get("content", {})
        title    = content.get("title", "Untitled")
        link     = (content.get("clickThroughUrl") or {}).get("url") \
                or (content.get("canonicalUrl") or {}).get("url") \
                or "#"
        provider = content.get("provider", {}).get("displayName", "Unknown")
        ts_str   = content.get("pubDate") or content.get("displayTime")
        summary  = content.get("summary") or content.get("description") or ""

        # parse and format timestamp
        if ts_str:
            ts = pd.to_datetime(ts_str)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            ts = ts.tz_convert(sg_tz)
            ts_fmt = ts.strftime("%Y-%m-%d %I:%M %p %Z")
            meta = f"_{provider} — {ts_fmt}_"
        else:
            meta = f"_{provider}_"

        st.markdown(f"#### [{title}]({link})")
        st.write(meta)
        st.write(summary)
        st.markdown("---")

    for article in raw_news[:first_n]:
        render_article(article)

    if len(raw_news) > first_n:
        with st.expander(f"Show {len(raw_news)-first_n} more articles"):
            for article in raw_news[first_n:]:
                render_article(article)

def get_earnings(T):
    
    cal = T.calendar

    if hasattr(cal, "index") and "Earnings Date" in cal.index:
        raw = cal.loc["Earnings Date"].values[0]
    elif isinstance(cal, dict) and cal.get("Earnings Date"):
        raw = cal["Earnings Date"][0]
    else:
        raw = None

    if raw is not None:
        if isinstance(raw, (list, tuple, np.ndarray)) and raw:
            ts = raw[0]
        else:
            ts = raw
        ts = pd.to_datetime(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        et = ts.tz_convert("US/Eastern")
        t = et.timetz()
        if t < time(9, 30):
            session = "Pre‑market"
        elif t >= time(16, 0):
            session = "Post‑market"
        else:
            session = "Market hours"
        sg_now = datetime.now(pytz.timezone("Asia/Singapore"))
        sg_ts = et.astimezone(pytz.timezone("Asia/Singapore"))
        days_to_er = (sg_ts.date() - sg_now.date()).days
        next_er = sg_ts.date()
        st.info(
            f"**{et.strftime('%Y-%m-%d %I:%M %p %Z')}** {session} (in {days_to_er} days)"
        )
    else:
        next_er = None
        st.info("No upcoming earnings date available.")
    return next_er

def get_expiry(T, next_er):
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
    expiry = st.selectbox(
        "Expiry",
        [c[0] for c in choices] or [default],
        format_func=lambda d: dict(choices).get(d, d)
    )
    return expiry

def formulate_df(T, current_price, expiry):
    chain = T.option_chain(expiry)
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


def display(subset, current_price, expiry):
    subset["Breakeven Call"] = subset.strike + subset.market_premium_call
    subset["Breakeven Put"]  = subset.strike - subset.market_premium_put

    display_df = subset[[
        "ask_call",
        "bid_call",
        "volume_call",
        "Breakeven Call",
        "market_premium_call",
        "strike",
        "market_premium_put",
        "Breakeven Put",
        "volume_put",
        "bid_put",
        "ask_put",
        
    ]].rename(columns={
        "ask_call":           "Call Ask",
        "bid_call":           "Call Bid",
        "volume_call":        "Call Volume",
        "market_premium_call":"Call Premium",
        "strike":             "Strike",
        "market_premium_put": "Put Premium",
        "volume_put":         "Put Volume",
        "bid_put":            "Put Bid",
        "ask_put":            "Put Ask",
    })

    for col in ["Call Bid", "Call Ask", "Put Ask", "Put Bid"]:
        display_df[col] = display_df[col].map("${:,.2f}".format)
    for col in ["Call Volume", "Put Volume"]:
        display_df[col] = display_df[col].map("{:,}".format)
    display_df["Call Premium"]    = display_df["Call Premium"].map("${:,.2f}".format)
    display_df["Put Premium"]     = display_df["Put Premium"].map("${:,.2f}".format)
    display_df["Strike"]          = display_df["Strike"].map("${:,.2f}".format)
    display_df["Breakeven Call"]  = display_df["Breakeven Call"].map("${:,.2f}".format)
    display_df["Breakeven Put"]   = display_df["Breakeven Put"].map("${:,.2f}".format)

    atm_idx = (subset.strike - current_price).abs().idxmin()
    styled = (
        display_df.style
        .apply(
            lambda col: ['background-color: lightgray'] * len(col)
                        if col.name == "Strike" else [''] * len(col),
            axis=0
        )
        .apply(
            lambda row: ['background-color: lightblue' if row.name == atm_idx else '' for _ in row],
            axis=1
        )
    )

    st.subheader(f"Option Premium & Volume @ {expiry}")
    height = display_df.shape[0] * 35 + 40
    st.dataframe(styled, height=height, use_container_width=True)

def main():
    st.set_page_config(page_title="Option Analysis", layout="wide")
    st.title("📝 Sell-Only Option Analysis")
    ticker = st.text_input("Ticker", "PLTR").upper()
    T = yf.Ticker(ticker)
    price, earnings = get_stock_data(T)
    if price is None:
        return
    get_news(T)
    expiry = get_expiry(T, earnings)
    subset = formulate_df(T, price, expiry)
    display(subset, price, expiry)

if __name__ == "__main__":
    main()
