import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def inputs():
    initial = st.number_input("Initial amount", min_value=0, value=1000, step=100)
    years   = st.number_input("Number of years", min_value=1, value=5, step=1)
    rate    = st.number_input("Estimated annual rate (%)", min_value=0.0, value=20.0, step=0.1) / 100
    variance= st.number_input("Interest‑rate variance (%)", min_value=0.0, value=5.0, step=0.1) / 100
    return initial, years, rate, variance

def calculate_display(initial, years, rate, variance):
    years_list = np.arange(years + 1)
    lower  = initial * (1 + rate - variance) ** years_list
    nominal= initial * (1 + rate)         ** years_list
    upper  = initial * (1 + rate + variance) ** years_list

    df = pd.DataFrame({
        "Lower Bound": lower,
        "Nominal":     nominal,
        "Upper Bound": upper
    }, index=years_list)
    df.index.name = "Year"

    df_plot = df.reset_index()
    fig = px.line(
        df_plot,
        x="Year",
        y=["Lower Bound", "Nominal", "Upper Bound"],
        markers=True,
        title="Projected Value Over Time"
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df.style.format("{:,.2f}"), use_container_width=True)

def main():
    st.set_page_config(page_title="Annual Compound Interest Calculator", layout="wide")
    st.title("💰 Annual Compound Interest Calculator")
    initial, years, rate, variance = inputs()
    calculate_display(initial, years, rate, variance)

if __name__ == "__main__":
    main()
