import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def inputs():
    initial    = st.number_input("Initial amount", min_value=0, value=1_000, step=100)
    monthly    = st.number_input("Monthly contribution", min_value=0, value=0, step=50)
    years      = st.number_input("Number of years", min_value=1, value=5, step=1)
    rate       = st.number_input("Estimated annual rate (%)", min_value=0.0, value=20.0, step=0.1) / 100
    variance   = st.number_input("Interest‑rate variance (%)", min_value=0.0, value=5.0, step=0.1) / 100
    return initial, monthly, years, rate, variance

def calculate_display(initial, monthly, years, rate, variance):
    yrs = np.arange(years + 1)
    rates = {
        "Lower Bound": rate - variance,
        "Nominal":     rate,
        "Upper Bound": rate + variance
    }
    totals = {}
    for label, r in rates.items():
        # yearly growth of initial
        growth = initial * (1 + r) ** yrs
        # monthly rate for contributions
        m_r = r / 12 if r != 0 else 0
        fv_contrib = np.array([
            monthly * (((1 + m_r) ** (12*y) - 1) / m_r) if y>0 and m_r>0 else monthly * 12 * y
            for y in yrs
        ])
        totals[label] = growth + fv_contrib

    df_totals = pd.DataFrame(totals, index=yrs)
    df_totals.index.name = "Year"

    # line chart with contributions included
    df_plot = df_totals.reset_index().melt(id_vars="Year", value_name="Value", var_name="Scenario")
    fig_line = px.line(
        df_plot, x="Year", y="Value", color="Scenario",
        markers=True,
        title="Projected Value Over Time (with monthly contrib)"
    )
    fig_line.update_layout(hovermode="x unified")
    st.plotly_chart(fig_line, use_container_width=True)

    principal_total = initial + monthly * 12 * years
    final_nominal   = totals["Nominal"][-1]
    interest_total  = final_nominal - principal_total
    pie_df = pd.DataFrame({
        "Category": ["Principal", "Interest"],
        "Amount":   [principal_total, interest_total]
    })
    fig_pie = px.pie(
        pie_df, names="Category", values="Amount",
        title=f"Final Breakdown after {years} years"
    )
    fig_pie.update_traces(textinfo="label+percent+value")
    st.plotly_chart(fig_pie, use_container_width=True)

    st.dataframe(df_totals.style.format("{:,.2f}"), use_container_width=True)

def main():
    st.set_page_config(page_title="Annual Compound Interest Calculator", layout="wide")
    st.title("💰 Annual Compound Interest Calculator")
    initial, monthly, years, rate, variance = inputs()
    calculate_display(initial, monthly, years, rate, variance)

if __name__ == "__main__":
    main()
