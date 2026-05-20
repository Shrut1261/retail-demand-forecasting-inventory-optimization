from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
FORECASTS = ROOT / "data" / "processed" / "forecast_results.csv"
RECOMMENDATIONS = ROOT / "data" / "processed" / "inventory_recommendations.csv"
METRICS = ROOT / "data" / "processed" / "model_metrics.csv"

st.set_page_config(page_title="Retail Demand Intelligence", page_icon="R", layout="wide")
st.title("Retail Demand Forecasting and Inventory Optimization")
st.caption("Forecast demand, quantify stockout risk, and prioritize replenishment actions.")

if not FORECASTS.exists():
    st.warning("Run `python main.py` first to generate processed data.")
    st.stop()

forecasts = pd.read_csv(FORECASTS, parse_dates=["date"])
recommendations = pd.read_csv(RECOMMENDATIONS)
metrics = pd.read_csv(METRICS)

top1, top2, top3, top4 = st.columns(4)
top1.metric("Forecasted Units", f"{forecasts['forecast_units'].sum():,.0f}")
top2.metric("Reorder Value", f"${recommendations['reorder_value'].sum():,.0f}")
top3.metric("Urgent Reorders", f"{(recommendations['priority'] == 'Urgent Reorder').sum():,.0f}")
top4.metric("Best Model MAE", f"{metrics.iloc[0]['mae']:.2f}")

store = st.sidebar.selectbox("Store", sorted(forecasts["store"].unique()))
sku = st.sidebar.selectbox("SKU", sorted(forecasts["sku"].unique()))
filtered = forecasts[(forecasts["store"] == store) & (forecasts["sku"] == sku)].sort_values("date")

left, right = st.columns([1.35, 1])
with left:
    fig = px.line(filtered, x="date", y=["units_sold", "forecast_units"], title=f"Actual vs Forecast Demand: {store} / {sku}")
    fig.update_layout(legend_title_text="", yaxis_title="Units", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Model Leaderboard")
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.subheader("Category Stockout Risk")
    risk = recommendations.groupby("category", as_index=False)["stockout_risk"].mean()
    st.plotly_chart(px.bar(risk, x="category", y="stockout_risk", color="category"), use_container_width=True)

st.subheader("Replenishment Action Queue")
priority = st.multiselect("Priority", sorted(recommendations["priority"].dropna().unique()), default=sorted(recommendations["priority"].dropna().unique()))
queue = recommendations[recommendations["priority"].isin(priority)]
st.dataframe(
    queue[["store", "sku", "category", "current_inventory", "reorder_point", "recommended_order_qty", "stockout_risk", "reorder_value", "priority"]],
    use_container_width=True,
    hide_index=True,
)
