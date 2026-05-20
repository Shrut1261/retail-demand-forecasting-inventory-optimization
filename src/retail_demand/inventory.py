import numpy as np
import pandas as pd


def build_inventory_recommendations(forecasts: pd.DataFrame, service_level_z: float = 1.65, lead_time_days: int = 7) -> pd.DataFrame:
    recent = forecasts.sort_values("date").groupby(["store", "sku", "category"]).tail(28)
    grouped = recent.groupby(["store", "sku", "category"]).agg(
        avg_daily_forecast=("forecast_units", "mean"),
        demand_std=("forecast_units", "std"),
        current_inventory=("inventory_on_hand", "last"),
        unit_price=("price", "mean"),
    ).reset_index()

    grouped["demand_std"] = grouped["demand_std"].fillna(0)
    grouped["lead_time_demand"] = grouped["avg_daily_forecast"] * lead_time_days
    grouped["safety_stock"] = service_level_z * grouped["demand_std"] * np.sqrt(lead_time_days)
    grouped["reorder_point"] = grouped["lead_time_demand"] + grouped["safety_stock"]
    grouped["recommended_order_qty"] = np.clip(grouped["reorder_point"] - grouped["current_inventory"], 0, None)
    grouped["stockout_risk"] = np.clip((grouped["reorder_point"] - grouped["current_inventory"]) / grouped["reorder_point"].replace(0, np.nan), 0, 1).fillna(0)
    grouped["reorder_value"] = grouped["recommended_order_qty"] * grouped["unit_price"]
    grouped["priority"] = pd.cut(
        grouped["stockout_risk"],
        bins=[-0.01, 0.25, 0.6, 1.0],
        labels=["Monitor", "Plan Replenishment", "Urgent Reorder"],
    )
    numeric_cols = ["avg_daily_forecast", "demand_std", "lead_time_demand", "safety_stock", "reorder_point", "recommended_order_qty", "stockout_risk", "reorder_value"]
    grouped[numeric_cols] = grouped[numeric_cols].round(2)
    return grouped.sort_values(["stockout_risk", "reorder_value"], ascending=False)
