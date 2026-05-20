import matplotlib.pyplot as plt
import pandas as pd


def create_figures(forecasts: pd.DataFrame, recommendations: pd.DataFrame, output_dir) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = forecasts[(forecasts["store"] == "Providence") & (forecasts["sku"] == "SKU-1001")].sort_values("date")

    plt.figure(figsize=(11, 5))
    plt.plot(sample["date"], sample["units_sold"], label="Actual demand", linewidth=2)
    plt.plot(sample["date"], sample["forecast_units"], label="Forecast demand", linewidth=2)
    plt.title("Demand Forecast: Providence / SKU-1001")
    plt.xlabel("Date")
    plt.ylabel("Units")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "demand_forecast.png", dpi=160)
    plt.close()

    category_risk = recommendations.groupby("category")["stockout_risk"].mean().sort_values()
    plt.figure(figsize=(9, 5))
    category_risk.plot(kind="barh", color="#0563bb")
    plt.title("Average Stockout Risk by Category")
    plt.xlabel("Stockout risk")
    plt.tight_layout()
    plt.savefig(output_dir / "category_stockout_risk.png", dpi=160)
    plt.close()

    reorder_value = recommendations.groupby("store")["reorder_value"].sum().sort_values(ascending=False)
    plt.figure(figsize=(9, 5))
    reorder_value.plot(kind="bar", color="#22c55e")
    plt.title("Recommended Reorder Value by Store")
    plt.ylabel("Reorder value")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "reorder_value_by_store.png", dpi=160)
    plt.close()
