from retail_demand.config import FIGURES, FORECAST_RESULTS, INVENTORY_RECOMMENDATIONS, MODEL_METRICS, RAW_SALES
from retail_demand.data import save_raw_sales
from retail_demand.features import build_features
from retail_demand.inventory import build_inventory_recommendations
from retail_demand.modeling import train_models
from retail_demand.reporting import create_figures


def main():
    sales = save_raw_sales(RAW_SALES)
    features = build_features(sales)
    metrics, forecasts, _ = train_models(features)
    recommendations = build_inventory_recommendations(forecasts)

    MODEL_METRICS.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(MODEL_METRICS, index=False)
    forecasts.to_csv(FORECAST_RESULTS, index=False)
    recommendations.to_csv(INVENTORY_RECOMMENDATIONS, index=False)
    create_figures(forecasts, recommendations, FIGURES)

    best = metrics.iloc[0]
    urgent = int((recommendations["priority"] == "Urgent Reorder").sum())
    print(f"Best model: {best['model']} | MAE: {best['mae']} | RMSE: {best['rmse']}")
    print(f"Urgent reorder actions: {urgent}")
    print(f"Total recommended reorder value: ${recommendations['reorder_value'].sum():,.0f}")


if __name__ == "__main__":
    main()
