from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"

RAW_SALES = DATA_RAW / "retail_daily_sales.csv"
MODEL_METRICS = DATA_PROCESSED / "model_metrics.csv"
FORECAST_RESULTS = DATA_PROCESSED / "forecast_results.csv"
INVENTORY_RECOMMENDATIONS = DATA_PROCESSED / "inventory_recommendations.csv"

RANDOM_STATE = 42
