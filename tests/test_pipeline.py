from src.retail_demand.data import generate_retail_sales
from src.retail_demand.features import build_features
from src.retail_demand.inventory import build_inventory_recommendations


def test_generate_retail_sales_has_expected_columns():
    df = generate_retail_sales()
    required = {"date", "store", "sku", "category", "price", "promotion", "units_sold", "inventory_on_hand", "revenue"}
    assert required.issubset(df.columns)
    assert len(df) > 10000


def test_feature_pipeline_creates_lag_columns():
    features = build_features(generate_retail_sales())
    assert {"lag_7", "rolling_7_mean", "rolling_28_mean"}.issubset(features.columns)
    assert features["lag_7"].isna().sum() == 0


def test_inventory_recommendations_are_non_negative():
    df = generate_retail_sales()
    features = build_features(df)
    sample = features.tail(500).copy()
    sample["forecast_units"] = sample["rolling_7_mean"]
    recommendations = build_inventory_recommendations(sample)
    assert (recommendations["recommended_order_qty"] >= 0).all()
    assert (recommendations["stockout_risk"].between(0, 1)).all()
