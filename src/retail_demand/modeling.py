import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import model_matrix


def train_models(features_df: pd.DataFrame):
    features, categorical = model_matrix(features_df)
    split_date = features_df["date"].max() - pd.Timedelta(days=60)
    train = features_df[features_df["date"] <= split_date]
    test = features_df[features_df["date"] > split_date]

    X_train, y_train = train[features + categorical], train["units_sold"]
    X_test, y_test = test[features + categorical], test["units_sold"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    candidates = {
        "lag_7_baseline": None,
        "ridge_regression": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=160, max_depth=14, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }

    metrics = []
    forecasts = test[["date", "store", "sku", "category", "units_sold", "inventory_on_hand", "price"]].copy()
    best_name, best_model, best_mae = None, None, np.inf

    for name, model in candidates.items():
        if model is None:
            pred = np.clip(test["lag_7"].to_numpy(), 0, None)
            pipeline = None
        else:
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
            pipeline.fit(X_train, y_train)
            pred = np.clip(pipeline.predict(X_test), 0, None)

        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mape = mean_absolute_percentage_error(y_test.replace(0, 1), pred)
        metrics.append({"model": name, "mae": round(mae, 3), "rmse": round(rmse, 3), "mape": round(mape, 4)})
        forecasts[f"pred_{name}"] = pred.round(2)

        if mae < best_mae and pipeline is not None:
            best_name, best_model, best_mae = name, pipeline, mae

    forecasts["best_model"] = best_name
    forecasts["forecast_units"] = forecasts[f"pred_{best_name}"]
    return pd.DataFrame(metrics).sort_values("mae"), forecasts, best_model
