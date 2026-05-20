import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["store", "sku", "date"])
    data["day_of_week"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month
    data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)

    group = data.groupby(["store", "sku"])["units_sold"]
    data["lag_7"] = group.shift(7)
    data["lag_14"] = group.shift(14)
    data["rolling_7_mean"] = group.transform(lambda s: s.shift(1).rolling(7).mean())
    data["rolling_28_mean"] = group.transform(lambda s: s.shift(1).rolling(28).mean())
    data["rolling_28_std"] = group.transform(lambda s: s.shift(1).rolling(28).std())
    data["price_change"] = data.groupby(["store", "sku"])["price"].pct_change().fillna(0)
    data = data.dropna().reset_index(drop=True)
    return data


def model_matrix(df: pd.DataFrame):
    features = [
        "price",
        "promotion",
        "holiday",
        "weather_index",
        "inventory_on_hand",
        "day_of_week",
        "month",
        "week_of_year",
        "is_weekend",
        "lag_7",
        "lag_14",
        "rolling_7_mean",
        "rolling_28_mean",
        "rolling_28_std",
        "price_change",
    ]
    categorical = ["store", "sku", "category"]
    return features, categorical
