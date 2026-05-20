import numpy as np
import pandas as pd


def generate_retail_sales(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", "2025-04-30", freq="D")
    stores = ["Providence", "Boston", "Hartford", "New Haven", "Worcester"]
    products = [
        ("SKU-1001", "Grocery", 4.99, 120),
        ("SKU-1002", "Grocery", 8.49, 95),
        ("SKU-2001", "Health", 12.99, 55),
        ("SKU-2002", "Health", 18.99, 42),
        ("SKU-3001", "Electronics", 79.99, 12),
        ("SKU-3002", "Electronics", 129.99, 8),
        ("SKU-4001", "Home", 24.99, 34),
        ("SKU-4002", "Home", 39.99, 26),
    ]
    rows = []

    for store_idx, store in enumerate(stores):
        store_multiplier = 0.82 + store_idx * 0.08 + rng.normal(0, 0.03)
        for sku, category, base_price, base_demand in products:
            category_multiplier = {"Grocery": 1.15, "Health": 0.9, "Electronics": 0.72, "Home": 0.82}[category]
            for date in dates:
                weekday_boost = 1.18 if date.dayofweek in [4, 5, 6] else 0.94
                seasonal = 1 + 0.16 * np.sin(2 * np.pi * date.dayofyear / 365)
                holiday = int(date.month == 12 or date.strftime("%m-%d") in ["07-04", "11-29"])
                promo = int(rng.random() < (0.11 if category in ["Grocery", "Health"] else 0.07))
                price = round(base_price * (0.9 if promo else 1.0) * rng.normal(1, 0.015), 2)
                weather_index = np.clip(rng.normal(0.55 + 0.18 * np.sin(2 * np.pi * (date.dayofyear + 40) / 365), 0.12), 0, 1)
                demand_mean = base_demand * store_multiplier * category_multiplier * weekday_boost * seasonal
                demand_mean *= 1.22 if promo else 1
                demand_mean *= 1.18 if holiday else 1
                demand_mean *= 1 + (0.1 * weather_index if category in ["Grocery", "Home"] else -0.04 * weather_index)
                units = max(0, int(rng.normal(demand_mean, max(2, demand_mean * 0.18))))
                on_hand = max(0, int(rng.normal(base_demand * 5, base_demand * 1.3)) - int(units * rng.uniform(0.2, 1.1)))
                rows.append({
                    "date": date,
                    "store": store,
                    "sku": sku,
                    "category": category,
                    "price": price,
                    "promotion": promo,
                    "holiday": holiday,
                    "weather_index": round(weather_index, 3),
                    "units_sold": units,
                    "inventory_on_hand": on_hand,
                    "revenue": round(units * price, 2),
                })

    return pd.DataFrame(rows)


def save_raw_sales(path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_retail_sales()
    df.to_csv(path, index=False)
    return df
