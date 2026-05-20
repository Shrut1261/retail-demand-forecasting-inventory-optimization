# Retail Demand Forecasting and Inventory Optimization

Industrial analytics project for retail and supply-chain decision support. The system forecasts SKU-store demand, estimates replenishment risk, and recommends reorder quantities using service-level inventory logic.

## Business Problem

Retail teams lose margin when inventory planning is too reactive: stockouts reduce revenue, overstock increases holding cost, and promotion demand is difficult to anticipate. This project turns transactional sales data into a forecasting and inventory control workflow that helps planners decide what to reorder, where, and why.

## What This Project Delivers

- Demand forecasting pipeline for store-SKU daily sales
- Feature engineering for seasonality, promotions, price, holidays, weather, and lag demand
- Model comparison using baseline, linear regression, random forest, and gradient boosting
- Inventory optimization using safety stock, reorder points, and recommended order quantity
- Executive Streamlit dashboard for KPIs, forecast curves, stockout risk, and replenishment actions
- Automated tests and GitHub Actions CI for project reliability

## Project Structure

```text
retail-demand-forecasting-inventory-optimization/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── reports/
│   └── figures/
├── src/
│   └── retail_demand/
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── inventory.py
│       ├── modeling.py
│       └── reporting.py
├── tests/
├── main.py
├── requirements.txt
└── README.md
```

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
streamlit run app/streamlit_app.py
```

## Key Outputs

Running `python main.py` creates:

- `data/raw/retail_daily_sales.csv`
- `data/processed/model_metrics.csv`
- `data/processed/forecast_results.csv`
- `data/processed/inventory_recommendations.csv`
- `reports/figures/demand_forecast.png`
- `reports/figures/category_stockout_risk.png`
- `reports/figures/reorder_value_by_store.png`

## Portfolio Value

This project demonstrates applied analytics for the retail and supply-chain industry: forecasting, operations planning, inventory optimization, dashboarding, and business translation.

