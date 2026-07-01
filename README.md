# Olist E-Commerce ML Pipeline
**End-to-End Machine Learning on Real Brazilian E-Commerce Data**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![MLflow](https://img.shields.io/badge/MLflow-2.13-orange)
![Power BI](https://img.shields.io/badge/PowerBI-Desktop-yellow)

---

## Project Overview

An end-to-end Business Intelligence and Machine Learning pipeline built on the real **Olist Brazilian E-Commerce** dataset from Kaggle (100,000+ real orders, 2016–2018).

The project covers the full data engineering and ML lifecycle:
- Raw data ingestion from 9 CSV files
- ETL pipeline with data cleaning and transformation
- PostgreSQL star schema data warehouse
- 4 machine learning models with MLflow experiment tracking
- REST API serving analytics and ML predictions
- Interactive Power BI dashboard

---

## Full Pipeline

```
Raw CSV Data (9 files from Kaggle)
        ↓
Python / Pandas — ETL + Data Cleaning
        ↓
PostgreSQL — Star Schema Warehouse (6 tables)
        ↓
Python / SQLAlchemy — Feature Engineering + Analytics
        ↓
ML Models — XGBoost, Gradient Boosting, Random Forest, Prophet
        ↓
MLflow — Experiment Tracking + Model Versioning
        ↓
FastAPI — REST API (10 Analytics + 4 ML endpoints)
        ↓
Power BI — Interactive Dashboard

```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data Processing | Pandas, NumPy |
| Database | PostgreSQL 16 |
| ORM / Queries | SQLAlchemy |
| ML Models | scikit-learn, XGBoost, Prophet / statsmodels |
| Experiment Tracking | MLflow |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Power BI Desktop |
| Version Control | Git, GitHub |
| Package Manager | uv |

---

## ML Models

| Model | Algorithm | Type | Metric |
|---|---|---|---|
| Customer Churn | XGBoost | Binary Classification | ROC-AUC: 1.00 |
| Sales Forecast | Prophet / Holt-Winters | Time Series | MAPE: ~8% |
| Delivery Delay | Gradient Boosting | Binary Classification | ROC-AUC: 0.77 |
| Review Sentiment | Random Forest | Multi-class Classification | Accuracy: 79.6% |

---

## Dataset

**Olist Brazilian E-Commerce** — [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

| File | Description | Rows |
|---|---|---|
| olist_orders_dataset.csv | Core orders — status, timestamps | 99,441 |
| olist_order_items_dataset.csv | Price, freight, seller per item | 112,650 |
| olist_order_payments_dataset.csv | Payment type, installments | 103,886 |
| olist_order_reviews_dataset.csv | Customer review scores | 99,224 |
| olist_customers_dataset.csv | Customer city and state | 99,441 |
| olist_products_dataset.csv | Product category, dimensions | 32,951 |
| olist_sellers_dataset.csv | Seller city and state | 3,095 |
| olist_geolocation_dataset.csv | Zip code lat/lng coordinates | 1,000,163 |
| product_category_name_translation.csv | Portuguese → English categories | 71 |

---

## Folder Structure

```
olist_bi/
├── data/                          ← Place 9 Kaggle CSVs here (not in git)
├── etl/
│   └── etl_pipeline.py            ← Extract → Transform → Load to PostgreSQL
├── analytics/
│   └── analytics.py               ← SQLAlchemy analytics queries
├── ml/
│   ├── train_all.py               ← Train all 4 models at once
│   ├── churn/
│   │   └── churn_model.py         ← XGBoost churn prediction
│   ├── forecast/
│   │   └── sales_forecast.py      ← Prophet / Holt-Winters forecasting
│   ├── delivery/
│   │   └── delay_model.py         ← Gradient Boosting delay prediction
│   └── sentiment/
│       └── sentiment_model.py     ← Random Forest sentiment analysis
├── api/
│   └── main.py                    ← FastAPI analytics + ML endpoints
├── models/                        ← Saved .pkl model files (auto-created)
│   └── .gitkeep
├── mlflow_tracking/               ← MLflow experiment database (auto-created)
│   └── .gitkeep
├── notebooks/
│   ├── 01_eda.ipynb               ← Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_churn_analysis.ipynb
│   ├── 04_forecast_analysis.ipynb
│   ├── 05_delivery_analysis.ipynb
│   └── 06_sentiment_analysis.ipynb
├── config.py                      ← DB + MLflow settings (reads from .env)
├── requirements.txt               ← All dependencies
├── .env                           ← Your credentials (never pushed)
├── .env.example                   ← Safe template
├── .gitignore
├── powerbi_connect.md             ← Power BI setup guide
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.11
- PostgreSQL 16
- Power BI Desktop (free)
- Kaggle account (free)
- uv package manager

### Step 1 — Clone the repository
```bash
git clone https://github.com/Haze45/olist-bi-pipeline.git
cd olist-bi-pipeline
```

### Step 2 — Create virtual environment
```bash
uv venv --python 3.11
.venv\Scripts\activate
```

### Step 3 — Install dependencies
```bash
uv add -r requirements.txt --active
```

### Step 4 — Configure environment
```bash
copy .env.example .env
notepad .env
```

Fill in your PostgreSQL credentials:
```
PG_HOST     = localhost
PG_PORT     = 5432
PG_DATABASE = olist_bi
PG_USER     = postgres
PG_PASSWORD = your_password
```

### Step 5 — Download dataset
```bash
kaggle datasets download -d olistbr/brazilian-ecommerce
Expand-Archive -Path brazilian-ecommerce.zip -DestinationPath data/
```

### Step 6 — Create database
```sql
CREATE DATABASE olist_bi;
```

---

## Running the Project

### Step 7 — Run ETL pipeline
```bash
python etl/etl_pipeline.py
```

Expected output:
```
INFO  EXTRACT: reading 9 Olist CSV files...
INFO  TRANSFORM: cleaning & building star schema...
INFO  LOAD → PostgreSQL localhost:5432/olist_bi
INFO  olist.fact_orders        112,650 rows ✓
INFO  olist.dim_customers       99,441 rows ✓
INFO  olist.dim_products        32,951 rows ✓
INFO  olist.dim_sellers          3,096 rows ✓
INFO  olist.dim_date             1,096 rows ✓
INFO  olist.dim_geolocation     19,023 rows ✓
INFO  Pipeline finished in ~10s 🎉
```

### Step 8 — Test analytics queries
```bash
python analytics/analytics.py
```

### Step 9 — Train all ML models
```bash
python ml/train_all.py
```

Expected output:
```
churn        ✅ Success  (ROC-AUC: 1.00)
forecast     ✅ Success  (MAPE: ~8%)
delivery     ✅ Success  (ROC-AUC: 0.77)
sentiment    ✅ Success  (Accuracy: 79.6%)
```

### Step 10 — Start FastAPI server
```bash
python -m uvicorn api.main:app --reload --port 8000
```

Open: http://localhost:8000/docs

### Step 11 — Start MLflow UI
```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow_tracking/mlflow.db
```

Open: http://localhost:5000

### Step 12 — Open Jupyter Notebooks
```bash
jupyter notebook
```

Navigate to `notebooks/` folder.

---

## API Endpoints

### Analytics Endpoints
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/kpis` | Revenue, orders, delivery rate |
| GET | `/api/revenue/monthly` | Monthly trend + MoM growth % |
| GET | `/api/revenue/yoy` | Year-over-year comparison |
| GET | `/api/categories` | Revenue by product category |
| GET | `/api/products/top` | Top N products by revenue |
| GET | `/api/regions` | Revenue by Brazilian state |
| GET | `/api/payments` | Payment type breakdown |
| GET | `/api/customers/rfm` | RFM customer segments |
| GET | `/api/delivery` | Delivery performance stats |
| GET | `/api/sellers/top` | Top N sellers by revenue |

### ML Prediction Endpoints
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/ml/churn/predict` | Predict customer churn probability |
| POST | `/api/ml/delivery/predict` | Predict if order will be late |
| POST | `/api/ml/sentiment/predict` | Predict review sentiment |
| GET | `/api/ml/forecast?periods=6` | Revenue forecast next N months |

---

## Star Schema

```
dim_date ────────────────────────────────────────────────────┐
dim_customers ──────────────────────────────────┐            │
dim_products ───────────────────────┐           │            │
dim_sellers ──────────┐             │           │            │
                      ↓             ↓           ↓            ↓
              ┌──────────────────────────────────────────────────┐
              │                  fact_orders                      │
              │  fact_id, order_id, order_item_id                 │
              │  seller_id (FK) → dim_sellers                     │
              │  product_id (FK) → dim_products                   │
              │  customer_id (FK) → dim_customers                 │
              │  order_date (FK) → dim_date                       │
              │  price, freight_value, revenue                    │
              │  review_score, order_status, is_delivered         │
              │  delivery_delay_days, payment_type                │
              └──────────────────────────────────────────────────┘
```

---

## Key Business Insights

- **São Paulo (SP)** generates ~30% of all revenue
- **Health & Beauty** and **Watches & Gifts** are top categories by orders
- **Credit card** accounts for ~75% of all payments
- **~30% of orders** are delivered late — key operational issue
- **Champions segment** (top RFM) drives disproportionate revenue
- Clear **upward revenue trend** from 2016 to 2018

---

## Notebooks

| Notebook | Description |
|---|---|
| 01_eda.ipynb | Full dataset exploration — KPIs, trends, distributions |
| 02_feature_engineering.ipynb | RFM calculation, correlation analysis, churn labeling |
| 03_churn_analysis.ipynb | XGBoost model — confusion matrix, ROC curve, feature importance |
| 04_forecast_analysis.ipynb | Revenue forecast — actual vs predicted, seasonality |
| 05_delivery_analysis.ipynb | Delay patterns by state, month, model performance |
| 06_sentiment_analysis.ipynb | Sentiment distribution, by category, model results |

---


---

## Daily Usage

```bash
# Terminal 1 — Start API
python -m uvicorn api.main:app --reload --port 8000

# Terminal 2 — Start MLflow UI
python -m mlflow ui --backend-store-uri sqlite:///mlflow_tracking/mlflow.db

# Then open Power BI Desktop and refresh data
```

---

