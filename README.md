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

- Built end-to-end ETL pipeline processing 100K+ real Brazilian
  e-commerce orders from 9 CSV sources into a PostgreSQL star schema
  warehouse (6 tables, 112K+ rows) using Python and Pandas

- Designed star schema data warehouse with fact + 4 dimension tables +
  geolocation; created 8 query indexes reducing execution time by ~60%

- Trained XGBoost churn model (ROC-AUC 0.76) on 96K customers — identified
  and resolved data leakage caused by target-derived feature (recency_days)
  that was producing a misleading ROC-AUC of 1.00

- Applied SMOTE oversampling + threshold tuning to Gradient Boosting
  delivery delay classifier, improving Late class recall from 2% to 55%+
  on a severely imbalanced dataset (93:7 class ratio)

- Built Prophet revenue forecasting model with monthly seasonality and
  95% confidence intervals; cross-validated on 24 months of time-series data

- Tracked all ML experiments with MLflow — parameters, metrics, and model
  artifacts versioned across 4 models and multiple training runs

- Exposed 14 REST API endpoints (10 analytics + 4 ML predictions) via
  FastAPI with Pydantic validation and auto-generated Swagger documentation

- Built interactive Power BI dashboard connected live to PostgreSQL with
  KPI cards, revenue trend chart, geolocation map, and slicers for
  year, category, and state filtering

- Performed RFM customer segmentation identifying 5 cohorts (Champions,
  Loyal, Potential, At Risk, Lost) across 96K+ unique customers

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

| Model | Algorithm | Type | Metric | Notes |
|---|---|---|---|---|
| Customer Churn | XGBoost | Binary Classification | ROC-AUC: 0.76 | Fixed data leakage — removed recency_days |
| Sales Forecast | Prophet | Time Series | MAPE: ~15-25% | Monthly seasonality, 95% CI |
| Delivery Delay | Gradient Boosting / RF / XGBoost | Binary Classification | Late Recall: ~55% | Fixed class imbalance with SMOTE |
| Review Sentiment | Random Forest | Binary Classification | Macro F1: ~0.72 | 2-class: Positive vs Not Positive |


### Key ML Challenges Identified and Fixed
 
**1. Data Leakage — Churn Model**
- Problem: `recency_days` was used as a feature while also defining the churn target (`recency_days > 180`), producing a misleading ROC-AUC of 1.00
- Fix: Dropped `recency_days` from features entirely, added behavioral features instead
- Result: ROC-AUC dropped to honest 0.76
**2. Class Imbalance — Delivery Model**
- Problem: 93.4% On-time vs 6.6% Late — model predicted everything as On-time, Late recall was only 2%
- Fix: SMOTE oversampling + `compute_sample_weight` + threshold tuning targeting Late recall ≥ 55%
- Result: Late recall improved from 2% to 55%+
**3. Neutral Class Never Predicted — Sentiment Model**
- Problem: 3-class model (Positive/Neutral/Negative) ignored Neutral (8.3% of data)
- Fix: Merged Neutral into Not Positive (2-class), added interaction features
- Result: Macro F1 improved from 0.44 to ~0.72
**4. Forecast MAPE 48,508% — Forecast Model**
- Problem: Yearly seasonality unreliable with only 24 months of data
- Fix: Disabled yearly seasonality, added manual monthly seasonality instead
- Result: MAPE dropped to ~15-25%, no negative revenue forecasts
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
## Files Created After Running
 
```
models/
├── churn_model.pkl          ← XGBoost churn model
├── churn_scaler.pkl         ← StandardScaler for churn features
├── forecast_model.json      ← Prophet model (JSON — not pkl)
├── forecast_results.csv     ← Full forecast with confidence intervals
├── delivery_model.pkl       ← Best of GB / RF / XGBoost
├── delivery_le_cust.pkl     ← LabelEncoder for customer states
├── delivery_le_seller.pkl   ← LabelEncoder for seller states
├── sentiment_model.pkl      ← Random Forest sentiment model
└── sentiment_le.pkl         ← LabelEncoder for sentiment classes
 
mlflow_tracking/
└── mlflow.db                ← All experiment logs and metrics
 
mlruns/                      ← MLflow artifact store (auto-created)
```

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

