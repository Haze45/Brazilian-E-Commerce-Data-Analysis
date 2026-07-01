"""
api/main.py  —  Olist BI + ML API
===================================
Run: uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from analytics.analytics import (
    kpi_summary, monthly_revenue, category_performance,
    top_products, state_performance, payment_analysis,
    rfm_segmentation, delivery_performance,
    seller_leaderboard, yoy_comparison,
)

app = FastAPI(
    title="Olist BI + ML API",
    description="Analytics + Machine Learning endpoints for the Olist E-Commerce Pipeline.",
    version="2.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/",                       tags=["health"])
def root():
    return {"status": "ok", "version": "2.0.0", "dataset": "Olist Brazilian E-Commerce"}

@app.get("/api/kpis",               tags=["analytics"])
def get_kpis():
    return kpi_summary()

@app.get("/api/revenue/monthly",    tags=["analytics"])
def get_monthly():
    return monthly_revenue()

@app.get("/api/revenue/yoy",        tags=["analytics"])
def get_yoy():
    return yoy_comparison()

@app.get("/api/categories",         tags=["analytics"])
def get_categories():
    return category_performance()

@app.get("/api/products/top",       tags=["analytics"])
def get_top_products(n: int = Query(10, ge=1, le=50)):
    return top_products(n)

@app.get("/api/regions",            tags=["analytics"])
def get_regions():
    return state_performance()

@app.get("/api/payments",           tags=["analytics"])
def get_payments():
    return payment_analysis()

@app.get("/api/customers/rfm",      tags=["analytics"])
def get_rfm():
    return rfm_segmentation()

@app.get("/api/delivery",           tags=["analytics"])
def get_delivery():
    return delivery_performance()

@app.get("/api/sellers/top",        tags=["analytics"])
def get_sellers(n: int = Query(10, ge=1, le=30)):
    return seller_leaderboard(n)


class ChurnRequest(BaseModel):
    recency_days      : float
    frequency         : float
    monetary          : float
    avg_review_score  : float = 4.0
    avg_freight_pct   : float = 0.15
    avg_installments  : float = 1.0
    late_deliveries   : float = 0.0
    avg_delay_days    : float = 0.0
    unique_categories : float = 1.0
    unique_products   : float = 1.0

class DeliveryRequest(BaseModel):
    price                : float
    freight_value        : float
    order_month          : int
    order_quarter        : int
    payment_installments : float = 1.0
    customer_state       : str   = "SP"
    seller_state         : str   = "SP"
    weight_g             : float = 500.0
    length_cm            : float = 20.0
    height_cm            : float = 10.0
    width_cm             : float = 15.0

class SentimentRequest(BaseModel):
    price                : float
    freight_value        : float
    delivery_delay_days  : float = 0.0
    payment_installments : float = 1.0
    is_delivered         : int   = 1
    order_month          : int   = 6
    order_quarter        : int   = 2


@app.post("/api/ml/churn/predict",     tags=["ml"])
def predict_churn(req: ChurnRequest):
    """Predict if a customer will churn based on their purchase history."""
    try:
        from ml.churn.churn_model import predict
        return predict(req.dict())
    except FileNotFoundError:
        return {"error": "Model not trained yet. Run: python ml/train_all.py"}


@app.post("/api/ml/delivery/predict",  tags=["ml"])
def predict_delivery(req: DeliveryRequest):
    """Predict if an order will be delivered late."""
    try:
        from ml.delivery.delay_model import predict
        return predict(req.dict())
    except FileNotFoundError:
        return {"error": "Model not trained yet. Run: python ml/train_all.py"}


@app.post("/api/ml/sentiment/predict", tags=["ml"])
def predict_sentiment(req: SentimentRequest):
    """Predict review sentiment (Positive/Neutral/Negative)."""
    try:
        from ml.sentiment.sentiment_model import predict
        return predict(req.dict())
    except FileNotFoundError:
        return {"error": "Model not trained yet. Run: python ml/train_all.py"}


@app.get("/api/ml/forecast",           tags=["ml"])
def get_forecast(periods: int = Query(6, ge=1, le=12)):
    """Get revenue forecast for next N months."""
    try:
        from ml.forecast.sales_forecast import predict
        return predict(periods)
    except FileNotFoundError:
        return {"error": "Model not trained yet. Run: python ml/train_all.py"}
