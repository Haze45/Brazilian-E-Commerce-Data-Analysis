"""
ml/train_all.py
================
Run all ML model training pipelines in sequence.

Usage:
    python ml/train_all.py
"""

import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml.churn.churn_model         import train as train_churn
from ml.forecast.sales_forecast   import train as train_forecast
from ml.delivery.delay_model      import train as train_delivery
from ml.sentiment.sentiment_model import train as train_sentiment


def train_all():
    t0 = time.time()
    print("=" * 55)
    print("  OLIST ML PIPELINE — TRAINING ALL MODELS")
    print("=" * 55)

    results = {}

    print("\n[1/4] Customer Churn Prediction...")
    try:
        train_churn()
        results["churn"] = "✅ Success"
    except Exception as e:
        results["churn"] = f"❌ Failed: {e}"

    print("\n[2/4] Sales Revenue Forecasting...")
    try:
        train_forecast()
        results["forecast"] = "✅ Success"
    except Exception as e:
        results["forecast"] = f"❌ Failed: {e}"

    print("\n[3/4] Delivery Delay Prediction...")
    try:
        train_delivery()
        results["delivery"] = "✅ Success"
    except Exception as e:
        results["delivery"] = f"❌ Failed: {e}"

    print("\n[4/4] Review Sentiment Classification...")
    try:
        train_sentiment()
        results["sentiment"] = "✅ Success"
    except Exception as e:
        results["sentiment"] = f"❌ Failed: {e}"

    elapsed = time.time() - t0
    print("\n" + "=" * 55)
    print("  TRAINING COMPLETE")
    print("=" * 55)
    for model, status in results.items():
        print(f"  {model:<12} {status}")
    print(f"\n  Total time : {elapsed:.1f}s")
    print(f"  Models saved in : models/")
    print(f"  MLflow UI : mlflow ui --backend-store-uri sqlite:///mlflow_tracking/mlflow.db")


if __name__ == "__main__":
    train_all()
