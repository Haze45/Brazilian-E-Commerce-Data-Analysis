"""
ml/sentiment/sentiment_model.py
================================
Review Sentiment Classification using Random Forest.

Maps review_score to sentiment:
  1-2  → Negative
  3    → Neutral
  4-5  → Positive
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from config import PG_URL, MODELS_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

engine = create_engine(PG_URL)


def build_features() -> pd.DataFrame:
    sql = """
    SELECT
        f.review_score,
        f.price,
        f.freight_value,
        f.delivery_delay_days,
        f.payment_installments,
        f.is_delivered,
        f.order_month,
        f.order_quarter,
        CASE
            WHEN f.review_score <= 2 THEN 'Negative'
            WHEN f.review_score = 3  THEN 'Neutral'
            ELSE 'Positive'
        END AS sentiment
    FROM olist.fact_orders f
    WHERE f.review_score IS NOT NULL
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)

    dist = df["sentiment"].value_counts()
    print(f"  Total reviews : {len(df):,}")
    for label, count in dist.items():
        print(f"  {label:<10} : {count:,}  ({count/len(df)*100:.1f}%)")

    return df


def train():
    print("\n── Review Sentiment Classification ───────────────")
    df = build_features()

    le = LabelEncoder()
    df["sentiment_enc"] = le.fit_transform(df["sentiment"])

    features = [
        "price", "freight_value",
        "delivery_delay_days", "payment_installments",
        "is_delivered", "order_month", "order_quarter"
    ]

    X = df[features].fillna(0)
    y = df["sentiment_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="sentiment_rf"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)

        mlflow.log_params({"model": "RandomForest", "n_estimators": 100})
        mlflow.log_metrics({"accuracy": acc})
        mlflow.sklearn.log_model(model, "sentiment_model")

        print(f"\n  Accuracy : {acc:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=le.classes_)}")

    joblib.dump(model, os.path.join(MODELS_DIR, "sentiment_model.pkl"))
    joblib.dump(le,    os.path.join(MODELS_DIR, "sentiment_le.pkl"))
    print(f"  Model saved → models/sentiment_model.pkl")

    return model, le


def predict(order_data: dict) -> dict:
    model = joblib.load(os.path.join(MODELS_DIR, "sentiment_model.pkl"))
    le    = joblib.load(os.path.join(MODELS_DIR, "sentiment_le.pkl"))

    features = [
        "price", "freight_value",
        "delivery_delay_days", "payment_installments",
        "is_delivered", "order_month", "order_quarter"
    ]

    X       = pd.DataFrame([order_data])[features].fillna(0)
    pred    = model.predict(X)[0]
    proba   = model.predict_proba(X)[0]
    label   = le.inverse_transform([pred])[0]
    conf    = round(float(proba.max()), 4)

    return {
        "sentiment"   : label,
        "confidence"  : conf,
        "probabilities": {
            cls: round(float(p), 4)
            for cls, p in zip(le.classes_, proba)
        }
    }


if __name__ == "__main__":
    train()
