"""
ml/churn/churn_model.py
=======================
Customer Churn Prediction using XGBoost.

Definition: A customer is "churned" if they have NOT placed an order
in the last 180 days (relative to the last order date in the dataset).
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, accuracy_score
)
from xgboost import XGBClassifier
from config import PG_URL, MODELS_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

engine = create_engine(PG_URL)


def build_features() -> pd.DataFrame:
    sql = """
    SELECT
        f.customer_id,
        MAX(f.order_date)::date                                         AS last_order_date,
        DATE '2018-09-01' - MAX(f.order_date)::date                     AS recency_days,
        COUNT(DISTINCT f.order_id)                                      AS frequency,
        ROUND(SUM(f.revenue)::numeric, 2)                               AS monetary,
        ROUND(AVG(f.price)::numeric, 2)                                 AS avg_order_value,
        ROUND(AVG(f.review_score)::numeric, 2)                          AS avg_review_score,
        ROUND(AVG(f.freight_value / NULLIF(f.revenue, 0))::numeric, 4)  AS avg_freight_pct,
        ROUND(AVG(f.payment_installments)::numeric, 2)                  AS avg_installments,
        SUM(CASE WHEN f.delivery_delay_days > 0 THEN 1 ELSE 0 END)     AS late_deliveries,
        ROUND(AVG(f.delivery_delay_days)::numeric, 1)                   AS avg_delay_days,
        COUNT(DISTINCT f.category)                                      AS unique_categories,
        COUNT(DISTINCT f.product_id)                                    AS unique_products
    FROM olist.fact_orders f
    WHERE f.is_delivered = 1
    GROUP BY f.customer_id
    HAVING COUNT(DISTINCT f.order_id) >= 1
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)

    df["churned"] = (df["recency_days"] > 180).astype(int)

    print(f"  Total customers : {len(df):,}")
    print(f"  Churned (1)     : {df['churned'].sum():,}  ({df['churned'].mean()*100:.1f}%)")
    print(f"  Active  (0)     : {(df['churned']==0).sum():,}  ({(df['churned']==0).mean()*100:.1f}%)")

    return df


def train():
    print("\n── Customer Churn Prediction ─────────────────────")
    df = build_features()

    features = [
        "recency_days", "frequency", "monetary",
        "avg_review_score", "avg_freight_pct", "avg_installments",
        "late_deliveries", "avg_delay_days",
        "unique_categories", "unique_products"
    ]

    X = df[features].fillna(0)
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,
        random_state=42,
        eval_metric="logloss",
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="churn_xgboost"):
        model.fit(X_train_sc, y_train)
        y_pred  = model.predict(X_test_sc)
        y_proba = model.predict_proba(X_test_sc)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        mlflow.log_params({
            "model"          : "XGBClassifier",
            "n_estimators"   : 200,
            "max_depth"      : 6,
            "learning_rate"  : 0.05,
            "features"       : str(features),
        })
        mlflow.log_metrics({"accuracy": acc, "roc_auc": auc})
        mlflow.sklearn.log_model(model, "churn_model")

        print(f"\n  Accuracy : {acc:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Active','Churned'])}")

    joblib.dump(model,  os.path.join(MODELS_DIR, "churn_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "churn_scaler.pkl"))
    print(f"  Model saved → models/churn_model.pkl")

    return model, scaler, features


def predict(customer_data: dict) -> dict:
    model  = joblib.load(os.path.join(MODELS_DIR, "churn_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "churn_scaler.pkl"))

    features = [
        "recency_days", "frequency", "monetary",
        "avg_review_score", "avg_freight_pct", "avg_installments",
        "late_deliveries", "avg_delay_days",
        "unique_categories", "unique_products"
    ]

    X = pd.DataFrame([customer_data])[features].fillna(0)
    X_sc = scaler.transform(X)

    proba = model.predict_proba(X_sc)[0][1]
    pred  = int(proba >= 0.5)

    return {
        "churned"       : pred,
        "churn_probability" : round(float(proba), 4),
        "risk_level"    : "High" if proba >= 0.7 else "Medium" if proba >= 0.4 else "Low"
    }


if __name__ == "__main__":
    train()
