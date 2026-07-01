"""
ml/delivery/delay_model.py
===========================
Delivery Delay Prediction using Gradient Boosting.
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from config import PG_URL, MODELS_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

engine = create_engine(PG_URL)


def build_features() -> pd.DataFrame:
    sql = """
    SELECT
        f.order_id,
        f.price,
        f.freight_value,
        f.order_month,
        f.order_quarter,
        f.payment_installments,
        f.state                             AS customer_state,
        s.seller_state,
        p.weight_g,
        p.length_cm,
        p.height_cm,
        p.width_cm,
        CASE WHEN f.delivery_delay_days > 0
             THEN 1 ELSE 0 END              AS is_late
    FROM olist.fact_orders f
    JOIN olist.dim_products p ON f.product_id = p.product_id
    JOIN olist.dim_sellers  s ON f.seller_id  = s.seller_id
    WHERE f.is_delivered = 1
      AND f.delivery_delay_days IS NOT NULL
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)

    print(f"  Total orders : {len(df):,}")
    print(f"  Late (1)     : {df['is_late'].sum():,}  ({df['is_late'].mean()*100:.1f}%)")
    print(f"  On-time (0)  : {(df['is_late']==0).sum():,}  ({(df['is_late']==0).mean()*100:.1f}%)")

    return df


def train():
    print("\n── Delivery Delay Prediction ─────────────────────")
    df = build_features()

    le_cust   = LabelEncoder()
    le_seller = LabelEncoder()
    df["customer_state_enc"] = le_cust.fit_transform(df["customer_state"].fillna("unknown"))
    df["seller_state_enc"]   = le_seller.fit_transform(df["seller_state"].fillna("unknown"))

    features = [
        "price", "freight_value",
        "order_month", "order_quarter",
        "payment_installments",
        "customer_state_enc", "seller_state_enc",
        "weight_g", "length_cm", "height_cm", "width_cm"
    ]

    X = df[features].fillna(0)
    y = df["is_late"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="delivery_delay_gb"):
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        mlflow.log_params({
            "model"         : "GradientBoostingClassifier",
            "n_estimators"  : 150,
            "max_depth"     : 5,
            "learning_rate" : 0.1,
        })
        mlflow.log_metrics({"accuracy": acc, "roc_auc": auc})
        mlflow.sklearn.log_model(model, "delivery_model")

        print(f"\n  Accuracy : {acc:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['On-time','Late'])}")

    joblib.dump(model,     os.path.join(MODELS_DIR, "delivery_model.pkl"))
    joblib.dump(le_cust,   os.path.join(MODELS_DIR, "delivery_le_cust.pkl"))
    joblib.dump(le_seller, os.path.join(MODELS_DIR, "delivery_le_seller.pkl"))
    print(f"  Model saved → models/delivery_model.pkl")

    return model


def predict(order_data: dict) -> dict:
    model     = joblib.load(os.path.join(MODELS_DIR, "delivery_model.pkl"))
    le_cust   = joblib.load(os.path.join(MODELS_DIR, "delivery_le_cust.pkl"))
    le_seller = joblib.load(os.path.join(MODELS_DIR, "delivery_le_seller.pkl"))

    cust_state = order_data.get("customer_state", "SP")
    sell_state = order_data.get("seller_state", "SP")
    order_data["customer_state_enc"] = le_cust.transform(
        [cust_state if cust_state in le_cust.classes_ else le_cust.classes_[0]])[0]
    order_data["seller_state_enc"] = le_seller.transform(
        [sell_state if sell_state in le_seller.classes_ else le_seller.classes_[0]])[0]

    features = [
        "price", "freight_value",
        "order_month", "order_quarter",
        "payment_installments",
        "customer_state_enc", "seller_state_enc",
        "weight_g", "length_cm", "height_cm", "width_cm"
    ]

    X     = pd.DataFrame([order_data])[features].fillna(0)
    proba = model.predict_proba(X)[0][1]
    pred  = int(proba >= 0.5)

    return {
        "will_be_late"      : bool(pred),
        "late_probability"  : round(float(proba), 4),
        "risk_level"        : "High" if proba >= 0.7 else "Medium" if proba >= 0.4 else "Low"
    }


if __name__ == "__main__":
    train()
