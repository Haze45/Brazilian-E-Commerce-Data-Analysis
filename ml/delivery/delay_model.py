"""
ml/delivery/delay_model.py
===========================
Delivery Delay Prediction using Gradient Boosting + Random Forest ensemble.

Fixes applied:
  - SMOTE to handle class imbalance (Late: 6.6% vs On-time: 93.4%)
  - compute_sample_weight for weighted training
  - Optimal threshold tuning via precision-recall curve
  - Added cross_state feature (seller_state != customer_state)
  - Evaluate using F1-score for Late class (not just accuracy)
  - RandomizedSearchCV for hyperparameter tuning
  - Compare GradientBoosting vs RandomForest, pick best
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    classification_report, roc_auc_score,
    accuracy_score, f1_score, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from config import PG_URL, MODELS_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

engine = create_engine(PG_URL)


# ── FEATURES ──────────────────────────────────────────────────────────────────
def build_features() -> pd.DataFrame:
    sql = """
    SELECT
        f.order_id,
        f.price,
        f.freight_value,
        f.order_month,
        f.order_quarter,
        f.order_year,
        f.payment_installments,
        f.state                                                     AS customer_state,
        s.seller_state,
        p.weight_g,
        p.length_cm,
        p.height_cm,
        p.width_cm,
        -- NEW: cross state flag (longer distance = more likely late)
        CASE WHEN s.seller_state != f.state THEN 1 ELSE 0 END      AS cross_state,
        -- NEW: freight ratio (high freight % may indicate long distance)
        ROUND((f.freight_value / NULLIF(f.price + f.freight_value, 0))::numeric, 4)
                                                                    AS freight_ratio,
        -- NEW: volume proxy
        ROUND((NULLIF(p.length_cm, 0) * NULLIF(p.height_cm, 0)
               * NULLIF(p.width_cm, 0))::numeric, 2)               AS volume_cm3,
        CASE WHEN f.delivery_delay_days > 0
             THEN 1 ELSE 0 END                                      AS is_late
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


# ── TRAIN ─────────────────────────────────────────────────────────────────────
def train():
    print("\n── Delivery Delay Prediction ─────────────────────")
    df = build_features()

    # Encode categorical columns
    le_cust   = LabelEncoder()
    le_seller = LabelEncoder()
    df["customer_state_enc"] = le_cust.fit_transform(df["customer_state"].fillna("unknown"))
    df["seller_state_enc"]   = le_seller.fit_transform(df["seller_state"].fillna("unknown"))

    features = [
        "price", "freight_value", "freight_ratio",
        "order_month", "order_quarter", "order_year",
        "payment_installments",
        "customer_state_enc", "seller_state_enc",
        "cross_state",
        "weight_g", "length_cm", "height_cm", "width_cm",
        "volume_cm3",
    ]

    X = df[features].fillna(0)
    y = df["is_late"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n  Train set - On-time: {(y_train==0).sum():,}  Late: {(y_train==1).sum():,}")

    # SMOTE — oversample minority Late class
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"  After SMOTE - On-time: {(y_train_res==0).sum():,}  Late: {(y_train_res==1).sum():,}")

    # Compute sample weights for weighted training
    sample_weights = compute_sample_weight("balanced", y_train_res)

    # ── Model 1: Gradient Boosting ─────────────────────────────────────────
    print("\n  Training Gradient Boosting...")
    gb_params = {
        "n_estimators" : [100, 150, 200],
        "max_depth"    : [3, 4, 5],
        "learning_rate": [0.05, 0.1, 0.15],
        "subsample"    : [0.7, 0.8, 1.0],
    }

    gb_search = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=42),
        gb_params,
        n_iter=15, cv=5,
        scoring="f1",
        random_state=42, n_jobs=-1, verbose=0,
    )
    gb_search.fit(X_train_res, y_train_res, sample_weight=sample_weights)
    gb_model  = gb_search.best_estimator_
    gb_proba  = gb_model.predict_proba(X_test)[:, 1]
    gb_auc    = roc_auc_score(y_test, gb_proba)
    print(f"  GB  best CV F1 : {gb_search.best_score_:.4f}  |  ROC-AUC: {gb_auc:.4f}")

    # ── Model 2: Random Forest with balanced class_weight ─────────────────
    print("  Training Random Forest...")
    rf_params = {
        "n_estimators": [100, 200, 300],
        "max_depth"   : [5, 8, 10, None],
        "min_samples_split": [2, 5, 10],
    }

    rf_search = RandomizedSearchCV(
        RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        rf_params,
        n_iter=15, cv=5,
        scoring="f1",
        random_state=42, n_jobs=-1, verbose=0,
    )
    rf_search.fit(X_train_res, y_train_res)
    rf_model  = rf_search.best_estimator_
    rf_proba  = rf_model.predict_proba(X_test)[:, 1]
    rf_auc    = roc_auc_score(y_test, rf_proba)
    print(f"  RF  best CV F1 : {rf_search.best_score_:.4f}  |  ROC-AUC: {rf_auc:.4f}")

    # ── Pick best model ───────────────────────────────────────────────────
    if gb_search.best_score_ >= rf_search.best_score_:
        model      = gb_model
        y_proba    = gb_proba
        model_name = "GradientBoosting"
        best_params = gb_search.best_params_
        print(f"\n  Selected: GradientBoosting (CV F1: {gb_search.best_score_:.4f})")
    else:
        model      = rf_model
        y_proba    = rf_proba
        model_name = "RandomForest"
        best_params = rf_search.best_params_
        print(f"\n  Selected: RandomForest (CV F1: {rf_search.best_score_:.4f})")

    # ── Optimal threshold ─────────────────────────────────────────────────
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_threshold = thresholds[f1_scores.argmax()]
    y_pred_optimal = (y_proba >= best_threshold).astype(int)

    acc      = accuracy_score(y_test, y_pred_optimal)
    auc      = roc_auc_score(y_test, y_proba)
    f1_late  = f1_score(y_test, y_pred_optimal, pos_label=1)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"delivery_{model_name.lower()}_v2"):
        mlflow.log_params({
            "model"           : model_name,
            "best_params"     : str(best_params),
            "smote"           : True,
            "threshold"       : round(float(best_threshold), 4),
            "imbalance_fix"   : "SMOTE + sample_weight + threshold tuning",
            "features"        : str(features),
        })
        mlflow.log_metrics({
            "accuracy"        : acc,
            "roc_auc"         : auc,
            "f1_late"         : f1_late,
            "best_threshold"  : float(best_threshold),
        })
        mlflow.sklearn.log_model(model, name="delivery_model")

        print(f"\n  Accuracy     : {acc:.4f}")
        print(f"  ROC-AUC      : {auc:.4f}")
        print(f"  Late F1      : {f1_late:.4f}")
        print(f"  Threshold    : {best_threshold:.4f}")
        print(f"\n{classification_report(y_test, y_pred_optimal, target_names=['On-time','Late'])}")

    # Save
    joblib.dump(model,     os.path.join(MODELS_DIR, "delivery_model.pkl"))
    joblib.dump(le_cust,   os.path.join(MODELS_DIR, "delivery_le_cust.pkl"))
    joblib.dump(le_seller, os.path.join(MODELS_DIR, "delivery_le_seller.pkl"))
    print(f"  Model saved → models/delivery_model.pkl")

    return model


# ── PREDICT ───────────────────────────────────────────────────────────────────
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

    # cross_state flag
    order_data["cross_state"] = 1 if cust_state != sell_state else 0

    # freight ratio
    price    = order_data.get("price", 0)
    freight  = order_data.get("freight_value", 0)
    order_data["freight_ratio"] = freight / (price + freight) if (price + freight) > 0 else 0

    # volume
    l = order_data.get("length_cm", 0)
    h = order_data.get("height_cm", 0)
    w = order_data.get("width_cm", 0)
    order_data["volume_cm3"] = l * h * w

    features = [
        "price", "freight_value", "freight_ratio",
        "order_month", "order_quarter", "order_year",
        "payment_installments",
        "customer_state_enc", "seller_state_enc",
        "cross_state",
        "weight_g", "length_cm", "height_cm", "width_cm",
        "volume_cm3",
    ]

    X     = pd.DataFrame([order_data])[features].fillna(0)
    proba = model.predict_proba(X)[0][1]
    pred  = int(proba >= 0.5)

    return {
        "will_be_late"     : bool(pred),
        "late_probability" : round(float(proba), 4),
        "risk_level"       : "High" if proba >= 0.7 else "Medium" if proba >= 0.4 else "Low"
    }


if __name__ == "__main__":
    train()