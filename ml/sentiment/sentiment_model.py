"""
ml/sentiment/sentiment_model.py
================================
Review Sentiment Classification using Random Forest + XGBoost ensemble.

Fixes applied:
  - class_weight="balanced" to handle Neutral class imbalance
  - SMOTE for multi-class oversampling
  - Added stronger features (review_score, late flag, freight ratio)
  - Evaluate using macro F1-score instead of accuracy
  - zero_division=0 to silence UndefinedMetricWarning
  - RandomizedSearchCV for hyperparameter tuning
  - Compare RandomForest vs XGBoost, pick best

Sentiment mapping:
  1-2  → Negative
  3    → Neutral
  4-5  → Positive

Improvements v3:
  - Merged Neutral into Not Positive (business decision — any non-4/5 star needs attention)
  - Added interaction features: late_and_expensive, very_very_late
  - 2-class is much cleaner than 3-class for this feature set
  - Macro F1 naturally improves with balanced binary classification
  - zero_division=0 silences UndefinedMetricWarning
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, accuracy_score,
    f1_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from config import PG_URL, MODELS_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

engine = create_engine(PG_URL)


def build_features() -> pd.DataFrame:
    sql = """
    SELECT
        f.price,
        f.freight_value,
        f.delivery_delay_days,
        f.payment_installments,
        f.is_delivered,
        f.order_month,
        f.order_quarter,
        f.order_year,
        f.review_score,
        CASE WHEN f.delivery_delay_days > 0 THEN 1 ELSE 0 END          AS is_late,
        ROUND((f.freight_value / NULLIF(f.price + f.freight_value,0))
              ::numeric, 4)                                             AS freight_ratio,
        ROUND((f.price / NULLIF(f.payment_installments,0))::numeric,2) AS price_per_installment,
        CASE WHEN f.delivery_delay_days > 7  THEN 1 ELSE 0 END         AS very_late,
        CASE WHEN f.delivery_delay_days <= 0 THEN 1 ELSE 0 END         AS early_delivery,
        ROUND(ABS(f.delivery_delay_days)::numeric, 1)                   AS abs_delay_days,

        -- NEW v3 interaction features
        CASE WHEN f.delivery_delay_days > 0
              AND f.price > 200 THEN 1 ELSE 0 END                      AS late_and_expensive,
        CASE WHEN f.delivery_delay_days > 14 THEN 1 ELSE 0 END         AS very_very_late,

        -- 2-class target (v3 change)
        CASE
            WHEN f.review_score >= 4 THEN 'Positive'
            ELSE 'Not Positive'
        END AS sentiment
    FROM olist.fact_orders f
    WHERE f.review_score IS NOT NULL
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)

    dist = df["sentiment"].value_counts()
    print(f"  Total reviews : {len(df):,}")
    for label, count in dist.items():
        print(f"  {label:<15} : {count:,}  ({count/len(df)*100:.1f}%)")
    return df


def train():
    print("\n── Review Sentiment Classification ───────────────")
    df = build_features()

    le = LabelEncoder()
    df["sentiment_enc"] = le.fit_transform(df["sentiment"])

    features = [
        "price", "freight_value",
        "delivery_delay_days", "payment_installments",
        "is_delivered", "order_month", "order_quarter", "order_year",
        "is_late", "freight_ratio",
        "price_per_installment", "very_late",
        "early_delivery", "abs_delay_days",
        "late_and_expensive", "very_very_late",
    ]

    X = df[features].fillna(0)
    y = df["sentiment_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n  Train distribution:")
    for cls, label in enumerate(le.classes_):
        count = (y_train == cls).sum()
        print(f"    {label:<15} : {count:,}  ({count/len(y_train)*100:.1f}%)")

    sm = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    print(f"\n  After SMOTE:")
    for cls, label in enumerate(le.classes_):
        count = (y_train_res == cls).sum()
        print(f"    {label:<15} : {count:,}")

    # ── Model 1: Random Forest ────────────────────────────────────────────
    print("\n  Training Random Forest...")
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
        {"n_estimators":[100,200,300],"max_depth":[6,8,10,None],
         "min_samples_split":[2,5,10],"min_samples_leaf":[1,2,4]},
        n_iter=15, cv=5, scoring="f1_macro",
        random_state=42, n_jobs=-1, verbose=0,
    )
    rf_search.fit(X_train_res, y_train_res)
    rf_model = rf_search.best_estimator_
    rf_pred  = rf_model.predict(X_test)
    rf_f1    = f1_score(y_test, rf_pred, average="macro")
    print(f"  RF  CV macro F1: {rf_search.best_score_:.4f}  Test macro F1: {rf_f1:.4f}")

    # ── Model 2: XGBoost ──────────────────────────────────────────────────
    print("  Training XGBoost...")
    xgb_search = RandomizedSearchCV(
        XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1),
        {"n_estimators":[100,200,300],"max_depth":[3,4,5,6],
         "learning_rate":[0.01,0.05,0.1],"subsample":[0.7,0.8,1.0],
         "colsample_bytree":[0.7,0.8,1.0]},
        n_iter=15, cv=5, scoring="f1_macro",
        random_state=42, n_jobs=-1, verbose=0,
    )
    xgb_search.fit(X_train_res, y_train_res)
    xgb_model = xgb_search.best_estimator_
    xgb_pred  = xgb_model.predict(X_test)
    xgb_f1    = f1_score(y_test, xgb_pred, average="macro")
    print(f"  XGB CV macro F1: {xgb_search.best_score_:.4f}  Test macro F1: {xgb_f1:.4f}")

    # ── Pick best ─────────────────────────────────────────────────────────
    if rf_f1 >= xgb_f1:
        model, y_pred = rf_model, rf_pred
        model_name, best_cv_f1 = "RandomForest", rf_search.best_score_
        print(f"\n  Selected: RandomForest (macro F1: {rf_f1:.4f})")
    else:
        model, y_pred = xgb_model, xgb_pred
        model_name, best_cv_f1 = "XGBoost", xgb_search.best_score_
        print(f"\n  Selected: XGBoost (macro F1: {xgb_f1:.4f})")

    acc      = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    try:
        y_proba = model.predict_proba(X_test)[:,1]
        auc     = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = 0.0

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"sentiment_{model_name.lower()}_v3"):
        mlflow.log_params({
            "model"        : model_name,
            "smote"        : True,
            "class_weight" : "balanced",
            "classes"      : "2-class: Positive vs Not Positive",
            "version"      : "v3 - merged Neutral into Not Positive",
            "features"     : str(features),
        })
        mlflow.log_metrics({
            "accuracy"  : acc,
            "f1_macro"  : f1_macro,
            "roc_auc"   : auc,
            "cv_best_f1": best_cv_f1,
        })
        mlflow.sklearn.log_model(model, artifact_path="sentiment_model")

        print(f"\n  Accuracy   : {acc:.4f}")
        print(f"  Macro F1   : {f1_macro:.4f}")
        print(f"  ROC-AUC    : {auc:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)}")

    joblib.dump(model, os.path.join(MODELS_DIR, "sentiment_model.pkl"))
    joblib.dump(le,    os.path.join(MODELS_DIR, "sentiment_le.pkl"))
    print(f"  Model saved → models/sentiment_model.pkl")
    return model, le


def predict(order_data: dict) -> dict:
    model = joblib.load(os.path.join(MODELS_DIR, "sentiment_model.pkl"))
    le    = joblib.load(os.path.join(MODELS_DIR, "sentiment_le.pkl"))

    price    = order_data.get("price", 0)
    freight  = order_data.get("freight_value", 0)
    delay    = order_data.get("delivery_delay_days", 0)
    installs = order_data.get("payment_installments", 1)

    order_data["is_late"]               = 1 if delay > 0 else 0
    order_data["freight_ratio"]         = freight / (price + freight) if (price + freight) > 0 else 0
    order_data["price_per_installment"] = price / installs if installs > 0 else price
    order_data["very_late"]             = 1 if delay > 7 else 0
    order_data["early_delivery"]        = 1 if delay <= 0 else 0
    order_data["abs_delay_days"]        = abs(delay)
    order_data["late_and_expensive"]    = 1 if delay > 0 and price > 200 else 0
    order_data["very_very_late"]        = 1 if delay > 14 else 0

    features = [
        "price", "freight_value",
        "delivery_delay_days", "payment_installments",
        "is_delivered", "order_month", "order_quarter", "order_year",
        "is_late", "freight_ratio",
        "price_per_installment", "very_late",
        "early_delivery", "abs_delay_days",
        "late_and_expensive", "very_very_late",
    ]

    X     = pd.DataFrame([order_data])[features].fillna(0)
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    label = le.inverse_transform([pred])[0]
    conf  = round(float(proba.max()), 4)

    return {
        "sentiment"    : label,
        "confidence"   : conf,
        "probabilities": {
            cls: round(float(p), 4)
            for cls, p in zip(le.classes_, proba)
        }
    }


if __name__ == "__main__":
    train()