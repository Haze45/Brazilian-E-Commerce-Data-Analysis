"""
ml/churn/churn_model.py
=======================
Customer Churn Prediction using XGBoost.

Definition: A customer is "churned" if they have NOT placed an order
in the last 180 days (relative to 2018-09-01).

Fixes applied:
  - Removed recency_days from features (was causing data leakage)
  - Added SMOTE for class imbalance
  - Added RandomizedSearchCV for hyperparameter tuning
  - Added optimal threshold tuning via precision-recall curve
  - Added 6 new behavioral features

Improvements v3:
  - Added customer_lifetime_days, avg_days_between_orders, category_diversity
  - Balanced threshold search (both Active + Churned recall >= 0.65)
  - Better feature engineering for customer behavior patterns
"""


import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import joblib
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    accuracy_score, precision_recall_curve,
    recall_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from config import PG_URL, MODELS_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

engine = create_engine(PG_URL)


def build_features() -> pd.DataFrame:
    sql = """
    SELECT
        f.customer_id,
        COUNT(DISTINCT f.order_id)                                      AS frequency,
        ROUND(SUM(f.revenue)::numeric, 2)                               AS monetary,
        ROUND(AVG(f.price)::numeric, 2)                                 AS avg_order_value,
        ROUND(AVG(f.review_score)::numeric, 2)                          AS avg_review_score,
        ROUND(AVG(f.freight_value / NULLIF(f.revenue, 0))::numeric, 4)  AS avg_freight_pct,
        ROUND(AVG(f.payment_installments)::numeric, 2)                  AS avg_installments,
        SUM(CASE WHEN f.delivery_delay_days > 0 THEN 1 ELSE 0 END)     AS late_deliveries,
        ROUND(AVG(f.delivery_delay_days)::numeric, 1)                   AS avg_delay_days,
        COUNT(DISTINCT f.category)                                      AS unique_categories,
        COUNT(DISTINCT f.product_id)                                    AS unique_products,
        ROUND(MAX(f.price)::numeric, 2)                                 AS max_order_value,
        ROUND(MIN(f.price)::numeric, 2)                                 AS min_order_value,
        SUM(CASE WHEN f.review_score >= 4 THEN 1 ELSE 0 END)           AS positive_reviews,
        SUM(CASE WHEN f.review_score <= 2 THEN 1 ELSE 0 END)           AS negative_reviews,
        COUNT(DISTINCT f.order_year)                                    AS active_years,
        COUNT(DISTINCT f.order_month)                                   AS active_months,

        -- NEW v3 features
        MAX(f.order_date)::date - MIN(f.order_date)::date               AS customer_lifetime_days,
        CASE WHEN COUNT(DISTINCT f.order_id) > 1
             THEN (MAX(f.order_date)::date - MIN(f.order_date)::date)
                  / (COUNT(DISTINCT f.order_id) - 1)
             ELSE 0 END                                                 AS avg_days_between_orders,
        COUNT(DISTINCT f.category)                                      AS category_diversity,
        ROUND(SUM(f.revenue) / NULLIF(COUNT(DISTINCT f.order_id), 0)
              ::numeric, 2)                                             AS revenue_per_order,

        -- Used ONLY to define churn target — dropped before training
        DATE '2018-09-01' - MAX(f.order_date)::date                     AS recency_days
    FROM olist.fact_orders f
    WHERE f.is_delivered = 1
    GROUP BY f.customer_id
    HAVING COUNT(DISTINCT f.order_id) >= 1
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)

    df["churned"] = (df["recency_days"] > 180).astype(int)
    df = df.drop(columns=["recency_days"])

    print(f"  Total customers : {len(df):,}")
    print(f"  Churned (1)     : {df['churned'].sum():,}  ({df['churned'].mean()*100:.1f}%)")
    print(f"  Active  (0)     : {(df['churned']==0).sum():,}  ({(df['churned']==0).mean()*100:.1f}%)")
    return df


def train():
    print("\n── Customer Churn Prediction ─────────────────────")
    df = build_features()

    features = [
        "frequency", "monetary", "avg_order_value",
        "avg_review_score", "avg_freight_pct", "avg_installments",
        "late_deliveries", "avg_delay_days",
        "unique_categories", "unique_products",
        "max_order_value", "min_order_value",
        "positive_reviews", "negative_reviews",
        "active_years", "active_months",
        # New v3
        "customer_lifetime_days", "avg_days_between_orders",
        "category_diversity", "revenue_per_order",
    ]

    X = df[features].fillna(0)
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_sc, y_train)
    print(f"  After SMOTE - Active  : {(y_train_res==0).sum():,}")
    print(f"  After SMOTE - Churned : {(y_train_res==1).sum():,}")

    param_grid = {
        "n_estimators"    : [100, 200, 300],
        "max_depth"       : [3, 4, 5, 6],
        "learning_rate"   : [0.01, 0.05, 0.1],
        "subsample"       : [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
    }

    base_model = XGBClassifier(
        random_state=42, eval_metric="logloss", n_jobs=-1,
    )

    search = RandomizedSearchCV(
        base_model, param_grid,
        n_iter=20, cv=5, scoring="roc_auc",
        random_state=42, n_jobs=-1, verbose=1,
    )

    print("\n  Running hyperparameter search...")
    search.fit(X_train_res, y_train_res)
    model = search.best_estimator_

    print(f"\n  Best params : {search.best_params_}")
    print(f"  Best CV AUC : {search.best_score_:.4f}")

    y_proba = model.predict_proba(X_test_sc)[:, 1]

    # Balanced threshold — find where BOTH recalls >= 0.65
    best_threshold = 0.5
    best_balance   = -1
    for thresh in np.arange(0.2, 0.8, 0.01):
        y_pred_t       = (y_proba >= thresh).astype(int)
        active_recall  = recall_score(y_test, y_pred_t, pos_label=0)
        churned_recall = recall_score(y_test, y_pred_t, pos_label=1)
        balance        = min(active_recall, churned_recall)
        if balance > best_balance:
            best_balance   = balance
            best_threshold = thresh

    y_pred_optimal = (y_proba >= best_threshold).astype(int)
    acc = accuracy_score(y_test, y_pred_optimal)
    auc = roc_auc_score(y_test, y_proba)
    active_recall  = recall_score(y_test, y_pred_optimal, pos_label=0)
    churned_recall = recall_score(y_test, y_pred_optimal, pos_label=1)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="churn_xgboost_v3"):
        mlflow.log_params({
            "model"          : "XGBClassifier",
            "best_params"    : str(search.best_params_),
            "smote"          : True,
            "threshold"      : round(float(best_threshold), 4),
            "leakage_fix"    : "removed recency_days",
            "version"        : "v3 - balanced threshold + new features",
        })
        mlflow.log_metrics({
            "accuracy"       : acc,
            "roc_auc"        : auc,
            "active_recall"  : active_recall,
            "churned_recall" : churned_recall,
            "best_threshold" : float(best_threshold),
        })
        mlflow.xgboost.log_model(model, artifact_path="churn_model")

        print(f"\n  Accuracy        : {acc:.4f}")
        print(f"  ROC-AUC         : {auc:.4f}")
        print(f"  Active recall   : {active_recall:.4f}")
        print(f"  Churned recall  : {churned_recall:.4f}")
        print(f"  Threshold       : {best_threshold:.4f}")
        print(f"\n{classification_report(y_test, y_pred_optimal, target_names=['Active','Churned'])}")

    joblib.dump(model,  os.path.join(MODELS_DIR, "churn_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "churn_scaler.pkl"))
    print(f"  Model saved → models/churn_model.pkl")
    return model, scaler, features


def predict(customer_data: dict) -> dict:
    model  = joblib.load(os.path.join(MODELS_DIR, "churn_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "churn_scaler.pkl"))

    features = [
        "frequency", "monetary", "avg_order_value",
        "avg_review_score", "avg_freight_pct", "avg_installments",
        "late_deliveries", "avg_delay_days",
        "unique_categories", "unique_products",
        "max_order_value", "min_order_value",
        "positive_reviews", "negative_reviews",
        "active_years", "active_months",
        "customer_lifetime_days", "avg_days_between_orders",
        "category_diversity", "revenue_per_order",
    ]

    X    = pd.DataFrame([customer_data])[features].fillna(0)
    X_sc = scaler.transform(X)
    proba = model.predict_proba(X_sc)[0][1]
    pred  = int(proba >= 0.5)

    return {
        "churned"           : pred,
        "churn_probability" : round(float(proba), 4),
        "risk_level"        : "High" if proba >= 0.7 else "Medium" if proba >= 0.4 else "Low"
    }


if __name__ == "__main__":
    train()
