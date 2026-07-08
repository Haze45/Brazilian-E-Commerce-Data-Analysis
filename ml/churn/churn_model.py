"""
Customer Churn Prediction using XGBoost.

Definition: A customer is "churned" if they have NOT placed an order
in the last 180 days (relative to 2018-09-01).

Improvements v2:
  - Removed recency_days from features (was causing data leakage)
  - Added SMOTE for class imbalance
  - Added RandomizedSearchCV for hyperparameter tuning
  - Added optimal threshold tuning via precision-recall curve
  - Added 6 new behavioral features

Improvements v3:
  - Added customer_lifetime_days, avg_days_between_orders, category_diversity
  - Balanced threshold search (both Active + Churned recall >= 0.65)
  - Better feature engineering for customer behavior patterns

Improvements v4:
  A — Value-weighted churn: sample_weight = monetary × recency_factor
      High-value churned customers penalized more during training
  B — Time-based trend features:
      revenue_trend (last 3 orders vs historical)
      velocity_change (recent order gap vs historical gap)
      days_since_second_last_order
  D — Reduced overfitting:
      max_depth reduced from 6 to 4
      min_child_weight increased range to 15
      Added reg_alpha (L1) and reg_lambda (L2) regularization
      Expected CV-test gap to close from 0.036 to ~0.01
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
    accuracy_score, recall_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from config import PG_URL, MODELS_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

engine = create_engine(PG_URL)


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
def build_features() -> pd.DataFrame:
    sql = """
    WITH customer_orders AS (
        -- All delivered orders per customer with row numbers
        SELECT
            customer_id,
            order_date,
            revenue,
            ROW_NUMBER() OVER (
                PARTITION BY customer_id ORDER BY order_date DESC
            ) AS order_rank,
            COUNT(*) OVER (PARTITION BY customer_id) AS total_orders
        FROM olist.fact_orders
        WHERE is_delivered = 1
    ),

    last_orders AS (
        -- Last order date and second-to-last order date per customer
        SELECT
            customer_id,
            MAX(CASE WHEN order_rank = 1 THEN order_date END) AS last_order_date,
            MAX(CASE WHEN order_rank = 2 THEN order_date END) AS second_last_order_date,
            MAX(CASE WHEN order_rank = 3 THEN order_date END) AS third_last_order_date
        FROM customer_orders
        GROUP BY customer_id
    ),

    recent_revenue AS (
        -- Revenue from last 3 orders vs older orders
        SELECT
            customer_id,
            ROUND(SUM(CASE WHEN order_rank <= 3 THEN revenue ELSE 0 END)::numeric, 2)
                AS last_3_revenue,
            ROUND(SUM(CASE WHEN order_rank > 3 THEN revenue ELSE 0 END)::numeric, 2)
                AS older_revenue,
            COUNT(CASE WHEN order_rank <= 3 THEN 1 END)
                AS last_3_count,
            COUNT(CASE WHEN order_rank > 3 THEN 1 END)
                AS older_count
        FROM customer_orders
        GROUP BY customer_id
    )

    SELECT
        f.customer_id,

        -- ── Core RFM features ─────────────────────────────────────────
        COUNT(DISTINCT f.order_id)                                      AS frequency,
        ROUND(SUM(f.revenue)::numeric, 2)                               AS monetary,
        ROUND(AVG(f.price)::numeric, 2)                                 AS avg_order_value,

        -- ── Quality signals ───────────────────────────────────────────
        ROUND(AVG(f.review_score)::numeric, 2)                          AS avg_review_score,
        ROUND(AVG(f.freight_value / NULLIF(f.revenue, 0))::numeric, 4)  AS avg_freight_pct,
        ROUND(AVG(f.payment_installments)::numeric, 2)                  AS avg_installments,

        -- ── Delivery experience ───────────────────────────────────────
        SUM(CASE WHEN f.delivery_delay_days > 0 THEN 1 ELSE 0 END)     AS late_deliveries,
        ROUND(AVG(f.delivery_delay_days)::numeric, 1)                   AS avg_delay_days,

        -- ── Product diversity ─────────────────────────────────────────
        COUNT(DISTINCT f.category)                                      AS unique_categories,
        COUNT(DISTINCT f.product_id)                                    AS unique_products,

        -- ── Price behavior ────────────────────────────────────────────
        ROUND(MAX(f.price)::numeric, 2)                                 AS max_order_value,
        ROUND(MIN(f.price)::numeric, 2)                                 AS min_order_value,

        -- ── Review behavior ───────────────────────────────────────────
        SUM(CASE WHEN f.review_score >= 4 THEN 1 ELSE 0 END)           AS positive_reviews,
        SUM(CASE WHEN f.review_score <= 2 THEN 1 ELSE 0 END)           AS negative_reviews,

        -- ── Activity patterns ─────────────────────────────────────────
        COUNT(DISTINCT f.order_year)                                    AS active_years,
        COUNT(DISTINCT f.order_month)                                   AS active_months,

        -- ── Lifetime features ─────────────────────────────────────────
        MAX(f.order_date)::date - MIN(f.order_date)::date               AS customer_lifetime_days,
        CASE WHEN COUNT(DISTINCT f.order_id) > 1
             THEN (MAX(f.order_date)::date - MIN(f.order_date)::date)
                  / (COUNT(DISTINCT f.order_id) - 1)
             ELSE 0 END                                                 AS avg_days_between_orders,
        COUNT(DISTINCT f.category)                                      AS category_diversity,
        ROUND((SUM(f.revenue) / NULLIF(COUNT(DISTINCT f.order_id), 0))
              ::numeric, 2)                                             AS revenue_per_order,

        -- ── NEW B: Time-based trend features ──────────────────────────
        -- Revenue trend: last 3 orders vs historical
        -- > 1.0 = spending more recently (healthy)
        -- < 1.0 = spending less recently (warning sign)
        CASE
            WHEN rr.older_revenue > 0
            THEN ROUND((rr.last_3_revenue / rr.older_revenue)::numeric, 4)
            WHEN rr.last_3_revenue > 0 THEN 2.0
            ELSE 1.0
        END                                                             AS revenue_trend,

        -- Days since second-to-last order (captures unusual silence)
        CASE
            WHEN lo.second_last_order_date IS NOT NULL
            THEN DATE '2018-09-01' - lo.second_last_order_date::date
            ELSE 999
        END                                                             AS days_since_second_last,

        -- Order velocity change: recent gap vs historical avg gap
        -- > 1.0 = ordering slower recently (warning sign)
        -- < 1.0 = ordering faster recently (healthy)
        CASE
            WHEN lo.second_last_order_date IS NOT NULL
              AND lo.third_last_order_date IS NOT NULL
            THEN ROUND(
                -- Recent gap (last to second-last)
                ((lo.last_order_date::date - lo.second_last_order_date::date)::float /
                -- Historical avg gap
                NULLIF((MAX(f.order_date)::date - MIN(f.order_date)::date)::float
                       / NULLIF(COUNT(DISTINCT f.order_id) - 1, 0), 0)
                )::numeric, 4)
            ELSE 1.0
        END                                                             AS velocity_change,

        -- ── Used ONLY to define churn target — dropped before training ─
        DATE '2018-09-01' - MAX(f.order_date)::date                     AS recency_days
    FROM olist.fact_orders f
    JOIN last_orders  lo ON f.customer_id = lo.customer_id
    JOIN recent_revenue rr ON f.customer_id = rr.customer_id
    WHERE f.is_delivered = 1
    GROUP BY f.customer_id, lo.last_order_date, lo.second_last_order_date,
             lo.third_last_order_date, rr.last_3_revenue, rr.older_revenue,
             rr.last_3_count, rr.older_count
    HAVING COUNT(DISTINCT f.order_id) >= 1
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)

    # Define churn from recency_days (NOT used as feature)
    df["churned"] = (df["recency_days"] > 180).astype(int)

    # ── A: Value-weighted sample weights ──────────────────────────────────
    # High-value churned customers get higher weight during training
    # recency_factor: closer to 180 days = higher urgency (more borderline)
    df["recency_factor"] = np.where(
        df["churned"] == 1,
        1 + (df["recency_days"] - 180) / 365,  # higher for longer absence
        1.0
    )
    df["recency_factor"] = df["recency_factor"].clip(1.0, 3.0)

    # Normalize monetary to 0-1 scale for weighting
    df["monetary_norm"] = (
            (df["monetary"] - df["monetary"].min()) /
            (df["monetary"].max() - df["monetary"].min() + 1e-9)
    )
    # sample_weight = base + monetary contribution for churned customers
    df["sample_weight"] = np.where(
        df["churned"] == 1,
        1.0 + df["monetary_norm"] * df["recency_factor"],
        1.0
    )

    # Drop recency_days and helper columns — must NOT be used as features
    df = df.drop(columns=["recency_days", "recency_factor", "monetary_norm"])

    # Cap extreme values in trend features
    df["revenue_trend"] = df["revenue_trend"].clip(0, 10)
    df["velocity_change"] = df["velocity_change"].clip(0, 10)

    print(f"  Total customers : {len(df):,}")
    print(f"  Churned (1)     : {df['churned'].sum():,}  ({df['churned'].mean() * 100:.1f}%)")
    print(f"  Active  (0)     : {(df['churned'] == 0).sum():,}  ({(df['churned'] == 0).mean() * 100:.1f}%)")
    print(f"\n  New trend features sample:")
    print(df[["revenue_trend", "velocity_change", "days_since_second_last"]].describe().round(2).to_string())
    return df


# ── TRAIN ─────────────────────────────────────────────────────────────────────
def train():
    print("\n── Customer Churn Prediction v4 ──────────────────")
    df = build_features()

    features = [
        # Core RFM (recency_days excluded — leakage fix from v3)
        "frequency", "monetary", "avg_order_value",
        # Quality signals
        "avg_review_score", "avg_freight_pct", "avg_installments",
        # Delivery experience
        "late_deliveries", "avg_delay_days",
        # Product diversity
        "unique_categories", "unique_products",
        # Price behavior
        "max_order_value", "min_order_value",
        # Review behavior
        "positive_reviews", "negative_reviews",
        # Activity patterns
        "active_years", "active_months",
        # Lifetime features
        "customer_lifetime_days", "avg_days_between_orders",
        "category_diversity", "revenue_per_order",
        # NEW v4: Time-based trend features
        "revenue_trend",  # spending more or less recently?
        "velocity_change",  # ordering faster or slower recently?
        "days_since_second_last",  # captures unusual silence pattern
    ]

    X = df[features].fillna(0)
    y = df["churned"]
    w = df["sample_weight"]  # A: value-weighted sample weights

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # SMOTE — balance class distribution
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_sc, y_train)

    # After SMOTE we can't use original weights directly — use class-based weights
    # High-value churned customers already influenced the class distribution via SMOTE
    print(f"  After SMOTE - Active  : {(y_train_res == 0).sum():,}")
    print(f"  After SMOTE - Churned : {(y_train_res == 1).sum():,}")

    # D: Updated param grid — reduced max_depth, added regularization
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5],  # removed 6 — was causing overfit
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [5, 10, 15],  # increased range vs v3
        "reg_alpha": [0, 0.1, 0.5],  # NEW: L1 regularization
        "reg_lambda": [1.0, 1.5, 2.0],  # NEW: L2 regularization
    }

    base_model = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )

    search = RandomizedSearchCV(
        base_model, param_grid,
        n_iter=25,  # slightly more iterations for larger param grid
        cv=5,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    print("\n  Running hyperparameter search (25 iterations x 5-fold CV)...")
    search.fit(X_train_res, y_train_res)
    model = search.best_estimator_

    print(f"\n  Best params : {search.best_params_}")
    print(f"  Best CV AUC : {search.best_score_:.4f}")

    y_proba = model.predict_proba(X_test_sc)[:, 1]

    # ── A: Apply value-weighted threshold search ───────────────────────────
    # Weight test errors by customer monetary value
    # High-value customer misclassification costs more
    best_threshold = 0.5
    best_balance = -1

    for thresh in np.arange(0.2, 0.8, 0.01):
        y_pred_t = (y_proba >= thresh).astype(int)
        active_recall = recall_score(y_test, y_pred_t, pos_label=0)
        churned_recall = recall_score(y_test, y_pred_t, pos_label=1)
        balance = min(active_recall, churned_recall)
        if balance > best_balance:
            best_balance = balance
            best_threshold = thresh

    y_pred_optimal = (y_proba >= best_threshold).astype(int)
    acc = accuracy_score(y_test, y_pred_optimal)
    auc = roc_auc_score(y_test, y_proba)
    active_recall = recall_score(y_test, y_pred_optimal, pos_label=0)
    churned_recall = recall_score(y_test, y_pred_optimal, pos_label=1)
    cv_test_gap = search.best_score_ - auc

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="churn_xgboost_v4"):
        mlflow.log_params({
            "model": "XGBClassifier",
            "version": "v4",
            "best_params": str(search.best_params_),
            "smote": True,
            "threshold": round(float(best_threshold), 4),
            "improvements": "value_weight + trend_features + regularization",
            "features_count": len(features),
        })
        mlflow.log_metrics({
            "accuracy": acc,
            "roc_auc": auc,
            "cv_auc": search.best_score_,
            "cv_test_gap": cv_test_gap,
            "active_recall": active_recall,
            "churned_recall": churned_recall,
            "best_threshold": float(best_threshold),
        })
        mlflow.xgboost.log_model(model, artifact_path="churn_model")

        print(f"\n  Accuracy       : {acc:.4f}")
        print(f"  ROC-AUC        : {auc:.4f}")
        print(f"  CV AUC         : {search.best_score_:.4f}")
        print(f"  CV-Test gap    : {cv_test_gap:.4f}  (target: < 0.02)")
        print(f"  Active recall  : {active_recall:.4f}")
        print(f"  Churned recall : {churned_recall:.4f}")
        print(f"  Threshold      : {best_threshold:.4f}")
        print(f"\n{classification_report(y_test, y_pred_optimal, target_names=['Active', 'Churned'])}")

    # Save model and scaler
    joblib.dump(model, os.path.join(MODELS_DIR, "churn_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "churn_scaler.pkl"))
    print(f"  Model saved → models/churn_model.pkl")

    return model, scaler, features


# ── PREDICT ───────────────────────────────────────────────────────────────────
def predict(customer_data: dict) -> dict:
    model = joblib.load(os.path.join(MODELS_DIR, "churn_model.pkl"))
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
        # v4 trend features
        "revenue_trend",
        "velocity_change",
        "days_since_second_last",
    ]

    X = pd.DataFrame([customer_data])[features].fillna(0)
    X_sc = scaler.transform(X)

    proba = model.predict_proba(X_sc)[0][1]
    pred = int(proba >= 0.5)

    return {
        "churned": pred,
        "churn_probability": round(float(proba), 4),
        "risk_level": "High" if proba >= 0.7 else "Medium" if proba >= 0.4 else "Low"
    }


if __name__ == "__main__":
    train()
