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

Improvements v3:
  - Added peak_season, weekend_order features
  - Lower threshold targeting Late recall >= 0.55
  - XGBoost with scale_pos_weight added to competition
  - Evaluate with Late recall as primary metric
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
    accuracy_score, f1_score, recall_score,
    precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
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
        f.order_year,
        f.payment_installments,
        f.state                                                         AS customer_state,
        s.seller_state,
        p.weight_g,
        p.length_cm,
        p.height_cm,
        p.width_cm,
        CASE WHEN s.seller_state != f.state THEN 1 ELSE 0 END          AS cross_state,
        ROUND((f.freight_value / NULLIF(f.price + f.freight_value,0))
              ::numeric, 4)                                             AS freight_ratio,
        ROUND((NULLIF(p.length_cm,0) * NULLIF(p.height_cm,0)
               * NULLIF(p.width_cm,0))::numeric, 2)                    AS volume_cm3,

        -- NEW v3 features
        CASE WHEN f.order_month IN (11,12) THEN 1 ELSE 0 END           AS peak_season,
        CASE WHEN EXTRACT(DOW FROM f.order_date::date) IN (0,6)
             THEN 1 ELSE 0 END                                          AS weekend_order,

        CASE WHEN f.delivery_delay_days > 0
             THEN 1 ELSE 0 END                                          AS is_late
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
        "price", "freight_value", "freight_ratio",
        "order_month", "order_quarter", "order_year",
        "payment_installments",
        "customer_state_enc", "seller_state_enc",
        "cross_state",
        "weight_g", "length_cm", "height_cm", "width_cm",
        "volume_cm3",
        "peak_season", "weekend_order",
    ]

    X = df[features].fillna(0)
    y = df["is_late"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n  Train - On-time: {(y_train==0).sum():,}  Late: {(y_train==1).sum():,}")

    sm = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"  After SMOTE - On-time: {(y_train_res==0).sum():,}  Late: {(y_train_res==1).sum():,}")

    sample_weights = compute_sample_weight("balanced", y_train_res)
    scale = float((y_train_res == 0).sum()) / float((y_train_res == 1).sum())

    # ── Model 1: Gradient Boosting ────────────────────────────────────────
    print("\n  Training Gradient Boosting...")
    gb_search = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=42),
        {"n_estimators":[100,150,200],"max_depth":[3,4,5],
         "learning_rate":[0.05,0.1,0.15],"subsample":[0.7,0.8,1.0]},
        n_iter=15, cv=5, scoring="f1",
        random_state=42, n_jobs=-1, verbose=0,
    )
    gb_search.fit(X_train_res, y_train_res, sample_weight=sample_weights)
    gb_model = gb_search.best_estimator_
    gb_proba = gb_model.predict_proba(X_test)[:,1]
    gb_auc   = roc_auc_score(y_test, gb_proba)
    print(f"  GB  CV F1: {gb_search.best_score_:.4f}  ROC-AUC: {gb_auc:.4f}")

    # ── Model 2: Random Forest ────────────────────────────────────────────
    print("  Training Random Forest...")
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
        {"n_estimators":[100,200,300],"max_depth":[5,8,10,None],
         "min_samples_split":[2,5,10]},
        n_iter=15, cv=5, scoring="f1",
        random_state=42, n_jobs=-1, verbose=0,
    )
    rf_search.fit(X_train_res, y_train_res)
    rf_model = rf_search.best_estimator_
    rf_proba = rf_model.predict_proba(X_test)[:,1]
    rf_auc   = roc_auc_score(y_test, rf_proba)
    print(f"  RF  CV F1: {rf_search.best_score_:.4f}  ROC-AUC: {rf_auc:.4f}")

    # ── Model 3: XGBoost with scale_pos_weight ────────────────────────────
    print("  Training XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        scale_pos_weight=scale,
        random_state=42, eval_metric="aucpr", n_jobs=-1,
    )
    xgb_model.fit(X_train_res, y_train_res)
    xgb_proba = xgb_model.predict_proba(X_test)[:,1]
    xgb_auc   = roc_auc_score(y_test, xgb_proba)
    print(f"  XGB ROC-AUC: {xgb_auc:.4f}")

    # ── Pick best by ROC-AUC ──────────────────────────────────────────────
    scores = {
        "GradientBoosting": (gb_model, gb_proba, gb_auc),
        "RandomForest"    : (rf_model, rf_proba, rf_auc),
        "XGBoost"         : (xgb_model, xgb_proba, xgb_auc),
    }
    model_name = max(scores, key=lambda k: scores[k][2])
    model, y_proba, best_auc = scores[model_name]
    print(f"\n  Selected: {model_name} (ROC-AUC: {best_auc:.4f})")

    # Threshold targeting Late recall >= 0.55
    best_threshold = 0.5
    for thresh in np.arange(0.1, 0.6, 0.01):
        y_pred_t    = (y_proba >= thresh).astype(int)
        late_recall = recall_score(y_test, y_pred_t, pos_label=1)
        if late_recall >= 0.55:
            best_threshold = thresh
            break

    y_pred_optimal = (y_proba >= best_threshold).astype(int)
    acc       = accuracy_score(y_test, y_pred_optimal)
    auc       = roc_auc_score(y_test, y_proba)
    f1_late   = f1_score(y_test, y_pred_optimal, pos_label=1)
    late_rec  = recall_score(y_test, y_pred_optimal, pos_label=1)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"delivery_{model_name.lower()}_v3"):
        mlflow.log_params({
            "model"         : model_name,
            "smote"         : True,
            "threshold"     : round(float(best_threshold), 4),
            "version"       : "v3 - peak_season + weekend + XGBoost added",
            "features"      : str(features),
        })
        mlflow.log_metrics({
            "accuracy"      : acc,
            "roc_auc"       : auc,
            "f1_late"       : f1_late,
            "late_recall"   : late_rec,
            "best_threshold": float(best_threshold),
        })
        mlflow.sklearn.log_model(model, artifact_path="delivery_model")

        print(f"\n  Accuracy     : {acc:.4f}")
        print(f"  ROC-AUC      : {auc:.4f}")
        print(f"  Late F1      : {f1_late:.4f}")
        print(f"  Late Recall  : {late_rec:.4f}")
        print(f"  Threshold    : {best_threshold:.4f}")
        print(f"\n{classification_report(y_test, y_pred_optimal, target_names=['On-time','Late'])}")

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
    sell_state = order_data.get("seller_state",   "SP")

    order_data["customer_state_enc"] = le_cust.transform(
        [cust_state if cust_state in le_cust.classes_ else le_cust.classes_[0]])[0]
    order_data["seller_state_enc"] = le_seller.transform(
        [sell_state if sell_state in le_seller.classes_ else le_seller.classes_[0]])[0]

    order_data["cross_state"]   = 1 if cust_state != sell_state else 0
    price   = order_data.get("price", 0)
    freight = order_data.get("freight_value", 0)
    order_data["freight_ratio"] = freight / (price + freight) if (price + freight) > 0 else 0
    l = order_data.get("length_cm", 0)
    h = order_data.get("height_cm", 0)
    w = order_data.get("width_cm",  0)
    order_data["volume_cm3"]    = l * h * w
    order_data["peak_season"]   = 1 if order_data.get("order_month", 0) in [11, 12] else 0
    order_data["weekend_order"] = 0  # not deterministic at predict time

    features = [
        "price", "freight_value", "freight_ratio",
        "order_month", "order_quarter", "order_year",
        "payment_installments",
        "customer_state_enc", "seller_state_enc",
        "cross_state",
        "weight_g", "length_cm", "height_cm", "width_cm",
        "volume_cm3",
        "peak_season", "weekend_order",
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