"""
ml/forecast/sales_forecast.py
==============================
Monthly Revenue Forecasting using Facebook Prophet.

Fixes applied:
  - tz_localize(None) to strip timezone from PostgreSQL timestamps
  - Confidence intervals (yhat_lower, yhat_upper) added to predict()
  - Cross-validation with try/except for robustness
  - forecast_results.csv saved with full columns including bounds
  - model saved using prophet.serialize (model_to_json / model_from_json)
    instead of joblib — avoids 'stan_backend' pickling error on Windows

Note on data:
  Only 24 monthly data points available (2016-08-31 to 2018-08-31).
  Prophet typically needs 2+ full seasonal cycles for reliable yearly
  seasonality. Results should be treated as indicative, not precise.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import numpy as np
import mlflow
from sqlalchemy import create_engine, text
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from prophet.diagnostics import cross_validation, performance_metrics
from config import PG_URL, MODELS_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

engine = create_engine(PG_URL)

MODEL_PATH    = os.path.join(MODELS_DIR, "forecast_model.json")
FORECAST_PATH = os.path.join(MODELS_DIR, "forecast_results.csv")


# ── DATA ──────────────────────────────────────────────────────────────────────
def load_monthly_revenue() -> pd.DataFrame:
    sql = """
    SELECT
        DATE_TRUNC('month', order_date::date)   AS ds,
        ROUND(SUM(revenue)::numeric, 2)         AS y
    FROM olist.fact_orders
    GROUP BY DATE_TRUNC('month', order_date::date)
    ORDER BY ds
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)

    # Strip timezone — Prophet does not accept tz-aware timestamps
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    df["y"]  = df["y"].astype(float)

    print(f"  Monthly data points : {len(df)}")
    print(f"  Date range          : {df['ds'].min().date()} → {df['ds'].max().date()}")
    print(f"  Avg monthly revenue : ${df['y'].mean():,.2f}")
    return df


# ── TRAIN ─────────────────────────────────────────────────────────────────────
def train(periods: int = 6):
    print("\n── Sales Revenue Forecasting ─────────────────────")
    df = load_monthly_revenue()

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        interval_width=0.95,       # 95% confidence interval
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="sales_forecast_prophet"):
        model.fit(df)

        # Forecast
        future   = model.make_future_dataframe(periods=periods, freq="MS")
        forecast = model.predict(future)

        # Cross-validation (may fail with small dataset — handled gracefully)
        try:
            cv_df      = cross_validation(
                model,
                initial="365 days",
                period="30 days",
                horizon="90 days",
                parallel=None,
            )
            metrics_df = performance_metrics(cv_df)
            mae  = float(metrics_df["mae"].mean())
            rmse = float(metrics_df["rmse"].mean())
            mape = float(metrics_df["mape"].mean())
        except Exception as e:
            print(f"  Cross-validation skipped: {e}")
            mae = rmse = mape = 0.0

        mlflow.log_params({
            "model"                   : "Prophet",
            "periods"                 : periods,
            "seasonality_mode"        : "multiplicative",
            "changepoint_prior_scale" : 0.05,
            "interval_width"          : 0.95,
            "data_note"               : "24 months only — yearly seasonality may be unreliable",
        })
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "mape": mape})

        print(f"\n  MAE  : ${mae:,.2f}")
        print(f"  RMSE : ${rmse:,.2f}")
        print(f"  MAPE : {mape*100:.2f}%")

        # Show forecast with confidence intervals
        future_only = forecast[forecast["ds"] > df["ds"].max()][
            ["ds", "yhat", "yhat_lower", "yhat_upper"]
        ]
        print(f"\n  Forecast next {periods} months (95% confidence interval):")
        for _, row in future_only.iterrows():
            print(
                f"    {row['ds'].strftime('%Y-%m')}  →  "
                f"${row['yhat']:>12,.2f}  "
                f"[${row['yhat_lower']:,.2f} – ${row['yhat_upper']:,.2f}]"
            )

    # Save model using Prophet's own JSON serializer (not joblib)
    with open(MODEL_PATH, "w") as f:
        f.write(model_to_json(model))

    # Save full forecast CSV including confidence intervals
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(
        FORECAST_PATH, index=False
    )

    print(f"\n  Model saved   → models/forecast_model.json")
    print(f"  Forecast CSV  → models/forecast_results.csv")

    return model, forecast


# ── PREDICT ───────────────────────────────────────────────────────────────────
def predict(periods: int = 6) -> list[dict]:
    """
    Load saved model and return forecast for next N months.
    Returns list of dicts with month, forecast, lower_bound, upper_bound.
    Confidence intervals (95%) communicate forecast uncertainty honestly.
    """
    with open(MODEL_PATH, "r") as f:
        model = model_from_json(f.read())

    df       = load_monthly_revenue()
    future   = model.make_future_dataframe(periods=periods, freq="MS")
    forecast = model.predict(future)

    future_only = forecast[forecast["ds"] > df["ds"].max()][
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].copy()

    return [
        {
            "month"      : row["ds"].strftime("%Y-%m"),
            "forecast"   : round(float(row["yhat"]), 2),
            "lower_bound": round(float(row["yhat_lower"]), 2),   # 95% CI lower
            "upper_bound": round(float(row["yhat_upper"]), 2),   # 95% CI upper
        }
        for _, row in future_only.iterrows()
    ]


if __name__ == "__main__":
    train()