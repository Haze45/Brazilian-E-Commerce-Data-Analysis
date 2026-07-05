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

Improvements v2:
  - Disabled yearly_seasonality (only 24 months — not enough data)
  - Added monthly seasonality manually (30-day period)
  - Changed to additive mode (more stable with limited data)
  - Increased changepoint_prior_scale for better trend flexibility
  - Confidence intervals in predict() response
  - model_to_json / model_from_json (avoids joblib stan_backend error)
  - tz_localize(None) to strip PostgreSQL timezone
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

engine    = create_engine(PG_URL)
MODEL_PATH    = os.path.join(MODELS_DIR, "forecast_model.json")
FORECAST_PATH = os.path.join(MODELS_DIR, "forecast_results.csv")


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

    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    df["y"]  = df["y"].astype(float)

    print(f"  Monthly data points : {len(df)}")
    print(f"  Date range          : {df['ds'].min().date()} → {df['ds'].max().date()}")
    print(f"  Avg monthly revenue : ${df['y'].mean():,.2f}")
    return df


def train(periods: int = 6):
    print("\n── Sales Revenue Forecasting ─────────────────────")
    df = load_monthly_revenue()

    # yearly_seasonality=False — only 24 months, not enough for reliable yearly cycle
    # Monthly seasonality added manually — we have enough months for this
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.1,
        interval_width=0.95,
    )
    model.add_seasonality(
        name="monthly",
        period=30.5,
        fourier_order=3,
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="sales_forecast_prophet_v2"):
        model.fit(df)

        future   = model.make_future_dataframe(periods=periods, freq="MS")
        forecast = model.predict(future)

        mae = rmse = mape = 0.0
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
            print(f"\n  MAE  : ${mae:,.2f}")
            print(f"  RMSE : ${rmse:,.2f}")
            print(f"  MAPE : {mape*100:.2f}%")
        except Exception as e:
            print(f"  Cross-validation skipped: {e}")

        mlflow.log_params({
            "model"                   : "Prophet v2",
            "yearly_seasonality"      : False,
            "monthly_seasonality"     : True,
            "seasonality_mode"        : "additive",
            "changepoint_prior_scale" : 0.1,
            "interval_width"          : 0.95,
            "fix"                     : "disabled yearly seasonality — only 24 months",
        })
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "mape": mape})

        future_only = forecast[forecast["ds"] > df["ds"].max()][
            ["ds", "yhat", "yhat_lower", "yhat_upper"]
        ]
        print(f"\n  Forecast next {periods} months (95% CI):")
        for _, row in future_only.iterrows():
            print(
                f"    {row['ds'].strftime('%Y-%m')}  →  "
                f"${row['yhat']:>12,.2f}  "
                f"[${row['yhat_lower']:,.2f} – ${row['yhat_upper']:,.2f}]"
            )

    with open(MODEL_PATH, "w") as f:
        f.write(model_to_json(model))

    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(
        FORECAST_PATH, index=False
    )

    print(f"\n  Model saved   → models/forecast_model.json")
    print(f"  Forecast CSV  → models/forecast_results.csv")
    return model, forecast


def predict(periods: int = 6) -> list[dict]:
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
            "lower_bound": round(float(row["yhat_lower"]), 2),
            "upper_bound": round(float(row["yhat_upper"]), 2),
        }
        for _, row in future_only.iterrows()
    ]


if __name__ == "__main__":
    train()
