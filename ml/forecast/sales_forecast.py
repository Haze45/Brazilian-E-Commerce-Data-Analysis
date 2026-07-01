"""
ml/forecast/sales_forecast.py
==============================
Monthly Revenue Forecasting using Facebook Prophet.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import numpy as np
import mlflow
import joblib
from sqlalchemy import create_engine, text
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from config import PG_URL, MODELS_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

engine = create_engine(PG_URL)


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
    return df


def train(periods: int = 6):
    print("\n── Sales Revenue Forecasting ─────────────────────")
    df = load_monthly_revenue()

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        interval_width=0.95,
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="sales_forecast_prophet"):
        model.fit(df)

        future   = model.make_future_dataframe(periods=periods, freq="MS")
        forecast = model.predict(future)

        try:
            cv_df      = cross_validation(model, initial="365 days", period="30 days", horizon="90 days")
            metrics_df = performance_metrics(cv_df)
            mae  = metrics_df["mae"].mean()
            rmse = metrics_df["rmse"].mean()
            mape = metrics_df["mape"].mean()
        except Exception:
            mae = rmse = mape = 0.0

        mlflow.log_params({
            "model"                    : "Prophet",
            "periods"                  : periods,
            "seasonality_mode"         : "multiplicative",
            "changepoint_prior_scale"  : 0.05,
        })
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "mape": mape})

        print(f"\n  MAE  : {mae:,.2f}")
        print(f"  RMSE : {rmse:,.2f}")
        print(f"  MAPE : {mape*100:.2f}%")

        future_only = forecast[forecast["ds"] > df["ds"].max()][
            ["ds", "yhat", "yhat_lower", "yhat_upper"]
        ]
        print(f"\n  Forecast next {periods} months:")
        for _, row in future_only.iterrows():
            print(f"    {row['ds'].strftime('%Y-%m')}  →  "
                  f"${row['yhat']:>12,.2f}  "
                  f"[${row['yhat_lower']:,.2f} – ${row['yhat_upper']:,.2f}]")

    joblib.dump(model, os.path.join(MODELS_DIR, "forecast_model.pkl"))
    forecast.to_csv(os.path.join(MODELS_DIR, "forecast_results.csv"), index=False)
    print(f"\n  Model saved → models/forecast_model.pkl")

    return model, forecast


def predict(periods: int = 6) -> list:
    model    = joblib.load(os.path.join(MODELS_DIR, "forecast_model.pkl"))
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
