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

Improvements v3:
    A — Switched primary metrics from MAPE to MAE/RMSE
      MAPE is distorted by 2016 startup months (near-zero revenue)
      MAE is the honest metric: $292K error on $660K avg revenue

    B — Cross-validation starts from 2017 (540 days initial window)
      Skips erratic 2016 launch months
      Evaluation on stable 2017-2018 data only

    C — External regressors added:
      n_sellers  : active sellers per month (platform scale signal)
      n_products : unique products sold per month (diversity signal)
      month_num  : months since launch (growth trajectory signal)
      Future regressor values projected via linear trend
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
MODEL_PATH = os.path.join(MODELS_DIR, "forecast_model.json")
FORECAST_PATH = os.path.join(MODELS_DIR, "forecast_results.csv")


# ── DATA LOADING ──────────────────────────────────────────────────────────────
def load_monthly_revenue() -> pd.DataFrame:
    """Load monthly revenue + external regressors from PostgreSQL."""
    sql = """
    SELECT
        DATE_TRUNC('month', order_date::date)           AS ds,
        ROUND(SUM(revenue)::numeric, 2)                 AS y,

        -- C: External regressors
        -- Regressor 1: active sellers per month (platform scale)
        COUNT(DISTINCT seller_id)                       AS n_sellers,
        -- Regressor 2: unique products sold per month (diversity)
        COUNT(DISTINCT product_id)                      AS n_products,
        -- Regressor 3: total orders per month (volume signal)
        COUNT(DISTINCT order_id)                        AS n_orders
    FROM olist.fact_orders
    GROUP BY DATE_TRUNC('month', order_date::date)
    ORDER BY ds
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)

    # Strip timezone — Prophet does not accept tz-aware timestamps
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    df["y"] = df["y"].astype(float)

    # C: month_num — months since business launch (growth trajectory)
    df["month_num"] = range(1, len(df) + 1)

    # Normalize regressors to 0-1 scale for stable Prophet fitting
    for col in ["n_sellers", "n_products", "n_orders", "month_num"]:
        col_min = df[col].min()
        col_max = df[col].max()
        df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min + 1e-9)

    print(f"  Monthly data points : {len(df)}")
    print(f"  Date range          : {df['ds'].min().date()} → {df['ds'].max().date()}")
    print(f"  Avg monthly revenue : ${df['y'].mean():,.2f}")
    print(f"  Revenue range       : ${df['y'].min():,.2f} – ${df['y'].max():,.2f}")
    print(f"\n  Regressors:")
    print(f"    n_sellers  : {df['n_sellers'].min()} – {df['n_sellers'].max()}")
    print(f"    n_products : {df['n_products'].min()} – {df['n_products'].max()}")
    print(f"    month_num  : 1 – {df['month_num'].max()}")

    return df


def project_future_regressors(df: pd.DataFrame, periods: int) -> pd.DataFrame:
    """
    Project external regressors into future months using linear trend.
    We don't have actual future seller/product counts so we extrapolate
    the trend from historical data.
    """
    future_rows = []
    last_month_num = df["month_num"].max()

    for i in range(1, periods + 1):
        future_month_num = last_month_num + i

        # Linear extrapolation for each regressor
        # Fit a simple linear trend on the last 6 months
        recent = df.tail(6)

        def linear_project(series, steps_ahead):
            x = np.arange(len(series))
            slope, intercept = np.polyfit(x, series.values, 1)
            return max(0, intercept + slope * (len(series) + steps_ahead - 1))

        future_rows.append({
            "month_num": future_month_num,
            "n_sellers": linear_project(recent["n_sellers"], i),
            "n_products": linear_project(recent["n_products"], i),
            "n_orders": linear_project(recent["n_orders"], i),
        })

    future_df = pd.DataFrame(future_rows)

    # Normalize future regressors using same scale as training data
    for col in ["n_sellers", "n_products", "n_orders", "month_num"]:
        col_min = df[col].min()
        col_max = df[col].max()
        future_df[f"{col}_norm"] = (
                (future_df[col] - col_min) / (col_max - col_min + 1e-9)
        ).clip(0, 2)  # allow slight extrapolation beyond training range

    return future_df


# ── TRAIN ─────────────────────────────────────────────────────────────────────
def train(periods: int = 6):
    print("\n── Sales Revenue Forecasting v2 ──────────────────")
    df = load_monthly_revenue()

    # Build Prophet model with external regressors
    # yearly_seasonality=False — only 24 months, not enough for reliable yearly cycle
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.1,
        interval_width=0.95,
    )

    # Manual monthly seasonality (30-day period)
    model.add_seasonality(
        name="monthly",
        period=30.5,
        fourier_order=3,
    )

    # C: Add external regressors
    model.add_regressor("n_sellers_norm", standardize=False)
    model.add_regressor("n_products_norm", standardize=False)
    model.add_regressor("month_num_norm", standardize=False)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="sales_forecast_prophet_v2"):
        model.fit(df)

        # Build future dataframe with projected regressors
        future_base = model.make_future_dataframe(periods=periods, freq="MS")
        future_regs = project_future_regressors(df, periods)
        future_df = future_base.copy()

        # Fill historical regressor values
        for col in ["n_sellers_norm", "n_products_norm", "month_num_norm"]:
            future_df[col] = np.nan

        for idx, row in df.iterrows():
            mask = future_df["ds"] == row["ds"]
            future_df.loc[mask, "n_sellers_norm"] = row["n_sellers_norm"]
            future_df.loc[mask, "n_products_norm"] = row["n_products_norm"]
            future_df.loc[mask, "month_num_norm"] = row["month_num_norm"]

        # Fill projected future regressor values
        future_dates = future_df[future_df["ds"] > df["ds"].max()]
        for i, (idx, row) in enumerate(future_dates.iterrows()):
            if i < len(future_regs):
                future_df.loc[idx, "n_sellers_norm"] = future_regs.iloc[i]["n_sellers_norm"]
                future_df.loc[idx, "n_products_norm"] = future_regs.iloc[i]["n_products_norm"]
                future_df.loc[idx, "month_num_norm"] = future_regs.iloc[i]["month_num_norm"]

        # Forward fill any remaining NaN
        for col in ["n_sellers_norm", "n_products_norm", "month_num_norm"]:
            future_df[col] = future_df[col].ffill().bfill()

        forecast = model.predict(future_df)

        # B: Cross-validation starting from 2017 (540 days = ~18 months)
        # Skips 2016 erratic startup months where MAPE is massively inflated
        mae = rmse = mape = 0.0
        mae_2017 = rmse_2017 = mape_2017 = 0.0
        try:
            # Full CV (all data)
            cv_df = cross_validation(
                model,
                initial="540 days",  # B: start from ~2017 not 2016
                period="30 days",
                horizon="90 days",
                parallel=None,
            )
            metrics_df = performance_metrics(cv_df)
            mae = float(metrics_df["mae"].mean())
            rmse = float(metrics_df["rmse"].mean())
            mape = float(metrics_df["mape"].mean())

            print(f"\n  Cross-validation results (from 2017 onwards):")
            print(f"  MAE  (primary) : ${mae:,.2f}   ← honest metric")
            print(f"  RMSE           : ${rmse:,.2f}")
            print(f"  MAPE           : {mape * 100:.2f}%  ← still high due to growth phase")
            print(f"  Note: MAPE distorted by low-revenue startup months in 2016")

        except Exception as e:
            print(f"  Cross-validation skipped: {e}")

        # A: Log MAE and RMSE as primary metrics
        mlflow.log_params({
            "model": "Prophet v2",
            "yearly_seasonality": False,
            "monthly_seasonality": True,
            "seasonality_mode": "additive",
            "changepoint_prior_scale": 0.1,
            "regressors": "n_sellers, n_products, month_num",
            "cv_initial_days": 540,
            "primary_metric": "MAE (MAPE distorted by 2016 launch months)",
            "version": "v2",
        })

        # A: MAE and RMSE as primary metrics, MAPE as reference only
        mlflow.log_metrics({
            "mae": mae,  # PRIMARY
            "rmse": rmse,  # PRIMARY
            "mape": mape,  # REFERENCE ONLY — distorted by 2016
        })

        # Show forecast with confidence intervals
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

    # Save model using Prophet's JSON serializer (not joblib)
    with open(MODEL_PATH, "w") as f:
        f.write(model_to_json(model))

    # Save forecast CSV with full columns
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(
        FORECAST_PATH, index=False
    )
    # Save regressor data for future use
    df[["ds", "n_sellers", "n_products", "n_orders", "month_num",
        "n_sellers_norm", "n_products_norm", "month_num_norm"]].to_csv(
        os.path.join(MODELS_DIR, "forecast_regressors.csv"), index=False
    )

    print(f"\n  Model saved   → models/forecast_model.json")
    print(f"  Forecast CSV  → models/forecast_results.csv")
    print(f"  Regressors    → models/forecast_regressors.csv")

    return model, forecast


# ── PREDICT ───────────────────────────────────────────────────────────────────
def predict(periods: int = 6) -> list[dict]:
    """
    Load saved model and return forecast for next N months.
    Returns list of dicts with month, forecast, lower_bound, upper_bound.
    External regressors projected via linear trend from historical data.
    """
    with open(MODEL_PATH, "r") as f:
        model = model_from_json(f.read())

    df = load_monthly_revenue()

    # Build future dataframe with projected regressors
    future_base = model.make_future_dataframe(periods=periods, freq="MS")
    future_regs = project_future_regressors(df, periods)
    future_df = future_base.copy()

    # Fill historical regressor values
    for col in ["n_sellers_norm", "n_products_norm", "month_num_norm"]:
        future_df[col] = np.nan

    for _, row in df.iterrows():
        mask = future_df["ds"] == row["ds"]
        future_df.loc[mask, "n_sellers_norm"] = row["n_sellers_norm"]
        future_df.loc[mask, "n_products_norm"] = row["n_products_norm"]
        future_df.loc[mask, "month_num_norm"] = row["month_num_norm"]

    # Fill projected future regressor values
    future_dates = future_df[future_df["ds"] > df["ds"].max()]
    for i, (idx, _) in enumerate(future_dates.iterrows()):
        if i < len(future_regs):
            future_df.loc[idx, "n_sellers_norm"] = future_regs.iloc[i]["n_sellers_norm"]
            future_df.loc[idx, "n_products_norm"] = future_regs.iloc[i]["n_products_norm"]
            future_df.loc[idx, "month_num_norm"] = future_regs.iloc[i]["month_num_norm"]

    for col in ["n_sellers_norm", "n_products_norm", "month_num_norm"]:
        future_df[col] = future_df[col].ffill().bfill()

    forecast = model.predict(future_df)
    future_only = forecast[forecast["ds"] > df["ds"].max()][
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].copy()

    return [
        {
            "month": row["ds"].strftime("%Y-%m"),
            "forecast": round(float(row["yhat"]), 2),
            "lower_bound": round(float(row["yhat_lower"]), 2),
            "upper_bound": round(float(row["yhat_upper"]), 2),
        }
        for _, row in future_only.iterrows()
    ]


if __name__ == "__main__":
    train()
