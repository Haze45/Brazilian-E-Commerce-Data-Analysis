"""
etl_pipeline.py  —  Olist Brazilian E-Commerce
================================================
Extract → Transform → Load into PostgreSQL (star schema)

Run: python etl/etl_pipeline.py

Tables created in PostgreSQL (schema: olist):
  fact_orders        — one row per order-item (~112K rows)
  dim_customers      — ~99K customers  WITH lat/lng
  dim_products       — ~32K products   with English categories
  dim_sellers        — ~3K  sellers    WITH lat/lng
  dim_date           — date dimension 2016–2018
  dim_geolocation    — zip → lat/lng lookup (deduplicated from ~1M rows)
"""

import os, sys, logging
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PG_URL, DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def extract():
    log.info("EXTRACT: reading 9 Olist CSV files from data/")

    def read(name):
        path = os.path.join(DATA_DIR, name)
        df = pd.read_csv(path)
        log.info(f"  {name:<50} {len(df):>8,} rows")
        return df

    return (
        read("olist_orders_dataset.csv"),
        read("olist_order_items_dataset.csv"),
        read("olist_order_payments_dataset.csv"),
        read("olist_order_reviews_dataset.csv"),
        read("olist_customers_dataset.csv"),
        read("olist_products_dataset.csv"),
        read("olist_sellers_dataset.csv"),
        read("olist_geolocation_dataset.csv"),
        read("product_category_name_translation.csv"),
    )


def transform(orders, items, payments, reviews, customers,
              products, sellers, geo, cat_trans):
    log.info("TRANSFORM: cleaning & building star schema...")

    # ── dim_geolocation ────────────────────────────────────────────────────
    dim_geo = (
        geo
        .rename(columns={
            "geolocation_zip_code_prefix": "zip_code_prefix",
            "geolocation_lat"            : "lat",
            "geolocation_lng"            : "lng",
            "geolocation_city"           : "city",
            "geolocation_state"          : "state",
        })
        .groupby(["zip_code_prefix", "state"], as_index=False)
        .agg(lat=("lat", "mean"), lng=("lng", "mean"), city=("city", "first"))
        [["zip_code_prefix", "state", "city", "lat", "lng"]]
    )
    dim_geo["lat"] = dim_geo["lat"].round(6)
    dim_geo["lng"] = dim_geo["lng"].round(6)
    # Deduplicate zip_code_prefix (one zip can map to multiple states after groupby)
    dim_geo = dim_geo.drop_duplicates(subset=["zip_code_prefix"], keep="first")
    log.info(f"  dim_geolocation: {len(dim_geo):,}  (deduplicated from {len(geo):,} raw rows)")

    # ── dim_customers  (with lat/lng joined from geo) ──────────────────────
    dim_customers = (
        customers[[
            "customer_id", "customer_unique_id",
            "customer_city", "customer_state",
            "customer_zip_code_prefix",
        ]]
        .drop_duplicates("customer_id")
        .rename(columns={
            "customer_city"            : "city",
            "customer_state"           : "state",
            "customer_zip_code_prefix" : "zip_code_prefix",
        })
        .merge(
            dim_geo[["zip_code_prefix", "lat", "lng"]],
            on="zip_code_prefix",
            how="left",
        )
        .drop_duplicates("customer_id")
    )
    dim_customers["lat"] = dim_customers["lat"].round(6)
    dim_customers["lng"] = dim_customers["lng"].round(6)
    log.info(f"  dim_customers  : {len(dim_customers):,}  (with lat/lng)")

    # ── dim_sellers  (with lat/lng joined from geo) ────────────────────────
    dim_sellers = (
        sellers[[
            "seller_id", "seller_city", "seller_state",
            "seller_zip_code_prefix",
        ]]
        .drop_duplicates("seller_id")
        .rename(columns={"seller_zip_code_prefix": "zip_code_prefix"})
        .merge(
            dim_geo[["zip_code_prefix", "lat", "lng"]].rename(
                columns={"lat": "seller_lat", "lng": "seller_lng"}
            ),
            on="zip_code_prefix",
            how="left",
        )
        .drop_duplicates("seller_id")
    )
    dim_sellers["seller_lat"] = dim_sellers["seller_lat"].round(6)
    dim_sellers["seller_lng"] = dim_sellers["seller_lng"].round(6)
    log.info(f"  dim_sellers    : {len(dim_sellers):,}  (with lat/lng)")

    # ── dim_products  (Portuguese → English categories) ───────────────────
    dim_products = products.merge(cat_trans, on="product_category_name", how="left")
    dim_products["category"] = (
        dim_products["product_category_name_english"]
        .fillna(dim_products["product_category_name"])
        .str.replace("_", " ")
        .str.title()
    )
    dim_products = (
        dim_products[[
            "product_id", "category",
            "product_weight_g", "product_length_cm",
            "product_height_cm", "product_width_cm",
        ]]
        .rename(columns={
            "product_weight_g"  : "weight_g",
            "product_length_cm" : "length_cm",
            "product_height_cm" : "height_cm",
            "product_width_cm"  : "width_cm",
        })
        .drop_duplicates("product_id")
        .copy()
    )
    log.info(f"  dim_products   : {len(dim_products):,}")

    # ── dim_date ───────────────────────────────────────────────────────────
    dates = pd.date_range("2016-01-01", "2018-12-31", freq="D")
    dim_date = pd.DataFrame({
        "date_id"   : dates.strftime("%Y%m%d").astype(int),
        "date"      : dates.date,
        "year"      : dates.year,
        "quarter"   : dates.quarter,
        "month"     : dates.month,
        "month_name": dates.strftime("%B"),
        "week"      : dates.isocalendar().week.astype(int).values,
        "day_name"  : dates.strftime("%A"),
        "is_weekend": (dates.dayofweek >= 5).astype(int),
        "yyyymm"    : dates.to_period("M").astype(str),
    })
    log.info(f"  dim_date       : {len(dim_date):,}")

    # ── payments: aggregate per order ─────────────────────────────────────
    pay_agg = (
        payments
        .groupby("order_id")
        .agg(
            total_payment        =("payment_value",        "sum"),
            payment_installments =("payment_installments", "max"),
            payment_type         =("payment_type",         "first"),
        )
        .reset_index()
    )

    # ── reviews: best score per order ─────────────────────────────────────
    rev_clean = (
        reviews[["order_id", "review_score"]]
        .dropna()
        .groupby("order_id")["review_score"]
        .max()
        .reset_index()
    )

    # ── fact_orders  (join 6 sources) ──────────────────────────────────────
    log.info("  Building fact_orders (joining 6 tables)...")

    for col in ["order_purchase_timestamp",
                "order_delivered_customer_date",
                "order_estimated_delivery_date"]:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    fact = (
        items
        .merge(orders[[
            "order_id", "customer_id", "order_status",
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]], on="order_id", how="left")
        .merge(rev_clean, on="order_id", how="left")
        .merge(pay_agg,   on="order_id", how="left")
        .merge(dim_customers[["customer_id", "state", "lat", "lng"]],
               on="customer_id", how="left")
        .merge(dim_products[["product_id", "category"]],
               on="product_id",  how="left")
    )

    fact["order_date"]    = fact["order_purchase_timestamp"].dt.date
    fact["order_year"]    = fact["order_purchase_timestamp"].dt.year
    fact["order_month"]   = fact["order_purchase_timestamp"].dt.month
    fact["order_quarter"] = fact["order_purchase_timestamp"].dt.quarter
    fact["yyyymm"]        = fact["order_purchase_timestamp"].dt.to_period("M").astype(str)
    fact["revenue"]       = (fact["price"] + fact["freight_value"]).round(2)
    fact["is_delivered"]  = (fact["order_status"] == "delivered").astype(int)
    fact["delivery_delay_days"] = (
        (fact["order_delivered_customer_date"] - fact["order_estimated_delivery_date"])
        .dt.days
    )

    before = len(fact)
    fact = fact.dropna(subset=["order_id", "customer_id", "product_id", "seller_id"])
    fact = fact.drop_duplicates(subset=["order_id", "order_item_id"])
    log.info(f"  Dropped {before - len(fact):,} rows (missing keys / duplicates)")

    median_score = fact["review_score"].median()
    missing = fact["review_score"].isna().sum()
    fact["review_score"] = fact["review_score"].fillna(median_score)
    log.info(f"  Imputed {missing:,} missing review_score with median={median_score}")

    fact = fact.reset_index(drop=True)
    fact.insert(0, "fact_id", fact.index + 1)

    keep = [
        "fact_id", "order_id", "order_item_id",
        "customer_id", "product_id", "seller_id",
        "order_date", "order_year", "order_month", "order_quarter", "yyyymm",
        "price", "freight_value", "revenue",
        "review_score", "order_status", "is_delivered", "delivery_delay_days",
        "total_payment", "payment_installments", "payment_type",
        "category", "state", "lat", "lng",
    ]
    fact = fact[[c for c in keep if c in fact.columns]]
    log.info(f"  fact_orders    : {len(fact):,} rows  |  {fact['order_id'].nunique():,} unique orders")

    return fact, dim_customers, dim_products, dim_sellers, dim_date, dim_geo


def load(fact, dim_customers, dim_products, dim_sellers, dim_date, dim_geo):
    log.info(f"LOAD → PostgreSQL  ({PG_URL.split('@')[-1]})")

    engine = create_engine(PG_URL)

    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS olist"))
        conn.commit()

    kw = dict(schema="olist", if_exists="replace", index=False)

    log.info("  Writing dimension tables...")
    dim_geo.to_sql("dim_geolocation", engine, **kw, chunksize=5000, method="multi")
    dim_customers.to_sql("dim_customers", engine, **kw)
    dim_products.to_sql("dim_products",   engine, **kw)
    dim_sellers.to_sql("dim_sellers",     engine, **kw)
    dim_date.to_sql("dim_date",           engine, **kw)

    log.info("  Writing fact_orders (~112K rows, may take ~15s)...")
    fact.to_sql("fact_orders", engine, **kw, chunksize=5000, method="multi")

    log.info("  Creating indexes...")
    with engine.connect() as conn:
        for sql in [
            "CREATE INDEX IF NOT EXISTS idx_fo_customer  ON olist.fact_orders(customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_fo_product   ON olist.fact_orders(product_id)",
            "CREATE INDEX IF NOT EXISTS idx_fo_seller    ON olist.fact_orders(seller_id)",
            "CREATE INDEX IF NOT EXISTS idx_fo_date      ON olist.fact_orders(order_date)",
            "CREATE INDEX IF NOT EXISTS idx_fo_yyyymm    ON olist.fact_orders(yyyymm)",
            "CREATE INDEX IF NOT EXISTS idx_fo_status    ON olist.fact_orders(order_status)",
            "CREATE INDEX IF NOT EXISTS idx_geo_zip      ON olist.dim_geolocation(zip_code_prefix)",
            "CREATE INDEX IF NOT EXISTS idx_cust_zip     ON olist.dim_customers(zip_code_prefix)",
        ]:
            conn.execute(text(sql))
        conn.commit()

    with engine.connect() as conn:
        for tbl in ["fact_orders", "dim_customers", "dim_products",
                    "dim_sellers", "dim_date", "dim_geolocation"]:
            n = conn.execute(text(f"SELECT COUNT(*) FROM olist.{tbl}")).scalar()
            log.info(f"  olist.{tbl:<20} {n:>8,} rows ✓")

    log.info("LOAD complete ✅")


def run_pipeline():
    t0 = datetime.now()
    log.info("=" * 58)
    log.info("  OLIST BI PIPELINE — ETL START")
    log.info("=" * 58)

    tables = extract()
    fact, dim_customers, dim_products, dim_sellers, dim_date, dim_geo = transform(*tables)
    load(fact, dim_customers, dim_products, dim_sellers, dim_date, dim_geo)

    elapsed = (datetime.now() - t0).total_seconds()
    log.info(f"Pipeline finished in {elapsed:.1f}s  🎉")


if __name__ == "__main__":
    run_pipeline()
