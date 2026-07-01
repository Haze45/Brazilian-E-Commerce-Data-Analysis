"""
analytics.py  —  Olist BI Pipeline
All analytics queries using SQLAlchemy + PostgreSQL.
"""

import os, sys
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PG_URL

engine = create_engine(PG_URL)

def q(sql):
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


def kpi_summary() -> dict:
    df = q("""
        SELECT
            COUNT(DISTINCT order_id)                                        AS total_orders,
            COUNT(DISTINCT customer_id)                                     AS unique_customers,
            ROUND(SUM(revenue)::numeric, 2)                                 AS total_revenue,
            ROUND(AVG(price)::numeric, 2)                                   AS avg_order_value,
            ROUND(AVG(review_score)::numeric, 2)                            AS avg_review_score,
            ROUND(SUM(freight_value)::numeric, 2)                           AS total_freight,
            ROUND((SUM(CASE WHEN is_delivered=1 THEN 1.0 ELSE 0 END)
                  / COUNT(*) * 100)::numeric, 2)                            AS delivery_rate_pct
        FROM olist.fact_orders
    """)
    return {k: float(v) if v else 0 for k, v in df.iloc[0].to_dict().items()}


def monthly_revenue() -> list[dict]:
    df = q("""
        SELECT
            order_year   AS year,
            order_month  AS month,
            yyyymm,
            ROUND(SUM(revenue)::numeric, 2)       AS revenue,
            ROUND(SUM(freight_value)::numeric, 2) AS freight,
            COUNT(DISTINCT order_id)              AS orders,
            COUNT(DISTINCT customer_id)           AS customers
        FROM olist.fact_orders
        GROUP BY order_year, order_month, yyyymm
        ORDER BY order_year, order_month
    """)
    df["mom_growth_pct"] = df["revenue"].pct_change().mul(100).round(2)
    return df.to_dict(orient="records")


def category_performance() -> list[dict]:
    df = q("""
        SELECT
            category,
            COUNT(DISTINCT order_id)              AS orders,
            ROUND(SUM(revenue)::numeric, 2)       AS revenue,
            ROUND(AVG(price)::numeric, 2)         AS avg_price,
            ROUND(AVG(review_score)::numeric, 2)  AS avg_review,
            ROUND((SUM(revenue) * 100.0 /
                  (SELECT SUM(revenue) FROM olist.fact_orders))::numeric, 2)
                                                  AS revenue_share_pct
        FROM olist.fact_orders
        WHERE category IS NOT NULL
        GROUP BY category
        ORDER BY revenue DESC
    """)
    return df.to_dict(orient="records")


def top_products(n: int = 10) -> list[dict]:
    df = q(f"""
        SELECT
            f.product_id,
            p.category,
            COUNT(f.order_id)                     AS orders,
            ROUND(SUM(f.revenue)::numeric, 2)     AS revenue,
            ROUND(AVG(f.price)::numeric, 2)       AS avg_price,
            ROUND(AVG(f.review_score)::numeric, 2) AS avg_review
        FROM olist.fact_orders f
        JOIN olist.dim_products p ON f.product_id = p.product_id
        GROUP BY f.product_id, p.category
        ORDER BY revenue DESC
        LIMIT {n}
    """)
    return df.to_dict(orient="records")


def state_performance() -> list[dict]:
    df = q("""
        SELECT
            c.state,
            COUNT(DISTINCT f.order_id)              AS orders,
            COUNT(DISTINCT f.customer_id)           AS customers,
            ROUND(SUM(f.revenue)::numeric, 2)       AS revenue,
            ROUND(AVG(f.review_score)::numeric, 2)  AS avg_review,
            ROUND(AVG(f.delivery_delay_days)::numeric, 1) AS avg_delay_days
        FROM olist.fact_orders f
        JOIN olist.dim_customers c ON f.customer_id = c.customer_id
        GROUP BY c.state
        ORDER BY revenue DESC
    """)
    return df.to_dict(orient="records")


def payment_analysis() -> list[dict]:
    df = q("""
        SELECT
            payment_type,
            COUNT(DISTINCT order_id)                    AS orders,
            ROUND(SUM(revenue)::numeric, 2)             AS revenue,
            ROUND(AVG(payment_installments)::numeric,1) AS avg_installments,
            ROUND(AVG(price)::numeric, 2)               AS avg_order_value
        FROM olist.fact_orders
        WHERE payment_type IS NOT NULL
        GROUP BY payment_type
        ORDER BY revenue DESC
    """)
    return df.to_dict(orient="records")


def rfm_segmentation() -> list[dict]:
    df = q("""
        SELECT
            f.customer_id,
            c.customer_unique_id,
            DATE '2018-09-01' - MAX(f.order_date)   AS recency_days,
            COUNT(DISTINCT f.order_id)              AS frequency,
            ROUND(SUM(f.revenue)::numeric, 2)       AS monetary
        FROM olist.fact_orders f
        JOIN olist.dim_customers c ON f.customer_id = c.customer_id
        WHERE f.is_delivered = 1
        GROUP BY f.customer_id, c.customer_unique_id
    """)
    df["r_score"] = pd.qcut(df["recency_days"],  5, labels=[5,4,3,2,1]).astype(int)
    df["f_score"] = pd.qcut(df["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    df["m_score"] = pd.qcut(df["monetary"].rank(method="first"),  5, labels=[1,2,3,4,5]).astype(int)
    df["rfm"]     = df["r_score"] + df["f_score"] + df["m_score"]

    def seg(s):
        if s >= 13: return "Champions"
        if s >= 10: return "Loyal Customers"
        if s >= 7:  return "Potential Loyalists"
        if s >= 5:  return "At Risk"
        return "Lost"

    df["segment"] = df["rfm"].apply(seg)
    return (
        df.groupby("segment")
          .agg(customers=("customer_id","count"),
               avg_monetary=("monetary","mean"),
               avg_frequency=("frequency","mean"),
               avg_recency=("recency_days","mean"))
          .round(2).reset_index()
          .to_dict(orient="records")
    )


def delivery_performance() -> dict:
    df = q("""
        SELECT
            ROUND(AVG(delivery_delay_days)::numeric, 1)  AS avg_delay_days,
            ROUND((SUM(CASE WHEN delivery_delay_days > 0 THEN 1.0 ELSE 0 END)
                  / COUNT(*) * 100)::numeric, 2)         AS late_pct,
            ROUND((SUM(CASE WHEN delivery_delay_days <= 0 THEN 1.0 ELSE 0 END)
                  / COUNT(*) * 100)::numeric, 2)         AS on_time_pct
        FROM olist.fact_orders
        WHERE is_delivered = 1 AND delivery_delay_days IS NOT NULL
    """)
    return {k: float(v) if v else 0 for k, v in df.iloc[0].to_dict().items()}


def seller_leaderboard(n: int = 10) -> list[dict]:
    df = q(f"""
        SELECT
            f.seller_id,
            s.seller_state,
            COUNT(DISTINCT f.order_id)                  AS orders,
            ROUND(SUM(f.revenue)::numeric, 2)           AS revenue,
            ROUND(AVG(f.review_score)::numeric, 2)      AS avg_review,
            ROUND(AVG(f.delivery_delay_days)::numeric,1) AS avg_delay_days
        FROM olist.fact_orders f
        JOIN olist.dim_sellers s ON f.seller_id = s.seller_id
        GROUP BY f.seller_id, s.seller_state
        ORDER BY revenue DESC
        LIMIT {n}
    """)
    return df.to_dict(orient="records")


def yoy_comparison() -> list[dict]:
    df = q("""
        SELECT
            order_year                              AS year,
            ROUND(SUM(revenue)::numeric, 2)         AS revenue,
            COUNT(DISTINCT order_id)                AS orders,
            COUNT(DISTINCT customer_id)             AS customers,
            ROUND(AVG(review_score)::numeric, 2)    AS avg_review
        FROM olist.fact_orders
        GROUP BY order_year
        ORDER BY order_year
    """)
    df["yoy_growth"] = df["revenue"].pct_change().mul(100).round(2)
    return df.to_dict(orient="records")

# ── SELF-TEST ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    import math

    def clean(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(i) for i in obj]
        return obj

    print("\n── KPI Summary ──")
    print(json.dumps(clean(kpi_summary()), indent=2))

    print("\n── YoY Comparison ──")
    for r in clean(yoy_comparison()):
        print(r)

    print("\n── Top 3 Categories ──")
    for r in clean(category_performance())[:3]:
        print(r)

    print("\n── RFM Segments ──")
    for r in clean(rfm_segmentation()):
        print(r)

    print("\n── Delivery Performance ──")
    print(json.dumps(clean(delivery_performance()), indent=2))

    print("\n✅  Analytics OK")