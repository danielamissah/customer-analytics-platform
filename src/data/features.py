"""
Feature Engineering
===================
Builds user-level feature vectors for churn and LTV models
from raw events, transactions, and subscription data.
"""

import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from loguru import logger
from sqlalchemy import create_engine


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_engine(config: dict):
    url = os.environ.get("DATABASE_URL", config["database"]["url"])
    return create_engine(url)


PLAN_MAP    = {"free": 0, "starter": 1, "pro": 2, "enterprise": 3}
CHANNEL_MAP = {"organic": 0, "paid_search": 1, "social": 2,
               "email": 3, "referral": 4, "affiliate": 5}
PLAN_MRR    = {"free": 0, "starter": 19, "pro": 49, "enterprise": 199}


def build_features(engine, lookback_days: int = 365) -> pd.DataFrame:
    """
    Build feature matrix for all active users.
    Returns one row per user with ML-ready features.
    """
    now    = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)

    # ── Load raw tables ──────────────────────────────────────────────────────
    users = pd.read_sql("""
        SELECT user_id, created_at, plan, channel, country,
               is_churned, churned_at, ltv_actual, age
        FROM users
        WHERE created_at >= %(cutoff)s
    """, engine, params={"cutoff": cutoff})

    events = pd.read_sql("""
        SELECT user_id, event_type, event_at,
               metadata->>'duration_minutes' AS duration_min
        FROM user_events
        WHERE event_at >= %(cutoff)s
    """, engine, params={"cutoff": cutoff})

    txns = pd.read_sql("""
        SELECT user_id, amount, status, tx_at
        FROM transactions
        WHERE tx_at >= %(cutoff)s
    """, engine, params={"cutoff": cutoff})

    if users.empty:
        logger.warning("No users found for feature engineering")
        return pd.DataFrame()

    logger.info(f"Building features for {len(users)} users")

    # ── User-level time features ──────────────────────────────────────────────
    users["created_at"]  = pd.to_datetime(users["created_at"], utc=True)
    users["days_since_signup"] = (now - users["created_at"]).dt.days.clip(lower=1)

    # ── Event features ────────────────────────────────────────────────────────
    events["event_at"] = pd.to_datetime(events["event_at"], utc=True)
    events["duration_min"] = pd.to_numeric(events["duration_min"], errors="coerce").fillna(0)

    logins = events[events["event_type"] == "login"]
    feat_use = events[events["event_type"] == "feature_use"]
    tickets  = events[events["event_type"] == "support_ticket"]

    w7  = now - timedelta(days=7)
    w30 = now - timedelta(days=30)

    def agg_events(df, col_name, window=None):
        if window:
            df = df[df["event_at"] >= window]
        return df.groupby("user_id").size().rename(col_name)

    session_7d  = agg_events(logins, "session_count_7d",  w7)
    session_30d = agg_events(logins, "session_count_30d", w30)
    avg_dur     = logins.groupby("user_id")["duration_min"].mean().rename("avg_session_duration")
    feat_score  = feat_use.groupby("user_id").size().rename("feature_usage_score")
    support_ct  = tickets.groupby("user_id").size().rename("support_tickets")

    last_login  = logins.groupby("user_id")["event_at"].max().rename("last_login_at")

    # ── Transaction features ──────────────────────────────────────────────────
    txns["tx_at"] = pd.to_datetime(txns["tx_at"], utc=True)
    total_rev   = txns[txns["status"] == "success"].groupby("user_id")["amount"].sum().rename("total_revenue")
    pay_fails   = txns[txns["status"] == "failed"].groupby("user_id").size().rename("payment_failures")

    # ── Merge all features ────────────────────────────────────────────────────
    feat = users[["user_id","plan","channel","country","age",
                  "days_since_signup","is_churned","ltv_actual"]].copy()

    for series in [session_7d, session_30d, avg_dur, feat_score,
                   support_ct, last_login, total_rev, pay_fails]:
        feat = feat.join(series, on="user_id", how="left")

    # ── Fill nulls ────────────────────────────────────────────────────────────
    feat["session_count_7d"]     = feat["session_count_7d"].fillna(0)
    feat["session_count_30d"]    = feat["session_count_30d"].fillna(0)
    feat["avg_session_duration"] = feat["avg_session_duration"].fillna(0)
    feat["feature_usage_score"]  = feat["feature_usage_score"].fillna(0)
    feat["support_tickets"]      = feat["support_tickets"].fillna(0)
    feat["total_revenue"]        = feat["total_revenue"].fillna(0)
    feat["payment_failures"]     = feat["payment_failures"].fillna(0)
    feat["age"]                  = feat["age"].fillna(32)

    feat["days_since_last_login"] = feat.apply(
        lambda r: (now - r["last_login_at"]).days
        if pd.notna(r.get("last_login_at")) else r["days_since_signup"],
        axis=1
    )

    # ── Encode categoricals ───────────────────────────────────────────────────
    feat["plan_encoded"]    = feat["plan"].map(PLAN_MAP).fillna(0).astype(int)
    feat["channel_encoded"] = feat["channel"].map(CHANNEL_MAP).fillna(0).astype(int)
    feat["country_encoded"] = feat["country"].astype("category").cat.codes

    # ── Expected MRR ─────────────────────────────────────────────────────────
    feat["expected_mrr"] = feat["plan"].map(PLAN_MRR).fillna(0)

    logger.success(f"Built {len(feat)} feature rows, {len(feat.columns)} columns")
    return feat


FEATURE_COLS = [
    "days_since_signup", "session_count_7d", "session_count_30d",
    "avg_session_duration", "feature_usage_score", "support_tickets",
    "plan_encoded", "channel_encoded", "country_encoded",
    "days_since_last_login", "total_revenue", "payment_failures", "age"
]


if __name__ == "__main__":
    cfg    = load_config()
    engine = get_engine(cfg)
    df     = build_features(engine)
    print(df[FEATURE_COLS + ["is_churned", "ltv_actual"]].describe())
    df.to_csv("outputs/features.csv", index=False)
    print(f"Saved {len(df)} rows to outputs/features.csv")
