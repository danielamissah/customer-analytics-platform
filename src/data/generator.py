"""
Nightly Data Generator
======================
Generates realistic synthetic customer data and appends to PostgreSQL.
Runs every night via GitHub Actions scheduled workflow.

Simulates:
- New user signups (80 per night)
- User activity events (sessions, feature usage, support tickets)
- Transactions and subscription events
- Churn events based on behavioural signals
"""

import os
import uuid
import random
import yaml
import numpy as np
from datetime import datetime, timezone, timedelta
from loguru import logger
from faker import Faker
from sqlalchemy import create_engine, text

fake = Faker(["de_DE", "en_GB", "fr_FR"])


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_engine(config: dict):
    url = os.environ.get("DATABASE_URL", config["database"]["url"])
    return create_engine(url)


# ── Plan pricing ──────────────────────────────────────────────────────────────

PLAN_MRR = {"free": 0, "starter": 19, "pro": 49, "enterprise": 199}
PLAN_CHURN_MULTIPLIER = {"free": 2.5, "starter": 1.2, "pro": 0.8, "enterprise": 0.3}
PLAN_LTV_MULTIPLIER  = {"free": 0.2, "starter": 0.8, "pro": 2.0, "enterprise": 5.0}


def generate_user(config: dict, signup_date: datetime) -> dict:
    """Generate a single realistic user."""
    cfg    = config["data_generation"]
    plan   = random.choices(cfg["plans"], weights=cfg["plan_weights"])[0]
    channel = random.choices(cfg["channels"], weights=cfg["channel_weights"])[0]
    country = random.choice(cfg["countries"])

    return {
        "user_id":    str(uuid.uuid4()),
        "created_at": signup_date.isoformat(),
        "plan":       plan,
        "channel":    channel,
        "country":    country,
        "age":        random.randint(22, 55),
        "is_churned": False,
        "churned_at": None,
        "ltv_actual": 0.0,
    }


def generate_events(user_id: str, created_at: datetime,
                    plan: str, days_active: int) -> list:
    """Generate user activity events for the given period."""
    events = []
    base_sessions = {"free": 1.5, "starter": 3, "pro": 5, "enterprise": 8}
    base = base_sessions.get(plan, 2)

    for d in range(days_active):
        event_day = created_at + timedelta(days=d)
        n_sessions = max(0, int(np.random.poisson(base)))
        for _ in range(n_sessions):
            events.append({
                "user_id":    user_id,
                "event_type": "login",
                "event_at":   (event_day + timedelta(
                    hours=random.randint(7, 22),
                    minutes=random.randint(0, 59)
                )).isoformat(),
                "metadata":   {"duration_minutes": random.randint(2, 45)}
            })
        if random.random() < 0.4:
            events.append({
                "user_id":    user_id,
                "event_type": "feature_use",
                "event_at":   (event_day + timedelta(hours=random.randint(8, 20))).isoformat(),
                "metadata":   {"feature": random.choice(["export", "dashboard", "api", "report", "integration"])}
            })
        if random.random() < 0.03:
            events.append({
                "user_id":    user_id,
                "event_type": "support_ticket",
                "event_at":   (event_day + timedelta(hours=random.randint(9, 17))).isoformat(),
                "metadata":   {"priority": random.choice(["low", "medium", "high"])}
            })

    return events


def generate_transactions(user_id: str, plan: str,
                           created_at: datetime, days_active: int) -> list:
    """Generate monthly subscription transactions."""
    txns = []
    mrr  = PLAN_MRR[plan]
    if mrr == 0:
        return txns

    months = max(1, days_active // 30)
    for m in range(months):
        tx_date  = created_at + timedelta(days=m * 30)
        failed   = random.random() < 0.04
        txns.append({
            "user_id":  user_id,
            "amount":   mrr,
            "currency": "EUR",
            "status":   "failed" if failed else "success",
            "tx_at":    tx_date.isoformat(),
        })

    return txns


def should_churn(plan: str, days_active: int,
                 n_sessions: int, n_support_tickets: int,
                 config: dict) -> bool:
    """Probabilistic churn model based on behavioural signals."""
    base_rate  = config["data_generation"]["churn_rate_base"]
    multiplier = PLAN_CHURN_MULTIPLIER[plan]
    daily_rate = (base_rate * multiplier) / 30

    # Behavioural modifiers
    if n_sessions < 2:
        daily_rate *= 2.0   # low engagement → higher churn
    if n_support_tickets > 2:
        daily_rate *= 1.5   # frustrated users churn more
    if days_active > 180:
        daily_rate *= 0.6   # long-tenured users churn less

    return random.random() < (daily_rate * days_active)


def seed_historical_data(engine, config: dict, days_back: int = 365,
                          n_users: int = 3000):
    """
    Seed historical data on first run.
    Use n_users=500, days_back=90 for fast cloud seeding.
    """
    logger.info(f"Seeding {n_users} users over {days_back} days...")
    now = datetime.now(timezone.utc)

    all_users, all_events, all_txns = [], [], []

    for _ in range(n_users):
        signup_date  = now - timedelta(days=random.randint(1, days_back))
        user         = generate_user(config, signup_date)
        days_active  = (now - signup_date).days

        events = generate_events(
            user["user_id"], signup_date, user["plan"],
            min(days_active, 30)  # cap at 30 days of events to keep inserts manageable
        )
        txns = generate_transactions(
            user["user_id"], user["plan"], signup_date, days_active
        )

        n_sessions = len([e for e in events if e["event_type"] == "login"])
        n_tickets  = len([e for e in events if e["event_type"] == "support_ticket"])

        if should_churn(user["plan"], days_active, n_sessions, n_tickets, config):
            churn_day = signup_date + timedelta(
                days=random.randint(1, max(1, days_active))
            )
            user["is_churned"] = True
            user["churned_at"] = churn_day.isoformat()

        ltv = sum(t["amount"] for t in txns if t["status"] == "success")
        user["ltv_actual"] = round(float(ltv), 2)

        all_users.append(user)
        all_events.extend(events)
        all_txns.extend(txns)

        # Insert in batches of 200 users to avoid memory issues
        if len(all_users) % 200 == 0:
            _insert_batch(engine, config, all_users, all_events, all_txns)
            logger.info(f"Inserted {len(all_users)} users so far...")
            all_users, all_events, all_txns = [], [], []

    # Insert remaining
    if all_users:
        _insert_batch(engine, config, all_users, all_events, all_txns)

    logger.success("Seeding complete")


def generate_nightly(engine, config: dict):
    """Generate and insert tonight's new users + activity."""
    now         = datetime.now(timezone.utc)
    n_new_users = config["data_generation"]["daily_new_users"]
    logger.info(f"Generating {n_new_users} new users for {now.date()}...")

    all_users, all_events, all_txns = [], [], []

    for _ in range(n_new_users):
        # Signup time distributed across today
        signup_offset = timedelta(
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        signup_date = now.replace(hour=0, minute=0, second=0) + signup_offset
        user        = generate_user(config, signup_date)

        events = generate_events(user["user_id"], signup_date, user["plan"], 1)
        txns   = generate_transactions(user["user_id"], user["plan"], signup_date, 1)

        user["ltv_actual"] = round(
            float(sum(t["amount"] for t in txns if t["status"] == "success")), 2
        )

        all_users.append(user)
        all_events.extend(events)
        all_txns.extend(txns)

    # Also simulate some churn events in existing user base
    with engine.connect() as conn:
        active = conn.execute(text(
            "SELECT user_id, plan, created_at FROM users WHERE is_churned = FALSE "
            "ORDER BY RANDOM() LIMIT 50"
        )).fetchall()

    churned_ids = []
    for row in active:
        days_active = (now - row.created_at.replace(tzinfo=timezone.utc)).days
        if should_churn(row.plan, days_active, 3, 0, config):
            churned_ids.append(str(row.user_id))

    _insert_batch(engine, config, all_users, all_events, all_txns)

    if churned_ids:
        with engine.connect() as conn:
            conn.execute(text(
                "UPDATE users SET is_churned = TRUE, churned_at = NOW() "
                "WHERE user_id = ANY(:ids)"
            ), {"ids": churned_ids})
            conn.commit()
        logger.info(f"Marked {len(churned_ids)} users as churned")

    logger.success(f"Nightly generation complete: {len(all_users)} users, "
                   f"{len(all_events)} events, {len(all_txns)} transactions")
    return {"new_users": len(all_users), "new_events": len(all_events),
            "new_txns": len(all_txns), "new_churns": len(churned_ids)}


def _insert_batch(engine, config, users, events, txns, chunk_size=500):
    """Bulk insert using pandas to_sql — much faster over remote connections."""
    import json
    import pandas as pd

    with engine.connect() as conn:
        if users:
            pd.DataFrame(users).to_sql(
                "users", conn, if_exists="append", index=False,
                method="multi", chunksize=500
            )
            conn.commit()

        if events:
            df_e = pd.DataFrame(events)
            df_e["metadata"] = df_e["metadata"].apply(
                lambda x: json.dumps(x) if isinstance(x, dict) else str(x)
            )
            df_e.to_sql(
                "user_events", conn, if_exists="append", index=False,
                method="multi", chunksize=2000
            )
            conn.commit()

        if txns:
            pd.DataFrame(txns).to_sql(
                "transactions", conn, if_exists="append", index=False,
                method="multi", chunksize=500
            )
            conn.commit()


if __name__ == "__main__":
    import sys
    cfg    = load_config()
    engine = get_engine(cfg)

    n_users = 1000
    for arg in sys.argv:
        if arg.startswith("--users="):
            n_users = int(arg.split("=")[1])

    if "--seed" in sys.argv:
        days = 90 if "--fast" in sys.argv else 365
        seed_historical_data(engine, cfg, days_back=days, n_users=n_users)
    else:
        result = generate_nightly(engine, cfg)
        print(result)