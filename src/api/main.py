"""
FastAPI Model Serving
=====================
REST API exposing churn prediction, LTV forecasting,
and A/B test management endpoints.
"""

import os
import yaml
import pandas as pd
from datetime import datetime, timezone
from typing import Optional
from loguru import logger
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data.features import load_config, get_engine, build_features, FEATURE_COLS
from src.models.churn import load_churn_model, predict_churn, train_churn_model
from src.models.ltv import load_ltv_model, predict_ltv, train_ltv_model, LTV_FEATURES
from src.models.ab_testing import (
    create_test, assign_users, record_conversion, analyse_test, required_sample_size
)

app = FastAPI(
    title="Customer Analytics Platform",
    description="Churn prediction, LTV forecasting, and A/B testing API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

config = load_config()
engine = get_engine(config)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/")
def root():
    return {
        "name":    "Customer Analytics Platform",
        "version": "1.0.0",
        "endpoints": [
            "/health", "/predict/churn", "/predict/ltv",
            "/ab/create", "/ab/assign", "/ab/analyse", "/ab/sample-size",
            "/retrain", "/stats"
        ]
    }


# ── Request / Response models ─────────────────────────────────────────────────

class UserFeatures(BaseModel):
    user_id:                str
    days_since_signup:      float
    session_count_7d:       float = 0
    session_count_30d:      float = 0
    avg_session_duration:   float = 0
    feature_usage_score:    float = 0
    support_tickets:        float = 0
    plan_encoded:           int   = 0
    channel_encoded:        int   = 0
    country_encoded:        int   = 0
    days_since_last_login:  float = 0
    total_revenue:          float = 0
    payment_failures:       float = 0
    age:                    float = 32
    expected_mrr:           float = 0


class ChurnResponse(BaseModel):
    user_id:           str
    churn_probability: float
    churn_risk:        str
    model_version:     str = "latest"


class LTVResponse(BaseModel):
    user_id:           str
    predicted_ltv_eur: float
    ltv_segment:       str
    model_version:     str = "latest"


class ABTestCreate(BaseModel):
    test_name:  str
    hypothesis: str
    metric:     str


class ABTestAssign(BaseModel):
    test_id:     int
    user_ids:    list[str]
    control_pct: float = 0.5


class ABTestAnalyse(BaseModel):
    test_id: int


class SampleSizeRequest(BaseModel):
    baseline_rate:           float = Field(..., gt=0, lt=1)
    min_detectable_effect:   float = Field(..., gt=0)
    alpha:                   float = 0.05
    power:                   float = 0.80


# ── Churn prediction ──────────────────────────────────────────────────────────

@app.post("/predict/churn", response_model=ChurnResponse)
def predict_churn_endpoint(user: UserFeatures):
    model = load_churn_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Churn model not trained yet")

    df   = pd.DataFrame([user.model_dump()])
    result = predict_churn(model, df)
    row  = result.iloc[0]

    return ChurnResponse(
        user_id=user.user_id,
        churn_probability=float(row["churn_probability"]),
        churn_risk=str(row["churn_risk"]),
    )


@app.post("/predict/churn/batch")
def predict_churn_batch(users: list[UserFeatures]):
    model = load_churn_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Churn model not trained yet")
    df     = pd.DataFrame([u.model_dump() for u in users])
    result = predict_churn(model, df)
    return result.to_dict(orient="records")


# ── LTV prediction ────────────────────────────────────────────────────────────

@app.post("/predict/ltv", response_model=LTVResponse)
def predict_ltv_endpoint(user: UserFeatures):
    model = load_ltv_model()
    if model is None:
        raise HTTPException(status_code=503, detail="LTV model not trained yet")

    df     = pd.DataFrame([user.model_dump()])
    result = predict_ltv(model, df)
    row    = result.iloc[0]

    return LTVResponse(
        user_id=user.user_id,
        predicted_ltv_eur=float(row["predicted_ltv_eur"]),
        ltv_segment=str(row["ltv_segment"]),
    )


# ── A/B testing ───────────────────────────────────────────────────────────────

@app.post("/ab/create")
def create_ab_test(payload: ABTestCreate):
    test_id = create_test(engine, payload.test_name,
                          payload.hypothesis, payload.metric)
    return {"test_id": test_id, "test_name": payload.test_name}


@app.post("/ab/assign")
def assign_ab_users(payload: ABTestAssign):
    result = assign_users(engine, payload.test_id,
                          payload.user_ids, payload.control_pct)
    return {"test_id": payload.test_id, **result}


@app.post("/ab/analyse")
def analyse_ab_test(payload: ABTestAnalyse):
    result = analyse_test(engine, payload.test_id)
    return result


@app.post("/ab/sample-size")
def sample_size(payload: SampleSizeRequest):
    n = required_sample_size(
        payload.baseline_rate,
        payload.min_detectable_effect,
        payload.alpha,
        payload.power,
    )
    return {
        "n_per_variant": n,
        "n_total":       n * 2,
        "baseline_rate": payload.baseline_rate,
        "mde":           payload.min_detectable_effect,
        "alpha":         payload.alpha,
        "power":         payload.power,
    }


# ── Retraining ────────────────────────────────────────────────────────────────

def _retrain_background():
    df = build_features(engine)
    train_churn_model(df, config)
    train_ltv_model(df, config)
    logger.info("Background retraining complete")


@app.post("/retrain")
def trigger_retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(_retrain_background)
    return {"status": "retraining_started",
            "timestamp": datetime.now(timezone.utc).isoformat()}


# ── Stats ─────────────────────────────────────────────────────────────────────

@app.get("/stats")
def platform_stats():
    with engine.connect() as conn:
        from sqlalchemy import text
        total_users  = conn.execute(text("SELECT COUNT(*) FROM users")).scalar()
        churned      = conn.execute(text("SELECT COUNT(*) FROM users WHERE is_churned")).scalar()
        total_events = conn.execute(text("SELECT COUNT(*) FROM user_events")).scalar()
        total_txns   = conn.execute(text("SELECT COUNT(*) FROM transactions")).scalar()
        total_revenue = conn.execute(
            text("SELECT COALESCE(SUM(amount),0) FROM transactions WHERE status='success'")
        ).scalar()
        ab_tests = conn.execute(text("SELECT COUNT(*) FROM ab_tests")).scalar()

    return {
        "total_users":    total_users,
        "churned_users":  churned,
        "churn_rate":     round(churned / max(total_users, 1), 4),
        "total_events":   total_events,
        "total_txns":     total_txns,
        "total_revenue_eur": float(total_revenue),
        "ab_tests":       ab_tests,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)