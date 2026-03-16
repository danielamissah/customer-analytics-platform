"""
LTV Forecasting Model
=====================
XGBoost regressor predicting 12-month customer lifetime value.
Trained nightly, tracked in MLflow.
"""

import os
import pickle
import yaml
import mlflow
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from loguru import logger
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.data.features import load_config, get_engine, build_features, FEATURE_COLS

MODEL_DIR = Path("outputs/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LTV_FEATURES = FEATURE_COLS + ["expected_mrr"]


def train_ltv_model(df: pd.DataFrame, config: dict) -> dict:
    """Train XGBoost LTV model and log to MLflow."""

    min_samples = config["models"]["ltv"].get("retrain_min_samples", 200)
    df_model    = df[df["ltv_actual"] > 0].copy()

    if len(df_model) < min_samples:
        logger.warning(f"Insufficient LTV data: {len(df_model)} < {min_samples}")
        return {"status": "insufficient_data", "n_samples": len(df_model)}

    # Cap extreme LTV values (top 1%)
    cap   = df_model["ltv_actual"].quantile(0.99)
    df_model["ltv_actual"] = df_model["ltv_actual"].clip(upper=cap)

    X = df_model[[c for c in LTV_FEATURES if c in df_model.columns]].fillna(0)
    y = df_model["ltv_actual"]

    logger.info(f"Training LTV model on {len(X)} samples, "
                f"mean LTV=€{y.mean():.2f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment"])

    with mlflow.start_run(run_name="ltv_training"):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
            ))
        ])

        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_pred  = np.clip(y_pred, 0, None)

        metrics = {
            "mae":      round(mean_absolute_error(y_test, y_pred), 2),
            "rmse":     round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
            "r2":       round(r2_score(y_test, y_pred), 4),
            "mean_ltv_actual":    round(float(y_test.mean()), 2),
            "mean_ltv_predicted": round(float(y_pred.mean()), 2),
            "n_train":  len(X_train),
            "n_test":   len(X_test),
        }

        mlflow.log_params({
            "algorithm":     "XGBoost",
            "n_estimators":  200,
            "max_depth":     5,
            "learning_rate": 0.05,
            "features":      len(X.columns),
        })
        mlflow.log_metrics(metrics)

        # Save model locally
        ts         = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_path = MODEL_DIR / f"ltv_model_{ts}.pkl"
        latest     = MODEL_DIR / "ltv_model_latest.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        with open(latest, "wb") as f:
            pickle.dump(model, f)

        logger.success(
            f"LTV model trained — MAE=€{metrics['mae']:.2f}, "
            f"RMSE=€{metrics['rmse']:.2f}, R²={metrics['r2']:.4f}"
        )

        return {"status": "trained", "metrics": metrics,
                "model_path": str(latest), "run_id": mlflow.active_run().info.run_id}


def load_ltv_model():
    path = MODEL_DIR / "ltv_model_latest.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_ltv(model, df: pd.DataFrame) -> pd.DataFrame:
    """Score users with predicted 12-month LTV."""
    cols  = [c for c in LTV_FEATURES if c in df.columns]
    X     = df[cols].fillna(0)
    preds = np.clip(model.predict(X), 0, None)

    result = df[["user_id", "plan", "channel", "country"]].copy()
    result["predicted_ltv_eur"] = preds.round(2)
    result["ltv_segment"] = pd.cut(
        preds,
        bins=[0, 50, 150, 400, float("inf")],
        labels=["low", "mid", "high", "vip"]
    )
    return result.sort_values("predicted_ltv_eur", ascending=False)


if __name__ == "__main__":
    cfg    = load_config()
    engine = get_engine(cfg)
    df     = build_features(engine)
    result = train_ltv_model(df, cfg)
    print(result)