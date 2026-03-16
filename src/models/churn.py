"""
Churn Prediction Model
======================
XGBoost binary classifier predicting 30-day churn probability.
Trained nightly on fresh data, tracked in MLflow.
"""

import pickle
import mlflow
import mlflow.sklearn
import pandas as pd
from datetime import datetime, timezone
from loguru import logger
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                              f1_score, average_precision_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.features import load_config, get_engine, build_features, FEATURE_COLS


MODEL_DIR = Path("outputs/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_churn_model(df: pd.DataFrame, config: dict) -> dict:
    """Train XGBoost churn model and log to MLflow."""

    min_samples = config["models"]["churn"].get("retrain_min_samples", 200)
    df_model    = df[df["is_churned"].notna()].copy()

    if len(df_model) < min_samples:
        logger.warning(f"Insufficient data: {len(df_model)} < {min_samples}")
        return {"status": "insufficient_data", "n_samples": len(df_model)}

    X = df_model[FEATURE_COLS].fillna(0)
    y = df_model["is_churned"].astype(int)

    logger.info(f"Training churn model on {len(X)} samples, "
                f"churn rate={y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment"])

    with mlflow.start_run(run_name="churn_training"):
        # Model
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
            ))
        ])

        model.fit(X_train, y_train)

        # Evaluate
        y_prob  = model.predict_proba(X_test)[:, 1]
        threshold = config["models"]["churn"]["threshold"]
        y_pred  = (y_prob >= threshold).astype(int)

        metrics = {
            "auc_roc":          round(roc_auc_score(y_test, y_prob), 4),
            "auc_pr":           round(average_precision_score(y_test, y_prob), 4),
            "precision":        round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":           round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1":               round(f1_score(y_test, y_pred, zero_division=0), 4),
            "churn_rate_train": round(float(y_train.mean()), 4),
            "churn_rate_test":  round(float(y_test.mean()), 4),
            "n_train":          len(X_train),
            "n_test":           len(X_test),
        }

        mlflow.log_params({
            "algorithm":         "XGBoost",
            "n_estimators":      200,
            "max_depth":         6,
            "learning_rate":     0.05,
            "threshold":         threshold,
            "features":          len(FEATURE_COLS),
        })
        mlflow.log_metrics(metrics)

        # Save model locally
        ts         = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_path = MODEL_DIR / f"churn_model_{ts}.pkl"
        latest     = MODEL_DIR / "churn_model_latest.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        with open(latest, "wb") as f:
            pickle.dump(model, f)

        logger.success(
            f"Churn model trained — AUC-ROC={metrics['auc_roc']:.4f}, "
            f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}"
        )

        return {"status": "trained", "metrics": metrics,
                "model_path": str(latest), "run_id": mlflow.active_run().info.run_id}


def load_churn_model():
    """Load the latest churn model."""
    path = MODEL_DIR / "churn_model_latest.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_churn(model, df: pd.DataFrame) -> pd.DataFrame:
    """Score users with churn probability."""
    X         = df[FEATURE_COLS].fillna(0)
    probs     = model.predict_proba(X)[:, 1]
    result    = df[["user_id", "plan", "channel", "country"]].copy()
    result["churn_probability"] = probs.round(4)
    result["churn_risk"]        = pd.cut(
        probs,
        bins=[0, 0.2, 0.5, 1.0],
        labels=["low", "medium", "high"]
    )
    return result.sort_values("churn_probability", ascending=False)


if __name__ == "__main__":
    cfg    = load_config()
    engine = get_engine(cfg)
    df     = build_features(engine)
    result = train_churn_model(df, cfg)
    print(result)