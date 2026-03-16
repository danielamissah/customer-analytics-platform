"""
Smoke tests for Customer Analytics Platform
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


def test_sample_size_calculation():
    from src.models.ab_testing import required_sample_size
    n = required_sample_size(baseline_rate=0.05, min_detectable_effect=0.20)
    assert n > 0
    assert isinstance(n, int)
    assert n > 100


def test_sample_size_larger_for_smaller_effect():
    from src.models.ab_testing import required_sample_size
    n_large = required_sample_size(baseline_rate=0.05, min_detectable_effect=0.10)
    n_small = required_sample_size(baseline_rate=0.05, min_detectable_effect=0.30)
    assert n_large > n_small


def test_config_loads():
    from src.data.features import load_config
    cfg = load_config()
    assert "database" in cfg
    assert "models" in cfg
    assert "ab_testing" in cfg


def test_feature_cols_defined():
    from src.data.features import FEATURE_COLS
    assert len(FEATURE_COLS) >= 10
    assert "days_since_signup" in FEATURE_COLS
    assert "churn_probability" not in FEATURE_COLS  # target not in features


def test_plan_mrr_mapping():
    from src.data.generator import PLAN_MRR
    assert PLAN_MRR["free"] == 0
    assert PLAN_MRR["enterprise"] > PLAN_MRR["pro"]
    assert PLAN_MRR["pro"] > PLAN_MRR["starter"]


def test_churn_multiplier_ordering():
    from src.data.generator import PLAN_CHURN_MULTIPLIER
    # Enterprise users should churn less than free users
    assert PLAN_CHURN_MULTIPLIER["enterprise"] < PLAN_CHURN_MULTIPLIER["free"]


def test_api_health(tmp_path):
    """Test API health endpoint without DB."""
    from fastapi.testclient import TestClient
    import sys
    sys.path.insert(0, ".")

    with patch("src.api.main.get_engine"), \
         patch("src.api.main.load_config", return_value={
             "database": {"url": "sqlite:///test.db"},
             "mlflow": {"tracking_uri": "sqlite:///mlflow_test.db",
                        "experiment": "test"},
             "ab_testing": {"significance_level": 0.05}
         }):
        from src.api.main import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
