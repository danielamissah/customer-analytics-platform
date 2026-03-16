"""
A/B Testing Framework
=====================
Statistical hypothesis testing for product experiments.
Supports conversion rate tests (chi-square) and
continuous metric tests (t-test / Mann-Whitney U).
"""

import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from loguru import logger
from scipy import stats
from sqlalchemy import create_engine, text
from typing import Optional

from src.data.features import load_config, get_engine


def create_test(engine, test_name: str, hypothesis: str, metric: str) -> int:
    """Register a new A/B test in the database."""
    with engine.connect() as conn:
        result = conn.execute(text("""
            INSERT INTO ab_tests (test_name, hypothesis, metric, started_at, status)
            VALUES (:name, :hypothesis, :metric, NOW(), 'running')
            ON CONFLICT (test_name) DO NOTHING
            RETURNING test_id
        """), {"name": test_name, "hypothesis": hypothesis, "metric": metric})
        conn.commit()
        row = result.fetchone()
        if row:
            logger.info(f"Created A/B test '{test_name}' (id={row[0]})")
            return row[0]
        # Already exists — fetch id
        result = conn.execute(
            text("SELECT test_id FROM ab_tests WHERE test_name = :name"),
            {"name": test_name}
        )
        return result.fetchone()[0]


def assign_users(engine, test_id: int, user_ids: list,
                 control_pct: float = 0.5) -> dict:
    """Randomly assign users to control/treatment variants."""
    assignments = []
    for uid in user_ids:
        variant = "control" if np.random.random() < control_pct else "treatment"
        assignments.append({
            "test_id":     test_id,
            "user_id":     uid,
            "variant":     variant,
            "assigned_at": datetime.now(timezone.utc).isoformat(),
            "converted":   False,
        })

    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO ab_assignments
              (test_id, user_id, variant, assigned_at, converted)
            VALUES
              (:test_id, :user_id, :variant, :assigned_at, :converted)
            ON CONFLICT (test_id, user_id) DO NOTHING
        """), assignments)
        conn.commit()

    n_control   = sum(1 for a in assignments if a["variant"] == "control")
    n_treatment = len(assignments) - n_control
    logger.info(f"Assigned {len(assignments)} users: "
                f"{n_control} control, {n_treatment} treatment")
    return {"control": n_control, "treatment": n_treatment}


def record_conversion(engine, test_id: int, user_id: str):
    """Mark a user as converted in an A/B test."""
    with engine.connect() as conn:
        conn.execute(text("""
            UPDATE ab_assignments
            SET converted = TRUE, converted_at = NOW()
            WHERE test_id = :test_id AND user_id = :user_id
        """), {"test_id": test_id, "user_id": user_id})
        conn.commit()


def analyse_test(engine, test_id: int,
                 metric_values: Optional[pd.DataFrame] = None) -> dict:
    """
    Analyse an A/B test using the appropriate statistical test.

    For binary metrics (conversion rate): chi-square test
    For continuous metrics (revenue, LTV): Mann-Whitney U test

    Returns p-value, lift, sample sizes, and significance verdict.
    """
    config = load_config()
    alpha  = config["ab_testing"]["significance_level"]

    # Load assignments
    assignments = pd.read_sql("""
        SELECT a.user_id, a.variant, a.converted, a.converted_at
        FROM ab_assignments a
        WHERE a.test_id = %(test_id)s
    """, engine, params={"test_id": test_id})

    test_info = pd.read_sql("""
        SELECT test_name, hypothesis, metric, started_at
        FROM ab_tests WHERE test_id = %(test_id)s
    """, engine, params={"test_id": test_id}).iloc[0]

    control   = assignments[assignments["variant"] == "control"]
    treatment = assignments[assignments["variant"] == "treatment"]

    n_control   = len(control)
    n_treatment = len(treatment)

    if n_control < 10 or n_treatment < 10:
        return {
            "status":    "insufficient_data",
            "test_name": test_info["test_name"],
            "n_control": n_control, "n_treatment": n_treatment,
            "message":   "Need at least 10 users per variant"
        }

    # ── Binary metric (conversion rate) ──────────────────────────────────────
    if metric_values is None:
        conv_control   = control["converted"].sum()
        conv_treatment = treatment["converted"].sum()

        rate_control   = conv_control / n_control
        rate_treatment = conv_treatment / n_treatment
        lift           = (rate_treatment - rate_control) / max(rate_control, 1e-9)

        # Chi-square test
        contingency = np.array([
            [conv_control,   n_control   - conv_control],
            [conv_treatment, n_treatment - conv_treatment]
        ])
        _, p_value, _, _ = stats.chi2_contingency(contingency)

        result = {
            "test_name":       test_info["test_name"],
            "metric":          test_info["metric"],
            "n_control":       n_control,
            "n_treatment":     n_treatment,
            "rate_control":    round(rate_control, 4),
            "rate_treatment":  round(rate_treatment, 4),
            "lift":            round(lift, 4),
            "p_value":         round(p_value, 4),
            "significant":     p_value < alpha,
            "test_type":       "chi_square",
            "alpha":           alpha,
            "verdict":         "significant" if p_value < alpha else "not_significant",
            "recommendation":  (
                f"Treatment shows {lift:+.1%} lift in {test_info['metric']}. "
                f"{'Roll out treatment.' if p_value < alpha else 'Continue testing or accept null hypothesis.'}"
            )
        }

    # ── Continuous metric (revenue, LTV, etc.) ────────────────────────────────
    else:
        merged_c = control.merge(metric_values, on="user_id", how="left")
        merged_t = treatment.merge(metric_values, on="user_id", how="left")
        val_col  = [c for c in metric_values.columns if c != "user_id"][0]

        vals_c = merged_c[val_col].dropna()
        vals_t = merged_t[val_col].dropna()

        mean_c = vals_c.mean()
        mean_t = vals_t.mean()
        lift   = (mean_t - mean_c) / max(mean_c, 1e-9)

        # Mann-Whitney U (non-parametric, better for skewed revenue data)
        stat, p_value = stats.mannwhitneyu(vals_t, vals_c, alternative="two-sided")

        result = {
            "test_name":    test_info["test_name"],
            "metric":       val_col,
            "n_control":    len(vals_c),
            "n_treatment":  len(vals_t),
            "mean_control": round(mean_c, 2),
            "mean_treatment": round(mean_t, 2),
            "lift":         round(lift, 4),
            "p_value":      round(p_value, 4),
            "significant":  p_value < alpha,
            "test_type":    "mann_whitney_u",
            "alpha":        alpha,
            "verdict":      "significant" if p_value < alpha else "not_significant",
        }

    # Persist result
    with engine.connect() as conn:
        conn.execute(text("""
            UPDATE ab_tests
            SET result = :verdict, p_value = :p_value, lift = :lift
            WHERE test_id = :test_id
        """), {"verdict": result["verdict"], "p_value": result["p_value"],
               "lift": result["lift"], "test_id": test_id})
        conn.commit()

    logger.info(
        f"A/B test '{result['test_name']}': "
        f"p={result['p_value']:.4f}, lift={result['lift']:+.1%}, "
        f"{'SIGNIFICANT ✓' if result['significant'] else 'not significant'}"
    )
    return result


def required_sample_size(baseline_rate: float, min_detectable_effect: float,
                          alpha: float = 0.05, power: float = 0.80) -> int:
    """
    Calculate minimum sample size per variant using power analysis.
    Based on the two-proportion z-test formula.
    """
    p1 = baseline_rate
    p2 = baseline_rate * (1 + min_detectable_effect)
    p_bar = (p1 + p2) / 2

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta  = stats.norm.ppf(power)

    n = ((z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) +
          z_beta  * np.sqrt(p1 * (1-p1) + p2 * (1-p2))) ** 2) / ((p2 - p1) ** 2)

    return int(np.ceil(n))


if __name__ == "__main__":
    cfg    = load_config()
    engine = get_engine(cfg)

    # Example: calculate required sample size
    n = required_sample_size(
        baseline_rate=0.05,          # 5% baseline conversion
        min_detectable_effect=0.20,  # detect 20% relative lift
        alpha=0.05,
        power=0.80
    )
    print(f"Required sample size per variant: {n:,}")
