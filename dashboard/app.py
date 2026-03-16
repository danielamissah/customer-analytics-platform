"""
Customer Analytics Platform — Dashboard
========================================
5-tab Streamlit dashboard:
  1. Overview       — KPIs, daily signups, revenue trends
  2. Churn          — risk scores, churn drivers, at-risk users
  3. LTV            — value distribution, segments, predicted vs actual
  4. A/B Testing    — create tests, analyse results, sample size calculator
  5. Model Health   — MLflow metrics, AUC-ROC, training history
"""

import io
import os
import json
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine, text

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
DB_URL  = os.environ.get("DATABASE_URL",
          "postgresql://analytics:analytics@localhost:5432/customer_analytics")
API_URL = os.environ.get("API_URL", "http://localhost:8000")

PLAN_COLORS = {
    "free": "#95A5A6", "starter": "#3498DB",
    "pro": "#2ECC71", "enterprise": "#9B59B6"
}
RISK_COLORS = {"low": "#2ECC71", "medium": "#F39C12", "high": "#E74C3C"}

@st.cache_resource
def get_engine():
    return create_engine(DB_URL)

engine = get_engine()

# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_users() -> pd.DataFrame:
    return pd.read_sql("""
        SELECT user_id, created_at, plan, channel, country,
               is_churned, churned_at, ltv_actual, age
        FROM users ORDER BY created_at
    """, engine)

@st.cache_data(ttl=300)
def load_daily_signups() -> pd.DataFrame:
    return pd.read_sql("""
        SELECT DATE(created_at) AS date, COUNT(*) AS signups,
               SUM(CASE WHEN is_churned THEN 1 ELSE 0 END) AS churns
        FROM users
        GROUP BY DATE(created_at)
        ORDER BY date
    """, engine)

@st.cache_data(ttl=300)
def load_revenue() -> pd.DataFrame:
    return pd.read_sql("""
        SELECT DATE(tx_at) AS date,
               SUM(CASE WHEN status='success' THEN amount ELSE 0 END) AS revenue,
               COUNT(*) AS transactions,
               SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed
        FROM transactions
        GROUP BY DATE(tx_at)
        ORDER BY date
    """, engine)

@st.cache_data(ttl=300)
def load_predictions() -> pd.DataFrame:
    return pd.read_sql("""
        SELECT p.user_id, p.prediction AS churn_probability,
               p.predicted_at, u.plan, u.channel, u.country,
               u.is_churned, u.ltv_actual
        FROM predictions p
        JOIN users u ON p.user_id = u.user_id
        WHERE p.model_name = 'churn_predictor'
        ORDER BY p.prediction DESC
    """, engine)

@st.cache_data(ttl=300)
def load_ab_tests() -> pd.DataFrame:
    return pd.read_sql("""
        SELECT t.test_id, t.test_name, t.hypothesis, t.metric,
               t.started_at, t.status, t.result, t.p_value, t.lift,
               COUNT(DISTINCT a.user_id) AS n_users,
               SUM(CASE WHEN a.variant='control' THEN 1 ELSE 0 END) AS n_control,
               SUM(CASE WHEN a.variant='treatment' THEN 1 ELSE 0 END) AS n_treatment,
               AVG(CASE WHEN a.converted THEN 1.0 ELSE 0.0 END) AS conversion_rate
        FROM ab_tests t
        LEFT JOIN ab_assignments a ON t.test_id = a.test_id
        GROUP BY t.test_id, t.test_name, t.hypothesis, t.metric,
                 t.started_at, t.status, t.result, t.p_value, t.lift
        ORDER BY t.started_at DESC
    """, engine)

@st.cache_data(ttl=300)
def load_mlflow_runs() -> pd.DataFrame:
    try:
        import mlflow
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI",
                                               "http://localhost:5002"))
        client = mlflow.tracking.MlflowClient()
        exp    = client.get_experiment_by_name("customer-analytics-platform")
        if exp is None:
            return pd.DataFrame()
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=20,
        )
        rows = []
        for r in runs:
            rows.append({
                "run_id":    r.info.run_id[:8],
                "run_name":  r.info.run_name,
                "status":    r.info.status,
                "started":   datetime.fromtimestamp(r.info.start_time / 1000),
                **{k: round(v, 4) for k, v in r.data.metrics.items()},
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def api_post(endpoint: str, payload: dict) -> dict:
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=30)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_get(endpoint: str) -> dict:
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Customer Analytics")
    st.caption("Live platform analytics · Refreshes every 5 min")
    st.divider()

    if st.button("Refresh Data", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

    if st.button("Trigger Retrain", use_container_width=True):
        result = api_post("/retrain", {})
        st.success("Retraining started")

    st.divider()

    stats = api_get("/stats")
    if "error" not in stats:
        st.metric("Total Users",    f"{stats['total_users']:,}")
        st.metric("Churn Rate",     f"{stats['churn_rate']:.1%}")
        st.metric("Total Revenue",  f"€{stats['total_revenue_eur']:,.0f}")
        st.metric("A/B Tests",      stats['ab_tests'])

    st.divider()
    st.caption("**Stack:** PostgreSQL · XGBoost · FastAPI · MLflow · Streamlit")
    st.caption("**GitHub:** [customer-analytics-platform](https://github.com/danielamissah/customer-analytics-platform)")


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Churn Analysis", "LTV Analysis",
    "A/B Testing", "Model Health"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Platform Overview")

    users   = load_users()
    revenue = load_revenue()
    daily   = load_daily_signups()

    # KPI cards
    total_users    = len(users)
    churned        = users["is_churned"].sum()
    churn_rate     = churned / total_users
    total_rev      = revenue["revenue"].sum()
    avg_ltv        = users[users["ltv_actual"] > 0]["ltv_actual"].mean()
    paying_users   = users[users["plan"] != "free"]
    mrr            = paying_users["plan"].map(
        {"starter": 19, "pro": 49, "enterprise": 199}
    ).sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Users",    f"{total_users:,}")
    c2.metric("Churn Rate",     f"{churn_rate:.1%}",
              delta=f"{churned:,} churned", delta_color="inverse")
    c3.metric("Total Revenue",  f"€{total_rev:,.0f}")
    c4.metric("Est. MRR",       f"€{mrr:,.0f}")
    c5.metric("Avg LTV",        f"€{avg_ltv:.0f}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Daily Signups vs Churns")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=daily["date"], y=daily["signups"],
            name="Signups", marker_color="#3498DB"))
        fig.add_trace(go.Bar(x=daily["date"], y=daily["churns"],
            name="Churns", marker_color="#E74C3C"))
        fig.update_layout(barmode="group", height=320,
            template="plotly_white", hovermode="x unified",
            xaxis_title="Date", yaxis_title="Users")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Cumulative Revenue")
        rev_cum = revenue.copy()
        rev_cum["cumulative"] = rev_cum["revenue"].cumsum()
        fig2 = px.area(rev_cum, x="date", y="cumulative",
            title="", template="plotly_white",
            labels={"cumulative": "Revenue (EUR)", "date": "Date"},
            color_discrete_sequence=["#2ECC71"])
        fig2.update_layout(height=320)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Users by Plan")
        plan_counts = users["plan"].value_counts().reset_index()
        plan_counts.columns = ["plan", "count"]
        fig3 = px.pie(plan_counts, values="count", names="plan",
            color="plan", color_discrete_map=PLAN_COLORS,
            hole=0.4, template="plotly_white")
        fig3.update_layout(height=300)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Users by Acquisition Channel")
        ch_counts = users["channel"].value_counts().reset_index()
        ch_counts.columns = ["channel", "count"]
        fig4 = px.bar(ch_counts, x="count", y="channel",
            orientation="h", template="plotly_white",
            color="count", color_continuous_scale="Blues",
            labels={"count": "Users", "channel": "Channel"})
        fig4.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHURN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Churn Analysis")
    users = load_users()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Churn Rate by Plan")
        churn_plan = users.groupby("plan").agg(
            total=("user_id", "count"),
            churned=("is_churned", "sum")
        ).reset_index()
        churn_plan["churn_rate"] = churn_plan["churned"] / churn_plan["total"]
        fig = px.bar(churn_plan, x="plan", y="churn_rate",
            color="plan", color_discrete_map=PLAN_COLORS,
            text=churn_plan["churn_rate"].apply(lambda x: f"{x:.1%}"),
            template="plotly_white",
            labels={"churn_rate": "Churn Rate", "plan": "Plan"})
        fig.update_traces(textposition="outside")
        fig.update_layout(height=320, showlegend=False,
                          yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Churn Rate by Channel")
        churn_ch = users.groupby("channel").agg(
            total=("user_id", "count"),
            churned=("is_churned", "sum")
        ).reset_index()
        churn_ch["churn_rate"] = churn_ch["churned"] / churn_ch["total"]
        churn_ch = churn_ch.sort_values("churn_rate", ascending=True)
        fig2 = px.bar(churn_ch, x="churn_rate", y="channel",
            orientation="h", template="plotly_white",
            color="churn_rate", color_continuous_scale="RdYlGn_r",
            labels={"churn_rate": "Churn Rate", "channel": "Channel"})
        fig2.update_layout(height=320, showlegend=False,
                           xaxis_tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        st.subheader("Churn Rate by Country")
        churn_co = users.groupby("country").agg(
            total=("user_id", "count"),
            churned=("is_churned", "sum")
        ).reset_index()
        churn_co["churn_rate"] = churn_co["churned"] / churn_co["total"]
        churn_co = churn_co.sort_values("churn_rate", ascending=False)
        fig3 = px.bar(churn_co, x="country", y="churn_rate",
            template="plotly_white",
            color="churn_rate", color_continuous_scale="RdYlGn_r",
            labels={"churn_rate": "Churn Rate", "country": "Country"})
        fig3.update_layout(height=320, showlegend=False,
                           yaxis_tickformat=".0%")
        st.plotly_chart(fig3, use_container_width=True)

    # Cohort retention
    st.subheader("Monthly Cohort Retention")
    users["cohort_month"] = pd.to_datetime(
        users["created_at"], utc=True).dt.to_period("M").astype(str)
    cohort = users.groupby("cohort_month").agg(
        total=("user_id","count"),
        retained=("is_churned", lambda x: (~x).sum())
    ).reset_index()
    cohort["retention_rate"] = cohort["retained"] / cohort["total"]
    fig_coh = px.bar(cohort.tail(12), x="cohort_month", y="retention_rate",
        template="plotly_white",
        labels={"retention_rate": "Retention Rate", "cohort_month": "Cohort"},
        color="retention_rate", color_continuous_scale="RdYlGn")
    fig_coh.update_layout(height=300, yaxis_tickformat=".0%")
    st.plotly_chart(fig_coh, use_container_width=True)

    # At-risk users from predictions
    preds = load_predictions()
    if not preds.empty:
        st.subheader("Top 20 At-Risk Users")
        st.caption("Users with highest predicted churn probability from the live model")
        at_risk = preds.head(20)[
            ["user_id", "churn_probability", "plan", "channel",
             "country", "ltv_actual"]
        ].copy()
        at_risk["churn_probability"] = at_risk["churn_probability"].apply(
            lambda x: f"{x:.1%}")
        at_risk["ltv_actual"] = at_risk["ltv_actual"].apply(
            lambda x: f"€{x:.0f}")
        st.dataframe(at_risk, use_container_width=True, hide_index=True)
    else:
        st.info("No churn predictions yet — run /predict/churn/batch via the API.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LTV ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Customer Lifetime Value Analysis")
    users = load_users()
    paying = users[users["ltv_actual"] > 0].copy()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("LTV Distribution")
        fig = px.histogram(paying, x="ltv_actual", nbins=40,
            color="plan", color_discrete_map=PLAN_COLORS,
            template="plotly_white", barmode="overlay",
            labels={"ltv_actual": "LTV (EUR)", "plan": "Plan"})
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Median LTV by Plan")
        ltv_plan = paying.groupby("plan")["ltv_actual"].agg(
            ["median", "mean", "std"]
        ).round(2).reset_index()
        ltv_plan.columns = ["plan", "median_ltv", "mean_ltv", "std_ltv"]
        fig2 = px.bar(ltv_plan, x="plan", y="median_ltv",
            color="plan", color_discrete_map=PLAN_COLORS,
            text=ltv_plan["median_ltv"].apply(lambda x: f"€{x:.0f}"),
            template="plotly_white",
            labels={"median_ltv": "Median LTV (EUR)", "plan": "Plan"})
        fig2.update_traces(textposition="outside")
        fig2.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("LTV by Acquisition Channel")
        ltv_ch = paying.groupby("channel")["ltv_actual"].median().reset_index()
        ltv_ch.columns = ["channel", "median_ltv"]
        ltv_ch = ltv_ch.sort_values("median_ltv", ascending=True)
        fig3 = px.bar(ltv_ch, x="median_ltv", y="channel",
            orientation="h", template="plotly_white",
            color="median_ltv", color_continuous_scale="Greens",
            labels={"median_ltv": "Median LTV (EUR)", "channel": "Channel"})
        fig3.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("LTV Segments")
        paying["ltv_segment"] = pd.cut(
            paying["ltv_actual"],
            bins=[0, 50, 150, 400, float("inf")],
            labels=["Low (<€50)", "Mid (€50-150)",
                    "High (€150-400)", "VIP (€400+)"]
        )
        seg_counts = paying["ltv_segment"].value_counts().reset_index()
        seg_counts.columns = ["segment", "count"]
        fig4 = px.pie(seg_counts, values="count", names="segment",
            color_discrete_sequence=["#95A5A6","#3498DB","#2ECC71","#9B59B6"],
            hole=0.4, template="plotly_white")
        fig4.update_layout(height=320)
        st.plotly_chart(fig4, use_container_width=True)

    # LTV summary table
    st.subheader("LTV Summary by Plan")
    summary = paying.groupby("plan")["ltv_actual"].agg(
        count="count", mean="mean", median="median",
        p25=lambda x: x.quantile(0.25),
        p75=lambda x: x.quantile(0.75),
        total="sum"
    ).round(2).reset_index()
    summary.columns = ["Plan", "Count", "Mean LTV",
                        "Median LTV", "P25", "P75", "Total LTV"]
    st.dataframe(summary, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — A/B TESTING
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("A/B Testing Framework")

    subtab1, subtab2, subtab3 = st.tabs([
        "Active Tests", "Create New Test", "Sample Size Calculator"
    ])

    with subtab1:
        tests = load_ab_tests()
        if tests.empty:
            st.info("No A/B tests created yet. Use the 'Create New Test' tab to start one.")
        else:
            for _, row in tests.iterrows():
                with st.expander(f"{row['test_name']}  —  {row['status'].upper()}",
                                 expanded=True):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Users", f"{int(row['n_users']):,}")
                    c2.metric("Control",     f"{int(row['n_control']):,}")
                    c3.metric("Treatment",   f"{int(row['n_treatment']):,}")
                    c4.metric("Conversion",  f"{row['conversion_rate']:.1%}"
                              if pd.notna(row['conversion_rate']) else "N/A")

                    if pd.notna(row.get("p_value")):
                        sig = row["p_value"] < 0.05
                        st.markdown(
                            f"**Result:** {'SIGNIFICANT' if sig else 'Not significant'}  "
                            f"| p-value: **{row['p_value']:.4f}**  "
                            f"| Lift: **{row['lift']:+.1%}**"
                            if pd.notna(row.get("lift")) else
                            f"**Result:** p-value: {row['p_value']:.4f}"
                        )
                        if sig:
                            st.success("Treatment wins — recommend rolling out.")
                        else:
                            st.warning("No significant difference detected yet.")

                    if row["n_users"] >= 10:
                        result = api_post("/ab/analyse",
                                          {"test_id": int(row["test_id"])})
                        if "error" not in result and "p_value" in result:
                            st.json(result)

    with subtab2:
        st.subheader("Create a New A/B Test")
        with st.form("create_test"):
            test_name  = st.text_input("Test name",
                placeholder="e.g. onboarding_email_v2")
            hypothesis = st.text_area("Hypothesis",
                placeholder="e.g. Sending a personalised onboarding email "
                            "will increase 7-day activation rate by 15%")
            metric     = st.selectbox("Primary metric",
                ["conversion", "revenue", "retention", "ltv", "activation"])
            n_assign   = st.slider("Users to assign", 100, 1000, 200, 50)
            submitted  = st.form_submit_button("Create Test & Assign Users",
                                               type="primary")

        if submitted and test_name:
            # Create test
            create_result = api_post("/ab/create", {
                "test_name": test_name,
                "hypothesis": hypothesis,
                "metric": metric,
            })
            if "error" not in create_result:
                test_id = create_result["test_id"]
                # Get random users to assign
                with engine.connect() as conn:
                    user_ids = [str(r[0]) for r in conn.execute(
                        text(f"SELECT user_id FROM users "
                             f"ORDER BY RANDOM() LIMIT {n_assign}")
                    ).fetchall()]
                assign_result = api_post("/ab/assign", {
                    "test_id": test_id,
                    "user_ids": user_ids,
                    "control_pct": 0.5,
                })
                st.success(
                    f"Test '{test_name}' created (id={test_id}). "
                    f"Assigned {assign_result.get('control',0)} control + "
                    f"{assign_result.get('treatment',0)} treatment users."
                )
                st.cache_data.clear()
            else:
                st.error(f"Error: {create_result['error']}")

    with subtab3:
        st.subheader("Sample Size Calculator")
        st.caption(
            "Calculate the minimum number of users needed per variant "
            "to detect a given effect size with statistical confidence."
        )
        c1, c2, c3, c4 = st.columns(4)
        baseline = c1.number_input("Baseline conversion rate",
            min_value=0.01, max_value=0.99, value=0.05, step=0.01,
            format="%.2f")
        mde = c2.number_input("Min detectable effect (relative)",
            min_value=0.01, max_value=1.0, value=0.20, step=0.01,
            format="%.2f")
        alpha = c3.selectbox("Significance level (α)", [0.05, 0.01, 0.10],
            index=0)
        power = c4.selectbox("Statistical power (1-β)", [0.80, 0.90, 0.95],
            index=0)

        result = api_post("/ab/sample-size", {
            "baseline_rate": baseline,
            "min_detectable_effect": mde,
            "alpha": alpha,
            "power": power,
        })

        if "error" not in result:
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Per variant",   f"{result['n_per_variant']:,}")
            rc2.metric("Total users",   f"{result['n_total']:,}")
            rc3.metric("Expected rate", f"{baseline*(1+mde):.1%}")
            st.caption(
                f"To detect a {mde:.0%} relative lift from {baseline:.0%} "
                f"baseline with α={alpha} and {power:.0%} power, "
                f"you need **{result['n_per_variant']:,} users per variant** "
                f"({result['n_total']:,} total)."
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — MODEL HEALTH
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Model Health & Training History")

    runs = load_mlflow_runs()

    if runs.empty:
        st.info("No MLflow runs found. Train models first with `make train`.")
    else:
        churn_runs = runs[runs["run_name"].str.contains("churn", case=False, na=False)]
        ltv_runs   = runs[runs["run_name"].str.contains("ltv",   case=False, na=False)]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Churn Model — Latest Metrics")
            if not churn_runs.empty:
                latest = churn_runs.iloc[0]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("AUC-ROC",   f"{latest.get('auc_roc', 'N/A')}")
                m2.metric("AUC-PR",    f"{latest.get('auc_pr', 'N/A')}")
                m3.metric("Recall",    f"{latest.get('recall', 'N/A')}")
                m4.metric("F1",        f"{latest.get('f1', 'N/A')}")

                if len(churn_runs) > 1:
                    fig = px.line(churn_runs.sort_values("started"),
                        x="started", y="auc_roc",
                        title="AUC-ROC Over Training Runs",
                        template="plotly_white",
                        markers=True,
                        labels={"auc_roc": "AUC-ROC", "started": "Run Date"})
                    fig.update_layout(height=280)
                    st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("LTV Model — Latest Metrics")
            if not ltv_runs.empty:
                latest_ltv = ltv_runs.iloc[0]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("R²",   f"{latest_ltv.get('r2', 'N/A')}")
                m2.metric("MAE",  f"€{latest_ltv.get('mae', 'N/A')}")
                m3.metric("RMSE", f"€{latest_ltv.get('rmse', 'N/A')}")
                m4.metric("Runs", len(ltv_runs))

                if len(ltv_runs) > 1:
                    fig2 = px.line(ltv_runs.sort_values("started"),
                        x="started", y="r2",
                        title="R² Over Training Runs",
                        template="plotly_white",
                        markers=True,
                        labels={"r2": "R²", "started": "Run Date"})
                    fig2.update_layout(height=280)
                    st.plotly_chart(fig2, use_container_width=True)

        st.subheader("All Training Runs")
        display_cols = [c for c in ["run_name", "started", "status",
                                     "auc_roc", "recall", "f1",
                                     "r2", "mae", "rmse"]
                        if c in runs.columns]
        st.dataframe(runs[display_cols], use_container_width=True,
                     hide_index=True)

    st.divider()
    st.subheader("Quick API Test")
    st.caption("Test the model API directly from the dashboard")

    with st.form("api_test"):
        c1, c2, c3 = st.columns(3)
        days_signup   = c1.number_input("Days since signup", 1, 500, 45)
        sessions_7d   = c2.number_input("Sessions (7d)", 0, 50, 2)
        plan_enc      = c3.selectbox("Plan", ["0=free","1=starter","2=pro","3=enterprise"])
        run_test      = st.form_submit_button("Predict Churn & LTV", type="primary")

    if run_test:
        payload = {
            "user_id": "demo-user",
            "days_since_signup": days_signup,
            "session_count_7d": sessions_7d,
            "session_count_30d": sessions_7d * 4,
            "avg_session_duration": 15,
            "feature_usage_score": sessions_7d * 0.4,
            "support_tickets": 0,
            "plan_encoded": int(plan_enc[0]),
            "channel_encoded": 0,
            "country_encoded": 0,
            "days_since_last_login": max(1, 7 - sessions_7d),
            "total_revenue": int(plan_enc[0]) * 19 * (days_signup // 30),
            "payment_failures": 0,
            "age": 32,
            "expected_mrr": [0, 19, 49, 199][int(plan_enc[0])],
        }
        churn_r = api_post("/predict/churn", payload)
        ltv_r   = api_post("/predict/ltv",   payload)

        rc1, rc2 = st.columns(2)
        if "churn_probability" in churn_r:
            rc1.metric("Churn Probability",
                       f"{churn_r['churn_probability']:.1%}",
                       delta=churn_r["churn_risk"])
        if "predicted_ltv_eur" in ltv_r:
            rc2.metric("Predicted LTV",
                       f"€{ltv_r['predicted_ltv_eur']:.2f}",
                       delta=ltv_r["ltv_segment"])