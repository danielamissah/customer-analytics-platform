# Customer Analytics Platform

Production-grade ML platform for customer churn prediction, lifetime value forecasting, and A/B testing — with nightly data refresh, FastAPI model serving, Kubernetes deployment, and GitHub Actions CI/CD.

**Live Demo → [custom-analytics-platform.streamlit.app](https://custom-analytics-platform.streamlit.app/)**

---

## Key Results

| Model                     | Metric          | Value               |
| ------------------------- | --------------- | ------------------- |
| Churn Predictor (XGBoost) | AUC-ROC         | **0.8792**    |
| Churn Predictor (XGBoost) | Recall          | **0.8477**    |
| Churn Predictor (XGBoost) | F1              | **0.806**     |
| LTV Forecaster (XGBoost)  | R²             | **0.9995**    |
| LTV Forecaster (XGBoost)  | MAE             | **€4.05**    |
| Platform                  | Users           | **1,500**     |
| Platform                  | Revenue         | **€491,785** |
| Nightly refresh           | New users/night | **80**        |

---

## Architecture

```
Nightly GitHub Actions (01:00 UTC)
    │
    ▼
Data Generator (Faker + statistical distributions)
    │  80 new users/night + churn events
    ▼
PostgreSQL on Supabase (7 tables)
    │
    ├── Feature Engineering (13 features)
    │       │
    │       ├── XGBoost Churn Model ──► MLflow
    │       └── XGBoost LTV Model   ──► MLflow
    │
    ▼
FastAPI REST API (9 endpoints)
    │
    ├── /predict/churn
    ├── /predict/ltv
    ├── /ab/create + /ab/assign + /ab/analyse
    ├── /ab/sample-size
    ├── /retrain
    └── /stats
    │
    ▼
Streamlit Dashboard (5 tabs)
    │
Docker → Kubernetes manifests (2 replicas, HPA 2–6 pods)
    │
GitHub Actions CI/CD (lint → test → Docker build → push to GHCR)
```

---

## Dashboard

| Tab                      | Contents                                                                          |
| ------------------------ | --------------------------------------------------------------------------------- |
| **Overview**       | KPI cards, daily signups vs churns, cumulative revenue, users by plan and channel |
| **Churn Analysis** | Churn by plan/channel/country, monthly cohort retention, top 20 at-risk users     |
| **LTV Analysis**   | LTV distribution, median by plan, by channel, LTV segments (Low/Mid/High/VIP)     |
| **A/B Testing**    | Create and analyse tests, chi-square + Mann-Whitney U, sample size calculator     |
| **Model Health**   | MLflow run history, AUC-ROC trend, live API test with custom user inputs          |

---

## Stack

* **ML:** XGBoost, scikit-learn, SciPy
* **Data:** PostgreSQL (Supabase), SQLAlchemy, pandas, Faker
* **API:** FastAPI, Pydantic v2, Uvicorn
* **Tracking:** MLflow
* **Dashboard:** Streamlit, Plotly
* **Infrastructure:** Docker, Kubernetes (deployment, HPA, ingress), GitHub Actions CI/CD
* **Language:** Python 3.11

---

## Running Locally

```bash
# 1. Clone and install
git clone https://github.com/danielamissah/customer-analytics-platform.git
cd customer-analytics-platform
pip install -r requirements.txt

# 2. Start PostgreSQL + MLflow
make db

# 3. Seed historical data
make seed

# 4. Train models
make train

# 5. Start API
make api         # http://localhost:8000 (Swagger: /docs)

# 6. Start dashboard
make dashboard   # http://localhost:8501
```

**Ports:** PostgreSQL `:5432` · MLflow `:5002` · API `:8000` · Dashboard `:8501`

---

## Nightly Refresh

```yaml
# .github/workflows/nightly.yml — runs at 01:00 UTC every night
on:
  schedule:
    - cron: "0 1 * * *"
```

Each night: generate 80 new users → retrain churn model → retrain LTV model → reload API.

---

## API Endpoints

| Method | Endpoint                 | Description                         |
| ------ | ------------------------ | ----------------------------------- |
| GET    | `/health`              | Health check                        |
| GET    | `/stats`               | Platform statistics                 |
| POST   | `/predict/churn`       | Churn probability for a single user |
| POST   | `/predict/churn/batch` | Batch churn scoring                 |
| POST   | `/predict/ltv`         | Predicted 12-month LTV              |
| POST   | `/ab/create`           | Register a new A/B test             |
| POST   | `/ab/assign`           | Assign users to variants            |
| POST   | `/ab/analyse`          | Run statistical analysis            |
| POST   | `/ab/sample-size`      | Power analysis — min sample size   |
| POST   | `/retrain`             | Trigger background model retraining |

---

## Project Structure

```
customer-analytics-platform/
├── src/
│   ├── data/
│   │   ├── generator.py       # Nightly synthetic data generation
│   │   └── features.py        # Feature engineering pipeline
│   ├── models/
│   │   ├── churn.py           # XGBoost churn predictor
│   │   ├── ltv.py             # XGBoost LTV forecaster
│   │   └── ab_testing.py      # A/B testing framework
│   └── api/
│       └── main.py            # FastAPI application
├── dashboard/
│   └── app.py                 # Streamlit dashboard (5 tabs)
├── k8s/
│   └── deployment.yaml        # Kubernetes manifests
├── .github/workflows/
│   ├── ci_cd.yml              # CI/CD — lint, test, Docker build, push to GHCR
│   └── nightly.yml            # Nightly data refresh + model retraining
├── scripts/
│   └── init_db.sql            # Database schema (7 tables)
├── configs/
│   └── config.yaml            # Platform configuration
├── tests/
│   └── test_smoke.py          # Smoke tests (pytest)
├── Dockerfile                 # API container
├── Dockerfile.dashboard       # Dashboard container
├── docker-compose.yml         # Local development stack
├── Makefile                   # Developer commands
└── requirements.txt
```
