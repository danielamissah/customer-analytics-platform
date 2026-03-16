PYTHON = /opt/anaconda3/bin/python3
PYTHONPATH = PYTHONPATH=.

.PHONY: setup db seed generate train retrain api dashboard test lint up down clean

setup:
	pip install -r requirements.txt

db:
	docker compose up -d postgres mlflow
	@sleep 5
	docker exec -i customer-analytics-platform-postgres-1 \
	  psql -U analytics -d customer_analytics \
	  < scripts/init_db.sql || true

seed:
	$(PYTHONPATH) $(PYTHON) src/data/generator.py --seed

generate:
	$(PYTHONPATH) $(PYTHON) src/data/generator.py

features:
	$(PYTHONPATH) $(PYTHON) src/data/features.py

train:
	$(PYTHONPATH) $(PYTHON) src/models/churn.py
	$(PYTHONPATH) $(PYTHON) src/models/ltv.py

retrain: generate features train

api:
	$(PYTHONPATH) uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	$(PYTHONPATH) streamlit run dashboard/app.py

test:
	$(PYTHONPATH) pytest tests/ -v

lint:
	ruff check src/ --ignore E501

up:
	docker compose up -d
	@echo "Services running:"
	@echo "  PostgreSQL : localhost:5432"
	@echo "  MLflow     : http://localhost:5001"
	@echo "  API        : http://localhost:8000"
	@echo "  Dashboard  : http://localhost:8501"

down:
	docker compose down

k8s-deploy:
	kubectl apply -f k8s/deployment.yaml
	kubectl rollout status deployment/analytics-api -n customer-analytics

clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	rm -f outputs/models/*.pkl outputs/*.csv