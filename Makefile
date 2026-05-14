.PHONY: help install test lint build up down logs shell clean

# ── Variables ──────────────────────────────────────────────────────────────────
IMAGE_NAME  ?= spam-phone-api
TAG         ?= latest
COMPOSE     := docker compose

# ── Help ───────────────────────────────────────────────────────────────────────
help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Development ────────────────────────────────────────────────────────────────
install:  ## Install Python dependencies
	pip install -r requirements.txt

test:  ## Run unit tests
	python -m unittest discover -s tests -p "test_*.py" -v

test-fast:  ## Run tests (stop on first failure)
	python -m unittest discover -s tests -p "test_*.py" -v --failfast

compile:  ## Check source compiles cleanly
	python -m compileall src scripts

smoke:  ## Run CLI smoke test
	python -m src.cli --help

cli:  ## Run a full pipeline command (usage: make cli CMD="infer --history-source data/...")
	python -m src.cli $(CMD)

# ── Docker ─────────────────────────────────────────────────────────────────────
build:  ## Build Docker image
	docker build -t $(IMAGE_NAME):$(TAG) .

up:  ## Start all services (API + MLflow)
	$(COMPOSE) up -d

down:  ## Stop all services
	$(COMPOSE) down

restart:  ## Restart API service only
	$(COMPOSE) restart api

logs:  ## Tail logs from all services
	$(COMPOSE) logs -f

logs-api:  ## Tail API logs only
	$(COMPOSE) logs -f api

logs-mlflow:  ## Tail MLflow logs only
	$(COMPOSE) logs -f mlflow

shell:  ## Open shell inside API container
	$(COMPOSE) exec api bash

health:  ## Check API health endpoint
	curl -s http://localhost:8000/health | python3 -m json.tool

# ── Cleanup ────────────────────────────────────────────────────────────────────
clean:  ## Remove __pycache__ and .pyc files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

clean-docker:  ## Remove all project Docker images and volumes
	$(COMPOSE) down -v --rmi local
