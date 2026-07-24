.DEFAULT_GOAL := help
SHELL := /bin/bash
UV := uv

.PHONY: help setup setup-ml test lint fmt db-up db-down db-logs mine stats index eval clean

help:  ## Show this help
	@echo "Bug Localizer — available commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Commands marked (Mn) are delivered by that milestone."

setup:  ## Create the venv and install core dependencies
	$(UV) sync

setup-ml:  ## Install heavy ML deps (sentence-transformers, psycopg) — needed from M2
	$(UV) sync --extra ml

test:  ## Run the test suite
	$(UV) run pytest

lint:  ## Check formatting and lint rules
	$(UV) run ruff check .
	$(UV) run ruff format --check .

fmt:  ## Auto-format the code
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

db-up:  ## Start Postgres + pgvector (M2)
	docker compose up -d
	@echo "postgres listening on localhost:5433"

db-down:  ## Stop Postgres
	docker compose down

db-logs:  ## Tail Postgres logs
	docker compose logs -f postgres

mine:  ## (M1) Mine fix commits into data/examples.jsonl
	$(UV) run bugloc mine

stats:  ## (M1) Print dataset statistics
	$(UV) run bugloc dataset-stats

index:  ## (M2) Build BM25 + pgvector indexes
	$(UV) run bugloc index

eval:  ## (M3) Run the evaluation and print the comparison table
	$(UV) run bugloc eval

clean:  ## Remove caches and build artifacts (keeps .cache/ clones and results/)
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
