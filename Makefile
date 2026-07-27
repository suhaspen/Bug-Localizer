.DEFAULT_GOAL := help
SHELL := /bin/bash
UV := uv

.PHONY: help setup setup-ml test lint fmt db-up db-down db-logs mine stats samples \
        index index-stats retrieve eval eval-rerank eval-dev peeks clean

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

db-up:  ## Start Postgres + pgvector
	docker compose up -d
	@echo "postgres listening on localhost:5433"

db-down:  ## Stop Postgres
	docker compose down

db-logs:  ## Tail Postgres logs
	docker compose logs -f postgres

index:  ## Build the corpus index (add ARGS="--repo flask --limit 100")
	$(UV) run bugloc index $(ARGS)

index-stats:  ## Show what is in the corpus index
	$(UV) run bugloc index-stats

retrieve:  ## Rank files for one example, BM25 vs dense (add ARGS="-e pandas@abc123")
	$(UV) run bugloc retrieve $(ARGS)

mine:  ## Mine fix commits into data/examples.jsonl
	$(UV) run bugloc mine

stats:  ## Print dataset statistics, per repo
	$(UV) run bugloc dataset-stats

samples:  ## Print labeled examples for hand-review of label quality
	$(UV) run bugloc samples

eval:  ## Evaluate BM25 vs dense vs hybrid, both corpus scopes (ARGS="--limit 100")
	$(UV) run bugloc eval $(ARGS)

eval-rerank:  ## Same, plus cross-encoder reranking of the hybrid shortlist
	$(UV) run bugloc eval --rerank $(ARGS)

eval-dev:  ## Same, on the dev split — use this for tuning, not held-out
	$(UV) run bugloc eval --split dev $(ARGS)

peeks:  ## How many times the held-out set has been evaluated
	@wc -l < results/heldout_log.jsonl 2>/dev/null || echo 0

clean:  ## Remove caches and build artifacts (keeps .cache/ clones and results/)
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
