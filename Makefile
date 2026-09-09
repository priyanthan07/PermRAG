.PHONY: help install up down logs migrate bootstrap run lint fmt clean reset

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies with uv
	uv sync --all-extras

up: ## Start SpiceDB and Qdrant
	docker compose up -d spicedb qdrant

down: ## Stop all containers (volumes survive)
	docker compose down

logs: ## Tail container logs
	docker compose logs -f

migrate: ## Apply database migrations
	uv run alembic upgrade head

bootstrap: ## Apply SpiceDB schema and create the first admin
	uv run python scripts/bootstrap.py

run: ## Start the API with autoreload
	uv run uvicorn permrag.api.main:app --reload --host 0.0.0.0 --port 8000

lint: ## Lint and type-check
	uv run ruff check src scripts
	uv run mypy src

fmt: ## Format
	uv run ruff format src scripts
	uv run ruff check --fix src scripts

clean: ## Remove caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache

reset: ## DESTRUCTIVE: wipe all data volumes and start over
	docker compose down -v
	docker compose up -d spicedb qdrant
	@echo "Waiting for services..." && sleep 12
	$(MAKE) migrate
	$(MAKE) bootstrap
	