.PHONY: help install up down logs migrate bootstrap run image up-full docker-clean lint fmt clean reset check-permissions

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies with uv (app, dev tools and UI)
	uv sync --all-groups

up: ## Start SpiceDB and Qdrant
	docker compose up -d spicedb qdrant

down: ## Stop all containers (volumes survive)
	docker compose --profile full down

logs: ## Tail container logs
	docker compose logs -f

migrate: ## Apply database migrations
	uv run alembic upgrade head

bootstrap: ## Apply SpiceDB schema and create the first admin
	uv run python scripts/bootstrap.py

run: ## Start the API with autoreload
	uv run uvicorn permrag.api.main:app --reload --host 0.0.0.0 --port 8000

image: ## Build the API and UI images (check free disk space first)
	docker compose --profile full build api ui

# --force-recreate: `up --build` can restart an existing stopped container on
# its old image; recreating guarantees the new build is what runs.
up-full: ## Start SpiceDB, Qdrant, the API (migrations first) and the UI on :8501
	docker compose up -d spicedb qdrant
	docker compose --profile full build api ui
	docker compose --profile full up -d --force-recreate --no-deps --wait api
	docker compose --profile full up -d --force-recreate --no-deps --wait ui

docker-clean: ## Free Docker disk: build cache and dangling images (never volumes)
	docker builder prune --all -f
	docker image prune -f

lint: ## Lint and type-check
	uv run ruff check src scripts
	uv run mypy src

fmt: ## Format
	uv run ruff format src scripts
	uv run ruff check --fix src scripts

check-permissions: ## Read-only: compare SpiceDB against the Postgres mirror
	uv run python scripts/check_permission_consistency.py

clean: ## Remove caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache

reset: ## DESTRUCTIVE: wipe Qdrant and both Postgres databases, then start over
	docker compose down -v
	uv run python scripts/reset_data.py --yes
	docker compose up -d spicedb qdrant
	@echo "Waiting for services..." && sleep 12
	$(MAKE) migrate
	$(MAKE) bootstrap
	