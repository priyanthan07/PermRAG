# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# uv is the package manager; copied from its official distroless image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Dependency layer first: this only rebuilds when the lockfile changes,
# not on every source edit.
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-install-project --no-dev

COPY src/ ./src/
COPY alembic/ ./alembic/
COPY schema/ ./schema/
COPY scripts/ ./scripts/
COPY alembic.ini ./

RUN uv sync --no-dev

# Run unprivileged.
RUN useradd --create-home --uid 10001 permrag \
    && chown -R permrag:permrag /app
USER permrag

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=4).status==200 else 1)"

CMD ["uvicorn", "permrag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
