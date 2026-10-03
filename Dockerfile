# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

# UV_NO_CACHE: uv's download cache would otherwise sit in the same layer as
# the installed packages, keeping a second copy of every wheel in the image.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

# uv is the package manager; copied from its official distroless image.
# Pinned: an unpinned "latest" makes two builds of the same commit differ.
COPY --from=ghcr.io/astral-sh/uv:0.8.4 /uv /uvx /bin/

WORKDIR /app

# Dependency layer first: this only rebuilds when the lockfile changes,
# not on every source edit. --locked fails the build if uv.lock is stale
# instead of silently re-resolving; --no-default-groups leaves the dev tools
# and the Streamlit UI out of the API image.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-default-groups

# pyproject.toml declares readme = "README.md"; building the project fails
# without it.
COPY README.md ./

COPY src/ ./src/
COPY alembic/ ./alembic/
COPY schema/ ./schema/
COPY scripts/ ./scripts/
COPY alembic.ini ./

RUN uv sync --locked --no-default-groups

# Run unprivileged. /app stays root-owned and read-only to the app user: the
# app never writes there (bytecode is compiled at build time), and a
# `chown -R /app` would copy the multi-GB .venv into one more image layer.
# The only runtime writes are the model cache under the user's home. That
# directory is created here, owned by the app user: docker-compose mounts a
# named volume on it, and a volume mounted on a path missing from the image
# comes up root-owned -- the app then cannot download its tokenizer and
# refuses to start.
RUN useradd --create-home --uid 10001 permrag \
    && mkdir -p /home/permrag/.cache/huggingface \
    && chown -R permrag:permrag /home/permrag/.cache
USER permrag

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=4).status==200 else 1)"

CMD ["uvicorn", "permrag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
