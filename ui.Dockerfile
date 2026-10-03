# syntax=docker/dockerfile:1
# The Streamlit console. It only talks to the API over HTTP, so it installs
# just the `ui` dependency group (Streamlit, httpx, file parsers) -- no torch,
# no API code.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

COPY --from=ghcr.io/astral-sh/uv:0.8.4 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --only-group ui

COPY ui/ ./ui/

RUN useradd --create-home --uid 10001 permrag
USER permrag

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health',timeout=4).status==200 else 1)"

CMD ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
