import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from permrag import __version__
from permrag.api.routers import admin, auth, chat, documents
from permrag.api.schemas import HealthResponse
from permrag.config import get_settings
from permrag.db.session import dispose_engine, get_engine
from permrag.exceptions import PermRAGError
from permrag.ingestion.chunker import get_tokenizer
from permrag.llm import check_provider_credentials
from permrag.logging_config import configure_logging
from permrag.observability.langfuse_client import get_langfuse, shutdown_langfuse
from permrag.permissions.client import get_spicedb_client
from permrag.rag.answerer import drain_background_tasks
from permrag.rag.reranker import warm_reranker
from permrag.vectorstore.qdrant import get_vector_store

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schema" / "permrag.zed"

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("starting permrag", extra={"version": __version__})

    check_provider_credentials()

    warm_reranker()
    logger.info("reranker warm")

    # Chunking counts tokens with the reranker's tokenizer whether or not the
    # reranker itself is enabled. Loading it here makes a missing model fail
    # the start rather than every upload, and keeps the load out of a request.
    get_tokenizer()

    get_engine()

    # SpiceDB: construct the client inside the loop so it selects the async
    # channel, then push the schema. Writing the schema is idempotent.
    spicedb = get_spicedb_client()
    if SCHEMA_PATH.exists():
        await spicedb.write_schema(SCHEMA_PATH.read_text(encoding="utf-8"))
    else:
        logger.warning("schema file not found", extra={"path": str(SCHEMA_PATH)})
    app.state.spicedb = spicedb

    vector_store = get_vector_store()
    await vector_store.ensure_collection()
    app.state.vector_store = vector_store

    get_langfuse()

    logger.info("startup complete")
    yield

    logger.info("shutting down")
    # Let in-flight evaluations push their scores before Langfuse is flushed.
    await drain_background_tasks()
    shutdown_langfuse()
    await vector_store.close()
    await dispose_engine()
    
def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="PermRAG",
        description="Permission-aware RAG chat system for a large organization",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url=None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.environment == "local" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(PermRAGError)
    async def permrag_error_handler(_: Request, exc: PermRAGError) -> JSONResponse:
        """Map the application exception hierarchy onto HTTP responses."""
        if exc.status_code >= 500:
            logger.error("request failed", extra={"error": exc.detail, "type": type(exc).__name__})
            # The instance detail can carry raw upstream errors (gRPC debug
            # strings, Qdrant responses). The caller gets the class's generic
            # message; the specific one is only in the log above.
            return JSONResponse(status_code=exc.status_code, content={"detail": type(exc).detail})
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
        """Never leak an internal traceback to the caller."""
        logger.exception("unhandled exception", extra={"type": type(exc).__name__})
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        """Liveness: the process is up. Says nothing about its dependencies."""
        return HealthResponse(status="ok", environment=settings.environment, version=__version__)

    @app.get("/ready", tags=["system"])
    async def ready(request: Request) -> JSONResponse:
        """Readiness: can this instance answer a question right now? 503 if not."""

        async def postgres() -> None:
            async with get_engine().connect() as conn:
                await conn.execute(text("SELECT 1"))

        probes = {
            "postgres": postgres(),
            # Any check will do: it must reach SpiceDB and get an answer.
            "spicedb": request.app.state.spicedb.check_permission("readiness-probe", "readiness-probe"),
            "qdrant": request.app.state.vector_store.ping(),
        }
        checks: dict[str, str] = {}
        for name, probe in probes.items():
            try:
                await probe
                checks[name] = "ok"
            except Exception as exc:
                logger.warning("readiness check failed", extra={"dependency": name, "error": str(exc)})
                checks[name] = "unavailable"

        ready_ = all(state == "ok" for state in checks.values())
        return JSONResponse(
            status_code=200 if ready_ else 503,
            content={"status": "ok" if ready_ else "unavailable", "checks": checks},
        )

    app.include_router(auth.router)
    app.include_router(admin.router)
    app.include_router(documents.router)
    app.include_router(chat.router)

    return app


app = create_app()
