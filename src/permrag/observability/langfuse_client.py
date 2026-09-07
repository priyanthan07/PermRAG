import logging
from langfuse import Langfuse
from permrag.config import get_settings

logger = logging.getLogger(__name__)

_client: Langfuse | None = None
_disabled: bool = False

def get_langfuse() -> Langfuse | None:

    global _client, _disabled

    if _disabled:
        return None
    if _client is not None:
        return _client

    settings = get_settings()
    if not settings.langfuse_configured:
        logger.info("langfuse disabled: credentials not configured")
        _disabled = True
        return None

    try:
        _client = Langfuse(
            public_key=settings.langfuse_public_key.get_secret_value(),  # type: ignore[union-attr]
            secret_key=settings.langfuse_secret_key.get_secret_value(),  # type: ignore[union-attr]
            host=settings.langfuse_host,
            environment=settings.environment,
            release=__import__("permrag").__version__,
        )
        logger.info("langfuse client initialised", extra={"host": settings.langfuse_host})
        return _client
    
    except Exception as exc:
        logger.warning("langfuse init failed, tracing disabled", extra={"error": str(exc)})
        _disabled = True
        return None
    
def get_trace_id() -> str | None:
    """Current trace id, for correlating a query_log row with a Langfuse trace."""
    client = get_langfuse()
    if client is None:
        return None
    try:
        return client.get_current_trace_id()
    except Exception:
        return None
    
def shutdown_langfuse() -> None:
    """Flush pending spans. Called from the FastAPI lifespan shutdown hook."""
    global _client
    if _client is not None:
        try:
            _client.flush()
            _client.shutdown()
        except Exception as exc:
            logger.warning("langfuse shutdown failed", extra={"error": str(exc)})
        _client = None
        