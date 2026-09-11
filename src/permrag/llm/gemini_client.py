import logging
import math
from typing import Any

from google import genai
from google.genai import types

from permrag.config import get_settings
from permrag.exceptions import LLMError

logger = logging.getLogger(__name__)

_client: genai.Client | None = None

# Only the full-width output is pre-normalised by the API.
_PRE_NORMALISED_DIMENSIONS = 3072

TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"


def get_gemini_client() -> genai.Client:
    global _client
    if _client is None:
        settings = get_settings()
        if settings.gemini_api_key is None:
            raise LLMError("LLM_PROVIDER is 'gemini' but GEMINI_API_KEY is not set")
        _client = genai.Client(
            api_key=settings.gemini_api_key.get_secret_value(),
            http_options=types.HttpOptions(timeout=int(settings.llm_timeout_seconds * 1000)),
        )
        logger.info("gemini client initialised")
    return _client


def _normalise(vector: list[float]) -> list[float]:
    """Scale to unit length. Required for any dimensionality below 3072."""
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0.0:
        return vector
    return [value / magnitude for value in vector]


async def _embed(texts: list[str], task_type: str) -> list[list[float]]:
    if not texts:
        return []

    settings = get_settings()
    client = get_gemini_client()

    try:
        response = await client.aio.models.embed_content(
            model=settings.gemini_embedding_model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=settings.gemini_embedding_dimensions,
            ),
        )
    except Exception as exc:
        raise LLMError(f"Embedding request failed: {exc}") from exc

    embeddings = response.embeddings or []
    if len(embeddings) != len(texts):
        # Guards against the aggregation behaviour of newer embedding models,
        # which can return one combined vector for a multi-input request
        # instead of one per input. Silently accepting that would misalign
        # every chunk with its vector.
        raise LLMError(
            f"Expected {len(texts)} embeddings, got {len(embeddings)}. "
            f"Model '{settings.gemini_embedding_model}' may aggregate multiple "
            "inputs into a single vector."
        )

    # Unlike OpenAI, ContentEmbedding carries no index field; the API returns
    # them in input order, so no re-sorting is possible or needed.
    vectors = [list(item.values or []) for item in embeddings]

    if settings.gemini_embedding_dimensions != _PRE_NORMALISED_DIMENSIONS:
        vectors = [_normalise(v) for v in vectors]

    return vectors


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed document chunks. Order of the returned vectors matches the input."""
    return await _embed(texts, TASK_DOCUMENT)


async def embed_query(text: str) -> list[float]:
    """Embed a user question, optimised for the query side of retrieval."""
    vectors = await _embed([text], TASK_QUERY)
    if not vectors:
        raise LLMError("Embedding returned no vectors")
    return vectors[0]


async def generate_answer(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 900,
) -> tuple[str, dict[str, Any]]:
    """Single generation. Returns (answer_text, usage_details)."""
    settings = get_settings()
    client = get_gemini_client()

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    # Answering from supplied passages is an extraction task, not a reasoning
    # one. Left unset, Gemini 3.x models spend output tokens on thinking that
    # this workload does not benefit from.
    if settings.gemini_thinking_level:
        config.thinking_config = types.ThinkingConfig(
            thinking_level=settings.gemini_thinking_level
        )

    try:
        response = await client.aio.models.generate_content(
            model=settings.gemini_chat_model,
            contents=user_prompt,
            config=config,
        )
    except Exception as exc:
        raise LLMError(f"Chat completion failed: {exc}") from exc

    answer = response.text or ""

    usage: dict[str, Any] = {}
    meta = response.usage_metadata
    if meta is not None:
        usage = {
            "input": meta.prompt_token_count,
            "output": meta.candidates_token_count,
            "total": meta.total_token_count,
        }

    return answer, usage
