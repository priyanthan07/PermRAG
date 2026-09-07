import logging
from typing import Any

from openai import AsyncOpenAI

from permrag.config import get_settings
from permrag.exceptions import LLMError

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None

def get_openai_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
        )
        logger.info("openai client initialised")
    return _client

async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch. Order of the returned vectors matches the input."""
    if not texts:
        return []

    settings = get_settings()
    client = get_openai_client()

    try:
        response = await client.embeddings.create(
            model=settings.openai_embedding_model,
            input=texts,
            dimensions=settings.openai_embedding_dimensions,
        )
    except Exception as exc:
        raise LLMError(f"Embedding request failed: {exc}") from exc

    # The API may return items out of order; sort by index before unpacking.
    ordered = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in ordered]

async def embed_query(text: str) -> list[float]:
    vectors = await embed_texts([text])
    if not vectors:
        raise LLMError("Embedding returned no vectors")
    return vectors[0]

async def generate_answer(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 900,
) -> tuple[str, dict[str, Any]]:
    """Single chat completion. Returns (answer_text, usage_details)."""
    settings = get_settings()
    client = get_openai_client()

    try:
        response = await client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        raise LLMError(f"Chat completion failed: {exc}") from exc

    choice = response.choices[0]
    answer = choice.message.content or ""

    usage: dict[str, Any] = {}
    if response.usage is not None:
        usage = {
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
            "total": response.usage.total_tokens,
        }

    return answer, usage
