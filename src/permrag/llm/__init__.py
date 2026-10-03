"""Provider-agnostic entry point for every LLM operation in PermRAG.

Callers import from here, never from a provider module directly, so swapping
LLM_PROVIDER between 'openai' and 'gemini' changes nothing above this line.

The provider is resolved per call rather than at import time. That keeps this
module import-safe when only one provider's credentials are configured, and
lets the test suite patch these names without any live client existing.
"""

from typing import Any

from permrag.config import get_settings
from permrag.exceptions import ConfigurationError


def _provider():
    if get_settings().llm_provider == "gemini":
        from permrag.llm import gemini_client

        return gemini_client

    from permrag.llm import openai_client

    return openai_client


def check_provider_credentials() -> None:
    """Fail at startup, not on the first question, when the active provider has no key."""
    settings = get_settings()
    key = settings.gemini_api_key if settings.llm_provider == "gemini" else settings.openai_api_key
    if key is None:
        raise ConfigurationError(
            f"LLM_PROVIDER is '{settings.llm_provider}' but {settings.llm_provider.upper()}_API_KEY is not set"
        )


def active_chat_model() -> str:
    """Name of the model answering questions, as reported to tracing."""
    settings = get_settings()
    if settings.llm_provider == "gemini":
        return settings.gemini_chat_model
    return settings.openai_chat_model


def active_embedding_model() -> str:
    """Name of the model producing embeddings, as reported to tracing."""
    settings = get_settings()
    if settings.llm_provider == "gemini":
        return settings.gemini_embedding_model
    return settings.openai_embedding_model


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed document chunks. Order of the returned vectors matches the input."""
    return await _provider().embed_texts(texts)


async def embed_query(text: str) -> list[float]:
    """Embed a single user question."""
    return await _provider().embed_query(text)


async def generate_answer(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 900,
) -> tuple[str, dict[str, Any]]:
    """Single completion. Returns (answer_text, usage_details)."""
    return await _provider().generate_answer(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )


__all__ = [
    "check_provider_credentials",
    "active_chat_model",
    "active_embedding_model",
    "embed_texts",
    "embed_query",
    "generate_answer",
]
