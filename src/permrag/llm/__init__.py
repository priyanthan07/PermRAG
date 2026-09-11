"""Provider-agnostic entry point for every LLM operation in PermRAG.

Callers import from here, never from a provider module directly, so swapping
LLM_PROVIDER between 'openai' and 'gemini' changes nothing above this line.

The provider is resolved per call rather than at import time. That keeps this
module import-safe when only one provider's credentials are configured, and
lets the test suite patch these names without any live client existing.
"""

from typing import Any
from permrag.config import get_settings


def _provider():
    if get_settings().llm_provider == "gemini":
        from permrag.llm import gemini_client

        return gemini_client

    from permrag.llm import openai_client

    return openai_client


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


__all__ = ["embed_texts", "embed_query", "generate_answer"]
