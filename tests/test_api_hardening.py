"""Error exposure, input limits, login throttling and startup checks."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pydantic
import pytest
from httpx import ASGITransport, AsyncClient

from permrag.api.main import create_app
from permrag.api.routers.chat import ask
from permrag.api.schemas import ChatRequest, UserCreate
from permrag.config import QUESTION_TOKEN_RESERVE, Settings
from permrag.exceptions import (
    ConfigurationError,
    NotFoundError,
    TooManyRequestsError,
    ValidationError,
    VectorStoreError,
)
from permrag.llm import check_provider_credentials
from permrag.security.throttle import LoginThrottle

# --- 5xx details never reach the caller -------------------------------------


async def _call(exc: Exception):
    app = create_app()

    @app.get("/boom")
    async def boom():
        raise exc

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        return await client.get("/boom")


async def test_5xx_detail_is_replaced_by_the_generic_message():
    response = await _call(VectorStoreError("Wrong input: Vector dimension error: expected dim: 3072"))
    assert response.status_code == 503
    assert response.json() == {"detail": "Vector store unavailable"}


async def test_4xx_detail_is_kept():
    response = await _call(NotFoundError("Document 42 not found"))
    assert response.status_code == 404
    assert response.json() == {"detail": "Document 42 not found"}


# --- passwords ----------------------------------------------------------------


def test_password_over_72_bytes_is_a_validation_error_not_a_500():
    with pytest.raises(pydantic.ValidationError, match="72 bytes"):
        UserCreate(email="a@example.com", full_name="A", password="é" * 40)  # 40 chars, 80 bytes
    UserCreate(email="a@example.com", full_name="A", password="é" * 36)  # exactly 72 bytes


# --- login throttle -----------------------------------------------------------


def test_throttle_blocks_after_the_limit_and_resets_on_success():
    throttle = LoginThrottle(max_failures=3, window_seconds=60)
    for _ in range(3):
        throttle.check("a@example.com|1.2.3.4")
        throttle.record_failure("a@example.com|1.2.3.4")
    with pytest.raises(TooManyRequestsError):
        throttle.check("a@example.com|1.2.3.4")
    throttle.check("b@example.com|1.2.3.4")  # other keys are unaffected
    throttle.reset("a@example.com|1.2.3.4")
    throttle.check("a@example.com|1.2.3.4")


def test_throttle_forgets_failures_outside_the_window():
    throttle = LoginThrottle(max_failures=2, window_seconds=60)
    with patch("permrag.security.throttle.time.monotonic", return_value=1000.0):
        throttle.record_failure("k")
        throttle.record_failure("k")
    with patch("permrag.security.throttle.time.monotonic", return_value=1061.0):
        throttle.check("k")


def test_throttle_memory_is_bounded():
    throttle = LoginThrottle(max_keys=100)
    for i in range(500):
        throttle.record_failure(f"user{i}")
    assert len(throttle._failures) <= 100


# --- settings -----------------------------------------------------------------


def test_chunk_size_must_fit_under_the_question_reserve():
    budget = 512 - 3 - QUESTION_TOKEN_RESERVE
    Settings(chunk_size_tokens=budget, reranker_enabled=True)
    with pytest.raises(pydantic.ValidationError, match="does not fit the reranker window"):
        Settings(chunk_size_tokens=budget + 1, reranker_enabled=True)
    # The window only constrains chunks when the reranker actually runs.
    Settings(chunk_size_tokens=budget + 1, reranker_enabled=False)


def test_qdrant_api_key_over_plain_http_is_rejected_outside_local():
    # External host over http: the key could cross a network in clear text.
    with pytest.raises(pydantic.ValidationError, match="unencrypted"):
        Settings(environment="production", qdrant_api_key="k", qdrant_url="http://qdrant.example.com:6333")
    Settings(environment="production", qdrant_api_key="k", qdrant_url="https://qdrant.example.com")
    # Internal service name (docker-compose network): allowed over http.
    Settings(environment="production", qdrant_api_key="k", qdrant_url="http://qdrant:6333")
    Settings(environment="local", qdrant_api_key="k", qdrant_url="http://localhost:6333")


def test_missing_provider_key_fails_the_startup_check():
    fake = SimpleNamespace(llm_provider="gemini", gemini_api_key=None, openai_api_key="x")
    with patch("permrag.llm.get_settings", return_value=fake), pytest.raises(ConfigurationError, match="GEMINI"):
        check_provider_credentials()


# --- question length ------------------------------------------------------------


async def test_question_longer_than_the_reserve_is_rejected_before_answering():
    answerer = AsyncMock()
    long_question = " ".join(["revenue"] * (QUESTION_TOKEN_RESERVE + 20))
    with (
        patch("permrag.api.routers.chat.get_settings", return_value=SimpleNamespace(reranker_enabled=True)),
        pytest.raises(ValidationError, match="too long"),
    ):
        await ask(ChatRequest(question=long_question), SimpleNamespace(id=uuid.uuid4()), answerer)
    assert not answerer.answer.called
