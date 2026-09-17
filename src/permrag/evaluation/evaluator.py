"""
Live RAG quality evaluation via DeepEval.

Runs three reference-free metrics on every chat response, using an LLM as
judge. No golden dataset is needed: each metric works only from the question,
the retrieved context, and the generated answer.

    Faithfulness       — is the answer supported by the retrieved context?
    Answer relevancy   — does the answer actually address the question?
    Contextual relevancy — is the retrieved context on-topic for the question?

Evaluation runs asynchronously *after* the answer has already been returned
to the user: it must never block the response or degrade the chat latency.

Results are pushed to Langfuse as numeric scores on the current trace, so
they appear in the dashboard alongside the retrieval and generation spans.
They're also logged as structured JSON for any non-Langfuse monitoring.

The judge model defaults to gemini-2.0-flash-lite. It's intentionally
different from the answering model (gemini-3.6-flash): a cheap, fast model
is the right choice for a judge that runs on every single request, and using
a different model avoids the circular problem of a model grading its own
work.
"""

import asyncio
import logging
from dataclasses import dataclass

from permrag.config import get_settings

logger = logging.getLogger(__name__)

_metrics_cache: dict | None = None


@dataclass(slots=True)
class EvalScores:
    faithfulness: float | None = None
    faithfulness_reason: str | None = None
    answer_relevancy: float | None = None
    answer_relevancy_reason: str | None = None
    contextual_relevancy: float | None = None
    contextual_relevancy_reason: str | None = None
    error: str | None = None


def _build_metrics():

    global _metrics_cache
    
    if _metrics_cache is not None:
        return _metrics_cache

    settings = get_settings()

    from deepeval.metrics import (AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric)

    judge = _build_judge_model(settings)

    _metrics_cache = {
        "faithfulness": FaithfulnessMetric(
            threshold=settings.eval_faithfulness_threshold,
            model=judge,
            async_mode=True,
            verbose_mode=False,
        ),
        "answer_relevancy": AnswerRelevancyMetric(
            threshold=settings.eval_answer_relevancy_threshold,
            model=judge,
            async_mode=True,
            verbose_mode=False,
        ),
        "contextual_relevancy": ContextualRelevancyMetric(
            threshold=settings.eval_contextual_relevancy_threshold,
            model=judge,
            async_mode=True,
            verbose_mode=False,
        ),
    }
    logger.info(
        "eval metrics initialised",
        extra={"judge_model": settings.eval_judge_model},
    )
    return _metrics_cache


def _build_judge_model(settings):
    """Construct the judge LLM. Gemini by default, OpenAI if configured."""
    judge_model = settings.eval_judge_model

    if judge_model.startswith("gemini"):
        from deepeval.models.llms.gemini_model import GeminiModel

        api_key = None
        if settings.gemini_api_key:
            api_key = settings.gemini_api_key.get_secret_value()
        return GeminiModel(model=judge_model, api_key=api_key)
    else:
        # Falls back to DeepEval's default OpenAI path, which reads
        # OPENAI_API_KEY from the environment.
        return judge_model


async def evaluate_response(
    question: str,
    answer: str,
    retrieval_context: list[str],
) -> EvalScores:
    """
        Score one chat response on all three metrics.

        This is designed to be called inside an asyncio.create_task() so it
        runs concurrently without blocking the HTTP response.
    """
    settings = get_settings()

    if not settings.eval_enabled:
        return EvalScores()

    if not retrieval_context:
        # Nothing was retrieved — faithfulness and contextual relevancy are
        # undefined. The answer should be the "nothing found" message, which
        # is correct by construction. Skip evaluation rather than score a
        # case the metrics weren't designed for.
        return EvalScores()

    try:
        from deepeval.test_case import LLMTestCase

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=retrieval_context,
        )

        metrics = _build_metrics()

        # Run all three metrics concurrently. Each calls the judge LLM
        # independently, so parallelism cuts wall-clock time roughly by 3x.
        scores = await asyncio.gather(
            metrics["faithfulness"].a_measure(test_case, _show_indicator=False),
            metrics["answer_relevancy"].a_measure(test_case, _show_indicator=False),
            metrics["contextual_relevancy"].a_measure(test_case, _show_indicator=False),
            return_exceptions=True,
        )

        result = EvalScores()

        # Each metric overwrites .score and .reason on its own instance
        # after a_measure completes.
        for name, score_or_exc, metric in zip(
            ["faithfulness", "answer_relevancy", "contextual_relevancy"],
            scores,
            [metrics["faithfulness"], metrics["answer_relevancy"], metrics["contextual_relevancy"]],
        ):
            if isinstance(score_or_exc, Exception):
                logger.warning(
                    f"eval metric failed: {name}",
                    extra={"error": str(score_or_exc)},
                )
                continue

            setattr(result, name, metric.score)
            setattr(result, f"{name}_reason", metric.reason)

        logger.info(
            "eval complete",
            extra={
                "faithfulness": result.faithfulness,
                "answer_relevancy": result.answer_relevancy,
                "contextual_relevancy": result.contextual_relevancy,
            },
        )
        return result

    except Exception as exc:
        logger.warning("eval failed entirely", extra={"error": str(exc)})
        return EvalScores(error=str(exc))


def push_scores_to_langfuse(scores: EvalScores) -> None:
    """Write eval scores to the current Langfuse trace as numeric scores.

    Must be called from within an active Langfuse span context. Scores
    appear in the Langfuse dashboard under the trace's Scores tab.
    """
    from permrag.observability.langfuse_client import get_langfuse

    client = get_langfuse()
    if client is None:
        return

    for name in ("faithfulness", "answer_relevancy", "contextual_relevancy"):
        value = getattr(scores, name, None)
        if value is None:
            continue
        reason = getattr(scores, f"{name}_reason", None)
        try:
            client.score_current_trace(
                name=name,
                value=float(value),
                data_type="NUMERIC",
                comment=reason,
            )
        except Exception as exc:
            logger.warning(
                f"failed to push {name} score to langfuse",
                extra={"error": str(exc)},
            )
