import asyncio
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

from permrag.config import get_settings
from permrag.exceptions import PermRAGError

logger = logging.getLogger(__name__)

class RerankError(PermRAGError):
    status_code = 503
    detail = "Reranking failed"
    
class Scoreable(Protocol):
    """Anything carrying text to score. SearchHit satisfies this."""
    text: str
    
_encoder = None
_unavailable = False

def _load_encoder():
    """
        Load the model once per process.
    
        The first call downloads weights to the HuggingFace cache, which takes a
        moment; every later call is local. Imports happen inside the function so
        importing this module never pulls in torch when reranking is switched off.
    """
    global _encoder, _unavailable
 
    if _unavailable:
        return None
    if _encoder is not None:
        return _encoder
 
    settings = get_settings()
    
    try:
        import torch
        from sentence_transformers import CrossEncoder
 
        # torch grabs every core by default. Inside an async server that also
        # runs rerank calls in worker threads, that oversubscribes the CPU and
        # makes concurrent requests slower rather than faster.
        if settings.reranker_threads > 0:
            torch.set_num_threads(settings.reranker_threads)
 
        _encoder = CrossEncoder(
            settings.reranker_model,
            device=settings.reranker_device or None,
            max_length=settings.reranker_max_length,
        )
        logger.info(
            "reranker loaded",
            extra={
                "model": settings.reranker_model,
                "device": settings.reranker_device or "auto",
            },
        )
        return _encoder
    except Exception as exc:
        # A missing model or absent dependency must not take down retrieval:
        # without reranking the system still returns permission-correct
        # results, just ordered by embedding score as before.
        logger.warning(
            "reranker unavailable, falling back to embedding order",
            extra={"error": str(exc)},
        )
        _unavailable = True
        return None

def warm_reranker() -> None:
    """
        Load the model at startup rather than inside the first user's query.
    """
    if get_settings().reranker_enabled:
        _load_encoder()
        
def _rank_blocking(query: str, texts: list[str]) -> list[tuple[int, float]]:
    """
        Score every text. Returns (index_into_texts, score), best first.

        The model scores all texts regardless of any top-k, so nothing is saved
        by truncating here -- and the rejected scores are what make a trace
        useful for judging the reranker.
    """

    encoder = _load_encoder()
    if encoder is None:
        raise RerankError("Reranker not loaded")

    settings = get_settings()
    ranked = encoder.rank(
        query,
        texts,
        top_k=None,
        return_documents=False,
        batch_size=settings.reranker_batch_size,
        show_progress_bar=False,
    )
    return [(int(r["corpus_id"]), float(r["score"])) for r in ranked]


RerankStatus = Literal["kept", "below_floor", "beyond_top_k"]


@dataclass(slots=True)
class RerankDecision[T]:
    """
        One candidate's outcome.

        ``below_floor``: scored under ``reranker_min_score``.
        ``beyond_top_k``: passed the floor but ranked past ``top_k``.
        ``score``/``rank`` are None when the reranker did not run.
    """

    candidate: T
    status: RerankStatus
    score: float | None = None
    rank: int | None = None


async def rerank_all[T: Scoreable](
    query: str, candidates: Sequence[T], top_k: int
) -> list[RerankDecision[T]]:
    """
        Every candidate with its outcome, best first.

        Kept = the top_k by score, minus any below the floor. If the reranker
        does not run, the incoming embedding order is kept instead. Every
        failure path degrades ranking quality, never correctness -- the
        permission filter was applied upstream and is not touched here.
    """
    settings = get_settings()
    fallback: list[RerankDecision[T]] = [
        RerankDecision(c, "kept" if position < top_k else "beyond_top_k")
        for position, c in enumerate(candidates)
    ]

    if not settings.reranker_enabled or not candidates:
        return fallback
    if _load_encoder() is None:
        return fallback

    texts = [c.text for c in candidates]

    try:
        # Inference is blocking CPU work. Running it directly would stall the
        # event loop for every other in-flight request.
        ranked = await asyncio.to_thread(_rank_blocking, query, texts)
    except Exception as exc:
        logger.warning(
            "reranking failed, falling back to embedding order",
            extra={"error": str(exc)},
        )
        return fallback

    floor = settings.reranker_min_score
    decisions: list[RerankDecision[T]] = []
    for index, score in ranked:
        # Validate the index rather than trusting it: an out-of-range corpus_id
        # would silently pair a score with the wrong chunk.
        if not 0 <= index < len(candidates):
            logger.warning("reranker returned an out-of-range index", extra={"index": index})
            continue
        # Scores arrive best first, so every candidate ranked ahead of this
        # one also passed the floor.
        position = len(decisions)
        if score < floor:
            status: RerankStatus = "below_floor"
        elif position < top_k:
            status = "kept"
        else:
            status = "beyond_top_k"
        decisions.append(RerankDecision(candidates[index], status, score, position + 1))

    if not decisions:
        return fallback

    counts = Counter(d.status for d in decisions)
    logger.info(
        "rerank complete",
        extra={
            "candidates": len(candidates),
            "kept": counts["kept"],
            "below_floor": counts["below_floor"],
            "beyond_top_k": counts["beyond_top_k"],
            "top_score": round(ranked[0][1], 4),
        },
    )
    return decisions


async def rerank_with_scores[T: Scoreable](
    query: str, candidates: Sequence[T], top_k: int
) -> list[tuple[T, float | None]]:
    """
        The kept candidates, best first, each with its rerank score.

        A None score means the reranker did not run and the order is the
        incoming embedding order.
    """
    decisions = await rerank_all(query, candidates, top_k)
    return [(d.candidate, d.score) for d in decisions if d.status == "kept"]

async def rerank[T: Scoreable](query: str, candidates: Sequence[T], top_k: int) -> list[T]:
    """Top_k candidates, best first, without the scores."""
    return [candidate for candidate, _ in await rerank_with_scores(query, candidates, top_k)]
