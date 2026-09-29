"""
    Page chunking, measured in tokens.

    Sizes are counted with the reranker's own tokenizer. The cross-encoder
    reranker is the smallest context window in the retrieval path: it reads
    [question + chunk] together, capped at ``reranker_max_length`` tokens, and
    silently scores only the head of anything longer. Counting in its tokens
    makes "the chunk fits" a guarantee rather than a words-to-tokens estimate,
    which varies with content (figures, codes and tables split into more tokens
    per word).
"""

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from permrag.config import get_settings

_WHITESPACE = re.compile(r"\s+")


@dataclass(slots=True)
class Chunk:
    index: int
    text: str
    token_count: int


@lru_cache(maxsize=1)
def get_tokenizer() -> Any:
    """The reranker model's tokenizer, loaded once. Imported lazily: it pulls in transformers."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(get_settings().reranker_model)


def chunking_fingerprint() -> str:
    """
        Identifies how pages are chunked. Part of every page hash, so changing
        the tokenizer or sizes re-indexes a page on its next ingest instead of
        skipping it as "unchanged".
    """
    settings = get_settings()
    return f"tokens:{settings.reranker_model}:{settings.chunk_size_tokens}:{settings.chunk_overlap_tokens}"


def normalise(text: str) -> str:
    """Collapse whitespace so trivial formatting edits do not change the hash."""
    return _WHITESPACE.sub(" ", text).strip()


def hash_content(text: str, fingerprint: str = "") -> str:
    """SHA-256 of the normalised page content plus the chunking fingerprint."""
    return hashlib.sha256(f"{fingerprint}\n{normalise(text)}".encode()).hexdigest()


def chunk_page(
    text: str,
    chunk_size_tokens: int,
    overlap_tokens: int,
    tokenizer: Any | None = None,
) -> list[Chunk]:
    """
        Split one page into overlapping token windows.

        Window edges snap back to the start of a word, so no chunk ends or
        begins mid-word; a chunk may therefore be a few tokens shorter than
        ``chunk_size_tokens``, never longer. Overlap exists so a passage split
        across a boundary still appears intact in at least one chunk.
    """
    if overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_size_tokens")

    cleaned = normalise(text)
    if not cleaned:
        return []

    tokenizer = tokenizer or get_tokenizer()
    # Offsets map each token back to its characters, so chunks are cut from
    # the original text rather than re-assembled from word pieces.
    offsets = tokenizer(
        cleaned, add_special_tokens=False, return_offsets_mapping=True, verbose=False
    )["offset_mapping"]
    total = len(offsets)

    if total <= chunk_size_tokens:
        return [Chunk(index=0, text=cleaned, token_count=total)]

    # A token starts a word when a space separates it from the previous one.
    # Punctuation and word pieces touch their neighbour, so they never do.
    starts_word = [i == 0 or offsets[i][0] > offsets[i - 1][1] for i in range(total)]

    chunks: list[Chunk] = []
    start = 0
    while True:
        end = min(start + chunk_size_tokens, total)
        if end < total:
            snapped = end
            while snapped > start and not starts_word[snapped]:
                snapped -= 1
            # A single "word" longer than the window (e.g. a long URL) is
            # cut where it is rather than looping forever.
            if snapped > start:
                end = snapped

        chunks.append(
            Chunk(
                index=len(chunks),
                text=cleaned[offsets[start][0] : offsets[end - 1][1]],
                token_count=end - start,
            )
        )
        if end >= total:
            break

        next_start = end - overlap_tokens
        while next_start > start and not starts_word[next_start]:
            next_start -= 1
        # Always move forward, even if snapping ate the whole overlap.
        start = next_start if next_start > start else end

    return chunks
