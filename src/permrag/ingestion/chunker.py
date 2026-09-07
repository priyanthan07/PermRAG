import hashlib
import re
from dataclasses import dataclass

_WHITESPACE = re.compile(r"\s+")


@dataclass(slots=True)
class Chunk:
    index: int
    text: str
    word_count: int
    
def normalise(text: str) -> str:
    """Collapse whitespace so trivial formatting edits do not change the hash."""
    return _WHITESPACE.sub(" ", text).strip()

def hash_content(text: str) -> str:
    """SHA-256 of the normalised page content."""
    return hashlib.sha256(normalise(text).encode("utf-8")).hexdigest()

def chunk_page(text: str, chunk_size_words: int, overlap_words: int) -> list[Chunk]:
    """
        Split one page into overlapping word windows.

        Overlap exists so a passage split across a boundary still appears intact
        in at least one chunk.
    """
    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words")

    cleaned = normalise(text)
    if not cleaned:
        return []

    words = cleaned.split(" ")
    if len(words) <= chunk_size_words:
        return [Chunk(index=0, text=cleaned, word_count=len(words))]

    step = chunk_size_words - overlap_words
    chunks: list[Chunk] = []
    index = 0

    for start in range(0, len(words), step):
        window = words[start : start + chunk_size_words]
        if not window:
            break
        chunks.append(Chunk(index=index, text=" ".join(window), word_count=len(window)))
        index += 1
        if start + chunk_size_words >= len(words):
            break

    return chunks