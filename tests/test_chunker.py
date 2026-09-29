"""Chunking and hashing behaviour (token-based, reranker tokenizer)."""

from itertools import pairwise

import pytest

from permrag.ingestion.chunker import chunk_page, get_tokenizer, hash_content, normalise

WORDS = ["revenue", "plan", "variance", "margin", "quarter", "budget", "forecast", "headcount", "leave", "policy"]


def token_count(text: str) -> int:
    return len(get_tokenizer()(text, add_special_tokens=False, verbose=False)["input_ids"])


def long_text(n_words: int) -> str:
    return " ".join(f"{WORDS[i % len(WORDS)]}{i}" for i in range(n_words))


def test_short_text_yields_single_chunk():
    chunks = chunk_page("a short page", chunk_size_tokens=400, overlap_tokens=60)
    assert len(chunks) == 1
    assert chunks[0].text == "a short page"


def test_empty_text_yields_nothing():
    assert chunk_page("", 400, 60) == []
    assert chunk_page("   \n  ", 400, 60) == []


def test_long_text_splits_terminates_and_respects_the_token_limit():
    text = long_text(600)
    chunks = chunk_page(text, chunk_size_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    assert [c.index for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert token_count(chunk.text) <= 100
        assert chunk.token_count == token_count(chunk.text)


def test_chunks_never_split_a_word_and_cover_the_whole_text():
    text = long_text(600)
    source_words = text.split()
    chunks = chunk_page(text, chunk_size_tokens=100, overlap_tokens=20)
    for chunk in chunks:
        assert set(chunk.text.split()) <= set(source_words)
    assert chunks[0].text.split()[0] == source_words[0]
    assert chunks[-1].text.split()[-1] == source_words[-1]


def test_consecutive_chunks_overlap():
    chunks = chunk_page(long_text(600), chunk_size_tokens=100, overlap_tokens=20)
    for first, second in pairwise(chunks):
        head = second.text.split()[0]
        assert head in first.text.split(), "next chunk must start inside the previous one"


def test_a_word_longer_than_the_window_still_terminates():
    # 88 characters with no space: 88 word-piece tokens (words over 100
    # characters collapse to a single [UNK] instead, so they never need this).
    word = "qzxv" * 22
    chunks = chunk_page(word, chunk_size_tokens=40, overlap_tokens=10)
    assert len(chunks) > 1
    assert "".join(c.text for c in chunks) == word
    assert all(c.token_count <= 40 for c in chunks)


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        chunk_page("some text here", chunk_size_tokens=50, overlap_tokens=50)


def test_hash_ignores_whitespace_noise():
    assert hash_content("alpha  beta") == hash_content("alpha beta")
    assert hash_content(" alpha beta\n") == hash_content("alpha beta")


def test_hash_changes_with_content():
    assert hash_content("alpha beta") != hash_content("alpha gamma")


def test_hash_changes_with_chunking_settings():
    """Changing how pages are chunked must re-index them, not skip them as unchanged."""
    assert hash_content("alpha beta", "tokens:m:400:60") != hash_content("alpha beta", "tokens:m:256:40")


def test_normalise_collapses_whitespace():
    assert normalise("  a \n\t b  ") == "a b"
