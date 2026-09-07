"""Chunking and hashing behaviour."""

from permrag.ingestion.chunker import chunk_page, hash_content, normalise


def test_short_text_yields_single_chunk():
    chunks = chunk_page("a short page", chunk_size_words=280, overlap_words=50)
    assert len(chunks) == 1
    assert chunks[0].text == "a short page"


def test_empty_text_yields_nothing():
    assert chunk_page("", 280, 50) == []
    assert chunk_page("   \n  ", 280, 50) == []


def test_long_text_splits_and_terminates():
    text = " ".join(f"w{i}" for i in range(1000))
    chunks = chunk_page(text, chunk_size_words=280, overlap_words=50)
    assert len(chunks) == 5
    assert [c.index for c in chunks] == [0, 1, 2, 3, 4]


def test_overlap_is_exact():
    text = " ".join(f"w{i}" for i in range(600))
    chunks = chunk_page(text, chunk_size_words=280, overlap_words=50)
    first = chunks[0].text.split()
    second = chunks[1].text.split()
    assert first[-50:] == second[:50]


def test_overlap_must_be_smaller_than_chunk():
    try:
        chunk_page("some text here", chunk_size_words=50, overlap_words=50)
    except ValueError:
        return
    raise AssertionError("expected ValueError for overlap >= chunk size")


def test_hash_ignores_whitespace_noise():
    assert hash_content("alpha  beta") == hash_content("alpha beta")
    assert hash_content(" alpha beta\n") == hash_content("alpha beta")


def test_hash_changes_with_content():
    assert hash_content("alpha beta") != hash_content("alpha gamma")


def test_normalise_collapses_whitespace():
    assert normalise("  a \n\t b  ") == "a b"
    