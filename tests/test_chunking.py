"""Tests for chunking and the sparse tokenizer.

Chunking is where a silent bug is most expensive: if chunks don't cover the
file, a region of code becomes unretrievable and every dense number is capped
without any error surfacing.
"""

from __future__ import annotations

import pytest

from buglocalizer.indexing.chunking import chunk_offsets
from buglocalizer.retrieval.base import tokenize


def _text(chunks, content):
    return [c.text(content) for c in chunks]


def test_short_file_is_one_chunk():
    content = "def f():\n    return 1\n"
    chunks = chunk_offsets(content, 700, 70)
    assert len(chunks) == 1
    assert chunks[0].text(content) == content


def test_empty_or_whitespace_file_yields_nothing():
    assert chunk_offsets("", 700, 70) == []
    assert chunk_offsets("   \n\n  \t ", 700, 70) == []


def test_chunks_cover_the_entire_file():
    """Every character must appear in at least one chunk, or code is unsearchable."""
    content = "".join(f"line {i}\n" for i in range(500))
    chunks = chunk_offsets(content, 700, 70)
    covered = set()
    for c in chunks:
        covered.update(range(c.start, c.end))
    assert covered == set(range(len(content)))


def test_chunks_overlap_by_the_configured_amount():
    content = "x" * 2000
    chunks = chunk_offsets(content, 700, 70)
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert a.end - b.start == 70


def test_no_chunk_exceeds_max_chars():
    content = "y" * 5000
    for c in chunk_offsets(content, 700, 70):
        assert c.end - c.start <= 700


def test_indices_are_sequential():
    content = "z" * 5000
    chunks = chunk_offsets(content, 700, 70)
    assert [c.idx for c in chunks] == list(range(len(chunks)))


def test_last_chunk_ends_exactly_at_end_of_file():
    content = "q" * 1234
    assert chunk_offsets(content, 700, 70)[-1].end == 1234


def test_zero_overlap_is_allowed():
    content = "a" * 1400
    chunks = chunk_offsets(content, 700, 0)
    assert [(c.start, c.end) for c in chunks] == [(0, 700), (700, 1400)]


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        chunk_offsets("abc", 0, 0)
    with pytest.raises(ValueError):
        chunk_offsets("abc", 100, 100)


# --- tokenizer ---------------------------------------------------------------


def test_tokenize_emits_identifier_whole_and_split():
    """A report saying "send file" must match code defining `send_file`."""
    tokens = tokenize("send_file")
    assert "send_file" in tokens
    assert "send" in tokens and "file" in tokens


def test_tokenize_splits_camel_case():
    tokens = tokenize("DataFrame.toCsv")
    assert "dataframe" in tokens
    assert "data" in tokens and "frame" in tokens
    assert "tocsv" in tokens and "csv" in tokens


def test_tokenize_lowercases():
    assert all(t == t.lower() for t in tokenize("MixedCase IDENTIFIER"))


def test_tokenize_keeps_numbers_drops_punctuation():
    tokens = tokenize("raises TypeError on line 42 -- see (#1234)")
    assert "42" in tokens and "1234" in tokens
    assert not any(c in "".join(tokens) for c in "()#-.")


def test_single_word_identifier_not_duplicated():
    """A name with no boundary must not be emitted twice and double-weighted."""
    assert tokenize("parse").count("parse") == 1
