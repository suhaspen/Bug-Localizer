"""Tests for cross-encoder reranking.

The model and the database are stubbed out. What is worth testing here is the
plumbing around the model — the shortlist boundary, max-over-chunks aggregation,
and preservation of the tail — because each of those silently corrupts the
comparison against hybrid if wrong.
"""

from __future__ import annotations

import pytest

from buglocalizer.config import Config
from buglocalizer.retrieval.base import RetrievalResult, ScoredFile
from buglocalizer.retrieval.rerank import rerank


class FakeConn:
    """Returns (blob_sha, start, end) rows like the chunk-selection query."""

    def __init__(self, rows):
        self._rows = rows

    def execute(self, *_args, **_kwargs):
        return self

    def fetchall(self):
        return self._rows


class FakeReranker:
    """Scores pairs from a lookup keyed on the document text."""

    def __init__(self, by_text: dict[str, float]):
        self.by_text = by_text
        self.calls: list[tuple[str, str]] = []

    def score(self, pairs):
        self.calls.extend(pairs)
        return [self.by_text.get(doc, 0.0) for _q, doc in pairs]


def _base(paths: list[str]) -> RetrievalResult:
    return RetrievalResult(
        method="hybrid",
        example_id="ex",
        ranked=[ScoredFile(path=p, score=1.0, blob_sha=f"sha-{p}") for p in paths],
        n_candidates=len(paths),
        seconds=0.0,
    )


def _cfg(top_n: int = 3, chunks_per_file: int = 1) -> Config:
    cfg = Config()
    cfg.rerank.enabled = True
    cfg.rerank.top_n = top_n
    cfg.rerank.chunks_per_file = chunks_per_file
    return cfg


def test_rerank_reorders_the_shortlist():
    cfg = _cfg(top_n=3)
    base = _base(["a.py", "b.py", "c.py"])
    contents = {"sha-a.py": "AAA", "sha-b.py": "BBB", "sha-c.py": "CCC"}
    rows = [("sha-a.py", 0, 3), ("sha-b.py", 0, 3), ("sha-c.py", 0, 3)]
    reranker = FakeReranker({"AAA": 0.1, "BBB": 0.9, "CCC": 0.5})

    out = rerank(cfg, FakeConn(rows), "r", "q", base, contents, [0.0], reranker)
    assert [s.path for s in out.ranked] == ["b.py", "c.py", "a.py"]
    assert out.method == "rerank"


def test_rerank_leaves_the_tail_untouched_and_below():
    """Everything past the shortlist keeps its order, and stays after it.

    The eval needs a full ranking; truncating at top_n would make top-10
    incomparable with the other methods.
    """
    cfg = _cfg(top_n=2)
    base = _base(["a.py", "b.py", "x.py", "y.py"])
    contents = {f"sha-{p}": p.upper() for p in ["a.py", "b.py", "x.py", "y.py"]}
    rows = [("sha-a.py", 0, 5), ("sha-b.py", 0, 5)]
    reranker = FakeReranker({"A.PY": 0.1, "B.PY": 0.9})

    out = rerank(cfg, FakeConn(rows), "r", "q", base, contents, [0.0], reranker)
    assert [s.path for s in out.ranked] == ["b.py", "a.py", "x.py", "y.py"]
    # Only the shortlist was sent to the model.
    assert len(reranker.calls) == 2


def test_rerank_scores_a_file_by_its_best_chunk():
    """Max over chunks: a file is relevant if any window is."""
    cfg = _cfg(top_n=2, chunks_per_file=2)
    base = _base(["a.py", "b.py"])
    contents = {"sha-a.py": "0123456789", "sha-b.py": "abcdefghij"}
    rows = [
        ("sha-a.py", 0, 5),  # "01234" -> 0.1
        ("sha-a.py", 5, 10),  # "56789" -> 0.95  (best chunk wins)
        ("sha-b.py", 0, 5),  # "abcde" -> 0.5
        ("sha-b.py", 5, 10),  # "fghij" -> 0.4
    ]
    reranker = FakeReranker({"01234": 0.1, "56789": 0.95, "abcde": 0.5, "fghij": 0.4})

    out = rerank(cfg, FakeConn(rows), "r", "q", base, contents, [0.0], reranker)
    assert [s.path for s in out.ranked] == ["a.py", "b.py"]
    assert out.ranked[0].score == pytest.approx(0.95)


def test_rerank_is_deterministic_on_tied_scores():
    cfg = _cfg(top_n=2)
    base = _base(["b.py", "a.py"])
    contents = {"sha-a.py": "AAA", "sha-b.py": "BBB"}
    rows = [("sha-a.py", 0, 3), ("sha-b.py", 0, 3)]
    reranker = FakeReranker({"AAA": 0.5, "BBB": 0.5})

    out = rerank(cfg, FakeConn(rows), "r", "q", base, contents, [0.0], reranker)
    assert [s.path for s in out.ranked] == ["a.py", "b.py"]  # path tiebreak


def test_rerank_empty_base_is_not_an_error():
    out = rerank(_cfg(), FakeConn([]), "r", "q", _base([]), {}, [0.0], FakeReranker({}))
    assert out.ranked == []


def test_rerank_skips_candidates_with_no_content():
    """A blob missing from `contents` must not crash or shift other files."""
    cfg = _cfg(top_n=2)
    base = _base(["a.py", "gone.py"])
    contents = {"sha-a.py": "AAA"}
    rows = [("sha-a.py", 0, 3)]
    reranker = FakeReranker({"AAA": 0.7})

    out = rerank(cfg, FakeConn(rows), "r", "q", base, contents, [0.0], reranker)
    assert [s.path for s in out.ranked] == ["a.py", "gone.py"]
    # The unscored file sinks rather than being dropped from the ranking.
    assert out.ranked[1].score == float("-inf")


def test_rerank_falls_back_to_file_head_when_no_chunks_returned():
    """If a blob has no chunk rows, use its first 700 chars rather than skipping."""
    cfg = _cfg(top_n=1)
    base = _base(["a.py"])
    contents = {"sha-a.py": "x" * 2000}
    reranker = FakeReranker({"x" * 700: 0.42})

    out = rerank(cfg, FakeConn([]), "r", "q", base, contents, [0.0], reranker)
    assert out.ranked[0].score == pytest.approx(0.42)
    assert len(reranker.calls[0][1]) == 700
