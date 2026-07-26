"""Reciprocal Rank Fusion — combining the sparse and dense rankings.

BM25 and dense retrieval fail differently. BM25 misses when the bug report and
the code share no vocabulary; dense retrieval misses when topic-level similarity
isn't enough to pick *this* file out of several conceptually similar ones. Since
the failures are partly independent, combining the two lists usually beats
either.

The awkward part is that their scores are not comparable. A BM25 score is an
unbounded positive number whose scale depends on corpus statistics — 35 in one
repo, 6 in another. A cosine similarity lives in [-1, 1]. Adding or averaging
them is meaningless, and normalising them (min-max, z-score) requires assuming a
distribution shape that neither actually has, and is sensitive to a single
outlier at the top of the list.

**RRF sidesteps the problem by throwing the scores away and using only rank
position.** Each list contributes `1 / (k + rank)` to a file's total. "Ranked
3rd" means the same thing in both lists, so no calibration is needed at all.

The constant `k` (conventionally 60) damps how much the very top ranks dominate.
With k=60, rank 1 contributes 1/61 and rank 2 contributes 1/62 — nearly equal,
so a file needs support from *both* retrievers to climb, rather than one
retriever's confident top pick winning outright. Small k makes the fusion
behave more like "trust whichever list ranked it highest".
"""

from __future__ import annotations

import time

from buglocalizer.retrieval.base import RetrievalResult, ScoredFile


def rrf_fuse(
    results: list[RetrievalResult],
    k: int = 60,
    example_id: str = "",
    method: str = "hybrid",
    top_k: int | None = None,
) -> RetrievalResult:
    """Fuse several ranked lists by reciprocal rank."""
    t0 = time.perf_counter()
    if not results:
        return RetrievalResult(method, example_id, [], 0, 0.0)

    scores: dict[str, float] = {}
    blob_of: dict[str, str] = {}
    for result in results:
        for rank, scored in enumerate(result.ranked, start=1):
            scores[scored.path] = scores.get(scored.path, 0.0) + 1.0 / (k + rank)
            blob_of.setdefault(scored.path, scored.blob_sha)

    ranked = sorted(
        (ScoredFile(path=p, score=s, blob_sha=blob_of[p]) for p, s in scores.items()),
        key=lambda s: (-s.score, s.path),
    )
    if top_k:
        ranked = ranked[:top_k]

    return RetrievalResult(
        method=method,
        example_id=example_id or results[0].example_id,
        ranked=ranked,
        n_candidates=max(r.n_candidates for r in results),
        seconds=time.perf_counter() - t0,
    )
