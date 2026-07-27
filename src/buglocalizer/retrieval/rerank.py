"""Cross-encoder reranking of a hybrid shortlist.

**Bi-encoder vs cross-encoder.** Dense retrieval uses a *bi-encoder*: query and
document are embedded separately and compared by cosine. That is what makes it
fast — every document vector is precomputed once, and a query is one forward
pass plus a similarity scan.

A *cross-encoder* reads the query and one document **together**, as a single
input, and outputs a relevance score directly. Because the two texts attend to
each other inside the model, it can notice things a bi-encoder structurally
cannot: that the query's `IntervalIndex` is the same object this code branches
on, rather than merely that both texts are "about indexing".

The price is that nothing can be precomputed. Scoring N candidates costs N
forward passes *at query time*. Measured here at 180 pairs/second, so scoring a
full 600-file corpus would take three seconds per query — hence the two-stage
pattern: a cheap retriever narrows thousands of files to a shortlist, and the
expensive model reorders only that.

**The ceiling this creates.** Reranking can only reorder what the shortlist
contains. If hybrid's top-25 accuracy is 0.93, then no amount of reranking can
push top-10 above 0.93 — that number is reported alongside the results, because
a rerank gain is only interpretable against the headroom it had.
"""

from __future__ import annotations

import time

from buglocalizer.config import Config
from buglocalizer.logging_setup import get_logger
from buglocalizer.retrieval.base import RetrievalResult, ScoredFile

log = get_logger(__name__)

# For each candidate blob, the chunks most similar to the query. Selecting by
# cosine rather than taking the file's head means the reranker sees the windows
# most likely to be relevant, which for a 400 KB file is the difference between
# reading the imports and reading the buggy function.
_CHUNK_SQL = """
SELECT blob_sha, start_char, end_char
FROM (
    SELECT blob_sha, start_char, end_char,
           row_number() OVER (
               PARTITION BY blob_sha ORDER BY embedding <=> %(q)s
           ) AS rn
    FROM chunk
    WHERE repo = %(repo)s AND blob_sha = ANY(%(blobs)s)
) ranked
WHERE rn <= %(m)s
ORDER BY blob_sha, rn
"""


class Reranker:
    """Wraps the cross-encoder. Loaded lazily and reused across queries."""

    def __init__(self, cfg: Config):
        from sentence_transformers import CrossEncoder

        from buglocalizer.indexing.embedder import resolve_device

        self.cfg = cfg
        self.device = resolve_device(cfg.retrieval.device)
        log.info("loading cross-encoder %s on %s", cfg.rerank.model, self.device)
        self.model = CrossEncoder(
            cfg.rerank.model, device=self.device, max_length=cfg.rerank.max_length
        )

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        scores = self.model.predict(
            pairs, batch_size=self.cfg.rerank.batch_size, show_progress_bar=False
        )
        return [float(s) for s in scores]


_RERANKER: Reranker | None = None


def get_reranker(cfg: Config) -> Reranker:
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = Reranker(cfg)
    return _RERANKER


def rerank(
    cfg: Config,
    conn,
    repo: str,
    query_text: str,
    base: RetrievalResult,
    contents: dict[str, str],
    query_vec,
    reranker: Reranker,
) -> RetrievalResult:
    """Re-score the top-N of `base` with the cross-encoder.

    Everything below the shortlist keeps its original order and is appended
    unchanged, so the result is still a full ranking — the eval needs one, and
    truncating here would silently make top-10 uncomparable with the other
    methods.
    """
    t0 = time.perf_counter()
    top_n = cfg.rerank.top_n
    shortlist = base.ranked[:top_n]
    tail = base.ranked[top_n:]
    if not shortlist:
        return RetrievalResult("rerank", base.example_id, [], base.n_candidates, 0.0)

    blob_shas = sorted({s.blob_sha for s in shortlist})
    rows = conn.execute(
        _CHUNK_SQL,
        {"q": query_vec, "repo": repo, "blobs": blob_shas, "m": cfg.rerank.chunks_per_file},
    ).fetchall()

    windows: dict[str, list[tuple[int, int]]] = {}
    for blob_sha, start, end in rows:
        windows.setdefault(blob_sha, []).append((start, end))

    pairs: list[tuple[str, str]] = []
    owners: list[str] = []
    for scored in shortlist:
        content = contents.get(scored.blob_sha)
        if content is None:
            continue
        for start, end in windows.get(scored.blob_sha, [(0, min(len(content), 700))]):
            pairs.append((query_text, content[start:end]))
            owners.append(scored.path)

    scores = reranker.score(pairs)

    # A file scores as the best of its windows: relevant anywhere is relevant.
    best: dict[str, float] = {}
    for path, score in zip(owners, scores, strict=True):
        if score > best.get(path, float("-inf")):
            best[path] = score

    reranked = sorted(
        (
            ScoredFile(path=s.path, score=best.get(s.path, float("-inf")), blob_sha=s.blob_sha)
            for s in shortlist
        ),
        key=lambda s: (-s.score, s.path),
    )
    return RetrievalResult(
        method="rerank",
        example_id=base.example_id,
        ranked=[*reranked, *tail],
        n_candidates=base.n_candidates,
        seconds=time.perf_counter() - t0,
    )
