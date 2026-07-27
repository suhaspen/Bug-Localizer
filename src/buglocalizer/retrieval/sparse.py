"""BM25 — the sparse baseline.

BM25 ranks a file by how many *rare* query words it contains, with two
corrections over naive word counting: a word appearing 50 times is not 50x more
relevant than once (saturation, controlled by k1), and long files should not win
just by containing more words (length normalisation, controlled by b).

We use `rank_bm25`'s `BM25Okapi`, which is a direct implementation of the
textbook Okapi BM25 formula with those exact parameters exposed. This is
deliberately not Postgres full-text search: `ts_rank_cd` is a different
weighting scheme, so calling it "our BM25 baseline" would be inaccurate. See
docs/decisions.md D7 for the measurements behind that.

A fresh index is built per query, because each query searches a different commit
and BM25's IDF statistics depend on the corpus. Measured on pandas: tokenising a
tree costs ~2.4s but is cacheable by blob hash, while the index build itself is
only ~0.5s and is not.
"""

from __future__ import annotations

import time
from collections import OrderedDict

from rank_bm25 import BM25Okapi

from buglocalizer.config import Config
from buglocalizer.corpus import CorpusFile
from buglocalizer.retrieval.base import RetrievalResult, ScoredFile, tokenize


class TokenCache:
    """Bounded LRU of tokenised blobs.

    Tokenisation dominates sparse retrieval cost, and consecutive examples share
    nearly their whole tree, so the hit rate is very high when examples are
    processed in commit order.

    The bound matters more than it looks. A pandas commit at the wide corpus
    scope has ~1,400 files, so the cache must hold at least one commit's worth to
    be useful at all — but holding several thousand tokenised blobs runs to
    hundreds of MB, and on a memory-pressured machine that is the difference
    between an eval that runs and one that gets paged out and crawls. 1,536 is
    one wide pandas tree plus headroom.
    """

    def __init__(self, maxsize: int = 1536):
        self.maxsize = maxsize
        self._data: OrderedDict[str, list[str]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, blob_sha: str, content: str) -> list[str]:
        cached = self._data.get(blob_sha)
        if cached is not None:
            self.hits += 1
            self._data.move_to_end(blob_sha)
            return cached
        self.misses += 1
        tokens = tokenize(content)
        self._data[blob_sha] = tokens
        if len(self._data) > self.maxsize:
            self._data.popitem(last=False)
        return tokens

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def bm25_search(
    cfg: Config,
    example_id: str,
    query_text: str,
    files: list[CorpusFile],
    contents: dict[str, str],
    token_cache: TokenCache | None = None,
    top_k: int | None = None,
) -> RetrievalResult:
    t0 = time.perf_counter()
    cache = token_cache or TokenCache()

    present = [f for f in files if f.blob_sha in contents]
    corpus_tokens = [cache.get(f.blob_sha, contents[f.blob_sha]) for f in present]
    if not corpus_tokens:
        return RetrievalResult("bm25", example_id, [], 0, time.perf_counter() - t0)

    bm25 = BM25Okapi(corpus_tokens, k1=cfg.retrieval.bm25_k1, b=cfg.retrieval.bm25_b)
    scores = bm25.get_scores(tokenize(query_text))

    ranked = sorted(
        (
            ScoredFile(path=f.path, score=float(s), blob_sha=f.blob_sha)
            for f, s in zip(present, scores, strict=True)
        ),
        key=lambda s: (-s.score, s.path),  # path tiebreak keeps ranking deterministic
    )
    if top_k:
        ranked = ranked[:top_k]
    return RetrievalResult("bm25", example_id, ranked, len(present), time.perf_counter() - t0)
