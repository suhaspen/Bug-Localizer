"""Run every method over an example set and score the rankings.

Design constraint that shapes this module: retrieval is expensive and the three
methods share almost all of their work. So each example is retrieved *once* per
method, hybrid is fused from those same two rankings, and every metric is
computed from the resulting path lists. Running the eval three times, once per
method, would triple the cost for no benefit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from buglocalizer.config import Config
from buglocalizer.corpus import list_corpus, read_blobs, repo_path
from buglocalizer.dataset import Example
from buglocalizer.eval.metrics import average_precision, hit_at_k, reciprocal_rank
from buglocalizer.logging_setup import get_logger
from buglocalizer.retrieval import TokenCache, bm25_search, dense_search
from buglocalizer.retrieval.hybrid import rrf_fuse

log = get_logger(__name__)

METHODS = ["bm25", "dense", "hybrid"]
K_VALUES = [1, 5, 10]


@dataclass
class MethodScores:
    """Accumulated per-example outcomes for one method on one slice."""

    n: int = 0
    hits: dict[int, int] = field(default_factory=lambda: dict.fromkeys(K_VALUES, 0))
    rr_sum: float = 0.0
    ap_sum: float = 0.0
    seconds: float = 0.0

    def add(self, ranked: list[str], gold: list[str], seconds: float) -> None:
        self.n += 1
        for k in K_VALUES:
            if hit_at_k(ranked, gold, k):
                self.hits[k] += 1
        self.rr_sum += reciprocal_rank(ranked, gold)
        self.ap_sum += average_precision(ranked, gold)
        self.seconds += seconds

    def top_k(self, k: int) -> float:
        return self.hits[k] / self.n if self.n else 0.0

    @property
    def mrr(self) -> float:
        return self.rr_sum / self.n if self.n else 0.0

    @property
    def map(self) -> float:
        return self.ap_sum / self.n if self.n else 0.0

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            **{f"top{k}": round(self.top_k(k), 4) for k in K_VALUES},
            "mrr": round(self.mrr, 4),
            "map": round(self.map, 4),
            "mean_seconds": round(self.seconds / self.n, 4) if self.n else 0.0,
        }


@dataclass
class EvalRun:
    """Everything one evaluation produced, ready to serialise."""

    include_tests: bool
    n_examples: int
    per_repo: dict[str, dict[str, MethodScores]] = field(default_factory=dict)
    overall: dict[str, MethodScores] = field(default_factory=dict)
    repo_counts: dict[str, int] = field(default_factory=dict)
    mean_candidates: float = 0.0
    seconds: float = 0.0
    skipped_unindexed: int = 0

    def composition(self) -> list[tuple[str, int, float]]:
        """Per-repo share of the eval set — printed so no aggregate reads as
        'cross-repo' when one repository dominates it."""
        total = sum(self.repo_counts.values()) or 1
        return [(r, n, n / total) for r, n in sorted(self.repo_counts.items())]


def evaluate(
    cfg: Config,
    examples: list[Example],
    conn,
    embedder,
    progress_every: int = 25,
) -> EvalRun:
    """Score BM25, dense and hybrid over `examples` at the configured scope."""
    from buglocalizer.indexing.store import covered_commits

    run = EvalRun(include_tests=cfg.corpus.include_tests, n_examples=0)
    covered = covered_commits(conn, cfg.corpus.include_tests)
    token_cache = TokenCache()
    t0 = time.perf_counter()
    candidate_total = 0

    # Commit order keeps the tokenised-blob cache warm: neighbouring commits
    # share nearly their whole tree.
    ordered = sorted(examples, key=lambda e: (e.repo, e.authored_at, e.fix_sha))

    for i, ex in enumerate(ordered, start=1):
        if (ex.repo, ex.parent_sha) not in covered:
            # Scoring an unindexed commit would silently record dense misses and
            # depress the dense column for a reason unrelated to retrieval.
            run.skipped_unindexed += 1
            continue

        repo_dir = repo_path(cfg, ex.repo)
        files = list_corpus(repo_dir, ex.parent_sha, cfg)
        if not files:
            run.skipped_unindexed += 1
            continue
        candidate_total += len(files)

        contents = read_blobs(repo_dir, sorted({f.blob_sha for f in files}))
        sparse = bm25_search(
            cfg, ex.example_id, ex.query_text, files, contents, token_cache=token_cache
        )
        dense = dense_search(cfg, conn, ex.repo, ex.example_id, ex.query_text, files, embedder)
        hybrid = rrf_fuse([sparse, dense], k=cfg.retrieval.rrf_k, example_id=ex.example_id)

        run.n_examples += 1
        run.repo_counts[ex.repo] = run.repo_counts.get(ex.repo, 0) + 1
        for name, result in (("bm25", sparse), ("dense", dense), ("hybrid", hybrid)):
            per_repo = run.per_repo.setdefault(ex.repo, {})
            per_repo.setdefault(name, MethodScores()).add(
                result.paths(), ex.gold_files, result.seconds
            )
            run.overall.setdefault(name, MethodScores()).add(
                result.paths(), ex.gold_files, result.seconds
            )

        if i % progress_every == 0:
            log.info(
                "%d/%d evaluated | bm25 top10 %.3f | dense top10 %.3f | hybrid top10 %.3f "
                "| token cache %.0f%%",
                i,
                len(ordered),
                run.overall["bm25"].top_k(10),
                run.overall["dense"].top_k(10),
                run.overall["hybrid"].top_k(10),
                100 * token_cache.hit_rate,
            )

    run.seconds = time.perf_counter() - t0
    run.mean_candidates = candidate_total / run.n_examples if run.n_examples else 0.0
    return run
