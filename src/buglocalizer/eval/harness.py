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

METHODS = ["bm25", "dense", "hybrid", "rerank"]
K_VALUES = [1, 5, 10]


@dataclass
class MethodScores:
    """Accumulated per-example outcomes for one method on one slice."""

    n: int = 0
    hits: dict[int, int] = field(default_factory=lambda: dict.fromkeys(K_VALUES, 0))
    rr_sum: float = 0.0
    ap_sum: float = 0.0
    seconds: float = 0.0
    # Per-example hit/miss, kept so methods can be compared *paired* rather than
    # as two independent accuracies. Both methods saw the same bug reports, and
    # ignoring that discards most of the statistical power.
    flags: dict[int, list[bool]] = field(default_factory=lambda: {k: [] for k in K_VALUES})

    def add(self, ranked: list[str], gold: list[str], seconds: float) -> None:
        self.n += 1
        for k in K_VALUES:
            hit = hit_at_k(ranked, gold, k)
            self.flags[k].append(hit)
            if hit:
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
    # Hybrid's accuracy at the rerank depth. Reranking can only reorder the
    # shortlist, so this is a hard ceiling on what it can achieve — a rerank
    # gain is only interpretable against the headroom it actually had.
    rerank_top_n: int = 0
    shortlist_hits: int = 0

    @property
    def shortlist_ceiling(self) -> float:
        return self.shortlist_hits / self.n_examples if self.n_examples else 0.0

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

    reranker = None
    if cfg.rerank.enabled:
        from buglocalizer.retrieval.rerank import get_reranker, rerank

        reranker = get_reranker(cfg)
        run.rerank_top_n = cfg.rerank.top_n

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
        query_vec = embedder.encode_one(ex.query_text)
        dense = dense_search(
            cfg,
            conn,
            ex.repo,
            ex.example_id,
            ex.query_text,
            files,
            embedder,
            query_vec=query_vec,
        )
        hybrid = rrf_fuse([sparse, dense], k=cfg.retrieval.rrf_k, example_id=ex.example_id)

        results: list[tuple[str, object]] = [("bm25", sparse), ("dense", dense), ("hybrid", hybrid)]
        if reranker is not None:
            reranked = rerank(
                cfg, conn, ex.repo, ex.query_text, hybrid, contents, query_vec, reranker
            )
            results.append(("rerank", reranked))
            if hit_at_k(hybrid.paths(), ex.gold_files, cfg.rerank.top_n):
                run.shortlist_hits += 1

        run.n_examples += 1
        run.repo_counts[ex.repo] = run.repo_counts.get(ex.repo, 0) + 1
        for name, result in results:
            per_repo = run.per_repo.setdefault(ex.repo, {})
            per_repo.setdefault(name, MethodScores()).add(
                result.paths(), ex.gold_files, result.seconds
            )
            run.overall.setdefault(name, MethodScores()).add(
                result.paths(), ex.gold_files, result.seconds
            )

        if i % progress_every == 0:
            extra = (
                f" | rerank {run.overall['rerank'].top_k(10):.3f}"
                if "rerank" in run.overall
                else ""
            )
            log.info(
                "%d/%d | bm25 top10 %.3f | dense %.3f | hybrid %.3f%s | token cache %.0f%%",
                i,
                len(ordered),
                run.overall["bm25"].top_k(10),
                run.overall["dense"].top_k(10),
                run.overall["hybrid"].top_k(10),
                extra,
                100 * token_cache.hit_rate,
            )

    run.seconds = time.perf_counter() - t0
    run.mean_candidates = candidate_total / run.n_examples if run.n_examples else 0.0
    return run
