"""Saving evaluation results, and the held-out peek ledger.

Every run against the held-out set is appended to `results/heldout_log.jsonl`.
That count is a deliberate artifact, not bookkeeping for its own sake: tuning by
hand against a held-out set leaks information slowly, and the only honest way to
bound how much is to record how many times you looked. "I evaluated on held-out
N times" is a statement an interviewer can check.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from buglocalizer.config import Config
from buglocalizer.eval.harness import K_VALUES, METHODS, EvalRun

PEEK_LOG = "heldout_log.jsonl"


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
        return bool(out.strip())
    except Exception:
        return False


def run_to_dict(cfg: Config, run: EvalRun, split: str, label: str) -> dict:
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "label": label,
        "split": split,
        # A result is only reproducible as (code + config), so both are recorded.
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "platform": platform.platform(),
        "config": {
            "include_tests": run.include_tests,
            "chunk_max_chars": cfg.retrieval.chunk_max_chars,
            "chunk_overlap_chars": cfg.retrieval.chunk_overlap_chars,
            "embedding_model": cfg.retrieval.embedding_model,
            "bm25_k1": cfg.retrieval.bm25_k1,
            "bm25_b": cfg.retrieval.bm25_b,
            "rrf_k": cfg.retrieval.rrf_k,
            "seed": cfg.seed,
            "dev_fraction": cfg.split.dev_fraction,
        },
        "n_examples": run.n_examples,
        "skipped_unindexed": run.skipped_unindexed,
        "mean_candidates": round(run.mean_candidates, 1),
        "seconds": round(run.seconds, 1),
        "composition": [
            {"repo": r, "n": n, "share": round(share, 4)} for r, n, share in run.composition()
        ],
        "overall": {m: run.overall[m].as_dict() for m in METHODS if m in run.overall},
        "per_repo": {
            repo: {m: scores[m].as_dict() for m in METHODS if m in scores}
            for repo, scores in sorted(run.per_repo.items())
        },
    }


def save_results(cfg: Config, payload: dict) -> Path:
    results_dir = cfg.paths.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = results_dir / f"{stamp}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def record_heldout_peek(cfg: Config, payload: dict, result_path: Path) -> int:
    """Append to the peek ledger and return the running count."""
    log_path = cfg.paths.results_dir / PEEK_LOG
    entry = {
        "timestamp": payload["timestamp"],
        "label": payload["label"],
        "git_sha": payload["git_sha"],
        "include_tests": payload["config"]["include_tests"],
        "n_examples": payload["n_examples"],
        "result_file": result_path.name,
    }
    with log_path.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return sum(1 for _ in log_path.open())


def peek_count(cfg: Config) -> int:
    log_path = cfg.paths.results_dir / PEEK_LOG
    if not log_path.exists():
        return 0
    return sum(1 for line in log_path.open() if line.strip())


def render_markdown(payloads: list[dict]) -> str:
    """One markdown document covering every scope that was run."""
    lines: list[str] = ["# Evaluation results", ""]
    first = payloads[0]
    lines += [
        f"- generated: `{first['timestamp']}`",
        f"- git: `{first['git_sha']}`{' (dirty tree)' if first['git_dirty'] else ''}",
        f"- split: `{first['split']}`",
        f"- embedding model: `{first['config']['embedding_model']}`",
        f"- chunk: {first['config']['chunk_max_chars']} chars / "
        f"{first['config']['chunk_overlap_chars']} overlap · RRF k={first['config']['rrf_k']}",
        "",
    ]

    for payload in payloads:
        scope = (
            "tests INCLUDED (harder)" if payload["config"]["include_tests"] else "tests excluded"
        )
        lines += [
            f"## Corpus scope: {scope}",
            "",
            f"{payload['n_examples']} examples · mean {payload['mean_candidates']:.0f} "
            f"candidate files per query · {payload['seconds'] / 60:.1f} min",
            "",
            "**Composition of the aggregate** — an aggregate is only as cross-repo "
            "as this table says it is:",
            "",
            "| repo | examples | share |",
            "| --- | ---: | ---: |",
        ]
        for c in payload["composition"]:
            lines.append(f"| {c['repo']} | {c['n']:,} | {100 * c['share']:.1f}% |")
        lines += ["", "### Aggregate", "", _table(payload["overall"]), ""]
        for repo, scores in payload["per_repo"].items():
            n = next(iter(scores.values()))["n"]
            lines += [f"### {repo} (n={n})", "", _table(scores), ""]

    if len(payloads) == 2:
        lines += ["## Scope comparison — top-10, aggregate", "", _scope_delta(payloads), ""]
    return "\n".join(lines)


def _table(scores: dict) -> str:
    head = "| method | " + " | ".join(f"top-{k}" for k in K_VALUES) + " | MRR | MAP |"
    sep = "| --- | " + " | ".join("---:" for _ in K_VALUES) + " | ---: | ---: |"
    rows = [head, sep]
    best = {
        col: max((s.get(col, 0) for s in scores.values()), default=0)
        for col in [f"top{k}" for k in K_VALUES] + ["mrr", "map"]
    }
    for method in METHODS:
        if method not in scores:
            continue
        s = scores[method]
        cells = []
        for col in [f"top{k}" for k in K_VALUES] + ["mrr", "map"]:
            v = s.get(col, 0.0)
            cells.append(f"**{v:.3f}**" if v == best[col] and v > 0 else f"{v:.3f}")
        rows.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _scope_delta(payloads: list[dict]) -> str:
    narrow = next(p for p in payloads if not p["config"]["include_tests"])
    wide = next(p for p in payloads if p["config"]["include_tests"])
    rows = [
        "| method | tests excluded | tests included | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        if method not in narrow["overall"] or method not in wide["overall"]:
            continue
        a = narrow["overall"][method]["top10"]
        b = wide["overall"][method]["top10"]
        rows.append(f"| {method} | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
    return "\n".join(rows)
