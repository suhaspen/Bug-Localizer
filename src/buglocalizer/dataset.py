"""The Example schema, its on-disk format, and the train/eval split.

An `Example` is one labeled bug-localization instance: a query describing a bug,
the files that actually had to change, and the commit state to search.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

DEV = "dev"
HELDOUT = "heldout"


class Example(BaseModel):
    model_config = ConfigDict(extra="forbid")

    example_id: str
    repo: str

    # The state of the world. `parent_sha` is the buggy state we index and
    # search; `fix_sha` is the answer key and must never be indexed.
    fix_sha: str
    parent_sha: str
    authored_at: datetime

    query_text: str
    gold_files: list[str]

    # Provenance, kept so every filtering and quality decision stays auditable
    # after the fact without re-mining.
    n_files_changed: int
    issue_refs: list[int] = Field(default_factory=list)
    borderline: list[str] = Field(default_factory=list)
    query_scrubbed: bool = False
    split: str | None = None


def write_jsonl(path: Path, examples: list[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for ex in examples:
            fh.write(ex.model_dump_json() + "\n")


def read_jsonl(path: Path) -> list[Example]:
    if not path.exists():
        raise FileNotFoundError(f"no dataset at {path} — run `make mine` first")
    with path.open() as fh:
        return [Example.model_validate_json(line) for line in fh if line.strip()]


def assign_temporal_split(examples: list[Example], dev_fraction: float) -> list[Example]:
    """Split each repo's examples by date: oldest `dev_fraction` dev, newest heldout.

    Per repo rather than one global cutoff date, because the repos have very
    different activity periods — a single calendar cutoff would let the busiest
    repo dominate the eval set and leave the quiet ones unrepresented.

    Sorting breaks ties on `fix_sha` so the split is byte-identical across runs
    even when two commits share a timestamp.
    """
    by_repo: dict[str, list[Example]] = {}
    for ex in examples:
        by_repo.setdefault(ex.repo, []).append(ex)

    out: list[Example] = []
    for repo in sorted(by_repo):
        group = sorted(by_repo[repo], key=lambda e: (e.authored_at, e.fix_sha))
        n_dev = int(len(group) * dev_fraction)
        for i, ex in enumerate(group):
            ex.split = DEV if i < n_dev else HELDOUT
        out.extend(group)
    return out


# --- statistics --------------------------------------------------------------

GOLD_BUCKETS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 10**9)]


def bucket_label(lo: int, hi: int) -> str:
    return str(lo) if lo == hi else (f"{lo}+" if hi >= 10**9 else f"{lo}-{hi}")


class RepoStats(BaseModel):
    repo: str
    total: int
    n_dev: int
    n_heldout: int
    avg_gold_files: float
    median_gold_files: float
    max_gold_files: int
    n_borderline: int
    earliest: datetime | None
    latest: datetime | None
    heldout_earliest: datetime | None
    heldout_latest: datetime | None
    gold_distribution: dict[str, int]


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    return float(s[mid]) if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def compute_repo_stats(examples: list[Example], repo: str) -> RepoStats:
    group = [e for e in examples if e.repo == repo]
    gold_counts = [len(e.gold_files) for e in group]
    heldout = [e for e in group if e.split == HELDOUT]

    distribution: dict[str, int] = {}
    for lo, hi in GOLD_BUCKETS:
        distribution[bucket_label(lo, hi)] = sum(1 for c in gold_counts if lo <= c <= hi)

    dates = sorted(e.authored_at for e in group)
    heldout_dates = sorted(e.authored_at for e in heldout)

    return RepoStats(
        repo=repo,
        total=len(group),
        n_dev=sum(1 for e in group if e.split == DEV),
        n_heldout=len(heldout),
        avg_gold_files=(sum(gold_counts) / len(gold_counts)) if gold_counts else 0.0,
        median_gold_files=_median(gold_counts),
        max_gold_files=max(gold_counts, default=0),
        n_borderline=sum(1 for e in group if e.borderline),
        earliest=dates[0] if dates else None,
        latest=dates[-1] if dates else None,
        heldout_earliest=heldout_dates[0] if heldout_dates else None,
        heldout_latest=heldout_dates[-1] if heldout_dates else None,
        gold_distribution=distribution,
    )


def borderline_counter(examples: list[Example]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for ex in examples:
        counter.update(ex.borderline)
    return counter


def examples_path(data_dir: Path) -> Path:
    return data_dir / "examples.jsonl"


def load_funnel(data_dir: Path) -> dict | None:
    """The per-filter rejection counts saved by the last `mine` run."""
    path = data_dir / "mining_funnel.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_funnel(data_dir: Path, funnel: dict) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "mining_funnel.json").write_text(json.dumps(funnel, indent=2))
