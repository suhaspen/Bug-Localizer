"""Ranking metrics.

Pure functions over (ranked_paths, gold_paths). No git, no database, no model —
so every number this project reports can be checked against hand-computed
examples in a unit test.

All three metrics answer "did we put a correct file near the top?", but they
disagree about what counts, and the disagreement is the point:

  top-k accuracy — did ANY gold file land in the first k? Binary per example.
  MRR            — how high was the FIRST gold file? Rewards rank 1 over rank 5.
  MAP            — how well were ALL gold files ranked? The only one that
                   notices when an example has four gold files and you found one.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def hit_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> bool:
    """True if at least one gold file appears in the top k.

    Deliberately "any", not "all": a developer handed 10 files needs one real
    lead to start pulling the thread, and most examples (77%) have a single gold
    file anyway.
    """
    gold_set = set(gold)
    return any(path in gold_set for path in ranked[:k])


def reciprocal_rank(ranked: Sequence[str], gold: Iterable[str]) -> float:
    """1 / (rank of the first gold file), or 0.0 if none is ranked.

    Rank 1 scores 1.0, rank 2 scores 0.5, rank 10 scores 0.1. The sharp drop is
    intentional — it encodes that being second is much better than being tenth,
    a distinction top-k accuracy throws away entirely.
    """
    gold_set = set(gold)
    for i, path in enumerate(ranked, start=1):
        if path in gold_set:
            return 1.0 / i
    return 0.0


def average_precision(ranked: Sequence[str], gold: Iterable[str]) -> float:
    """Mean of precision@i taken at each position i where a gold file appears.

    Worked example. Gold = {a, b}, ranking = [x, a, y, b]:
      position 2 holds `a`  -> precision so far = 1/2 = 0.500
      position 4 holds `b`  -> precision so far = 2/4 = 0.500
      AP = (0.500 + 0.500) / 2 = 0.500

    Divided by the number of gold files, not by the number found, so failing to
    rank a gold file at all is penalised rather than ignored. An example whose
    gold files are entirely missing scores 0.
    """
    gold_set = set(gold)
    if not gold_set:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for i, path in enumerate(ranked, start=1):
        if path in gold_set:
            hits += 1
            precision_sum += hits / i
    return precision_sum / len(gold_set)


def recall_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Fraction of gold files that appear in the top k."""
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    return sum(1 for path in ranked[:k] if path in gold_set) / len(gold_set)
