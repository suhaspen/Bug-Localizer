"""Tests for the ranking metrics and RRF fusion.

Every expected value here is computed by hand in the test itself, not by
running the code and pasting the output. That distinction matters: a metric bug
does not crash, it just shifts every number in the results table by a plausible
amount, and a test that asserts whatever the code currently does would happily
lock the bug in.
"""

from __future__ import annotations

import math

import pytest

from buglocalizer.eval.metrics import (
    average_precision,
    hit_at_k,
    mcnemar,
    recall_at_k,
    reciprocal_rank,
    unpaired_se,
)
from buglocalizer.retrieval.base import RetrievalResult, ScoredFile
from buglocalizer.retrieval.hybrid import rrf_fuse

RANKING = ["x.py", "a.py", "y.py", "b.py", "z.py"]


# --- top-k accuracy ----------------------------------------------------------


@pytest.mark.parametrize(
    ("k", "expected"),
    [(1, False), (2, True), (5, True), (10, True)],
)
def test_hit_at_k_boundary(k, expected):
    """Gold sits at rank 2, so top-1 misses and everything from top-2 hits."""
    assert hit_at_k(RANKING, ["a.py"], k) is expected


def test_hit_at_k_is_any_not_all():
    """One gold file in the window is a hit even when others are missing."""
    assert hit_at_k(RANKING, ["a.py", "nowhere.py"], 2) is True


def test_hit_at_k_no_gold_in_ranking():
    assert hit_at_k(RANKING, ["nowhere.py"], 10) is False


def test_hit_at_k_empty_ranking():
    assert hit_at_k([], ["a.py"], 10) is False


# --- MRR ---------------------------------------------------------------------


def test_reciprocal_rank_uses_the_first_gold_only():
    """Gold at ranks 2 and 4 -> 1/2, because only the first one counts."""
    assert reciprocal_rank(RANKING, ["a.py", "b.py"]) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("gold", "expected"),
    [(["x.py"], 1.0), (["a.py"], 0.5), (["y.py"], 1 / 3), (["z.py"], 0.2)],
)
def test_reciprocal_rank_by_position(gold, expected):
    assert reciprocal_rank(RANKING, gold) == pytest.approx(expected)


def test_reciprocal_rank_zero_when_absent():
    assert reciprocal_rank(RANKING, ["nowhere.py"]) == 0.0


# --- MAP ---------------------------------------------------------------------


def test_average_precision_worked_example():
    """Gold = {a, b}; ranking [x, a, y, b].

    a at position 2 -> precision 1/2 = 0.5
    b at position 4 -> precision 2/4 = 0.5
    AP = (0.5 + 0.5) / 2 = 0.5
    """
    ap = average_precision(["x.py", "a.py", "y.py", "b.py"], ["a.py", "b.py"])
    assert ap == pytest.approx(0.5)


def test_average_precision_perfect_ranking_is_one():
    assert average_precision(["a.py", "b.py", "x.py"], ["a.py", "b.py"]) == pytest.approx(1.0)


def test_average_precision_penalises_unfound_gold():
    """Finding 1 of 2 gold files at rank 1 scores 0.5, not 1.0.

    This is the property that distinguishes MAP from MRR, and it depends on
    dividing by the number of gold files rather than the number found.
    """
    assert average_precision(["a.py", "x.py"], ["a.py", "missing.py"]) == pytest.approx(0.5)
    assert reciprocal_rank(["a.py", "x.py"], ["a.py", "missing.py"]) == pytest.approx(1.0)


def test_average_precision_single_gold_equals_reciprocal_rank():
    """With one gold file MAP degenerates to MRR — worth pinning, because 77%
    of this dataset has exactly one gold file, so the two columns should track."""
    for gold in (["x.py"], ["a.py"], ["y.py"], ["z.py"]):
        assert average_precision(RANKING, gold) == pytest.approx(reciprocal_rank(RANKING, gold))


def test_average_precision_zero_when_nothing_found():
    assert average_precision(RANKING, ["nowhere.py"]) == 0.0


def test_average_precision_empty_gold():
    assert average_precision(RANKING, []) == 0.0


def test_average_precision_rewards_higher_placement():
    early = average_precision(["a.py", "b.py", "x.py", "y.py"], ["a.py", "b.py"])
    late = average_precision(["x.py", "y.py", "a.py", "b.py"], ["a.py", "b.py"])
    assert early > late


# --- recall ------------------------------------------------------------------


def test_recall_at_k():
    assert recall_at_k(RANKING, ["a.py", "b.py"], 2) == pytest.approx(0.5)
    assert recall_at_k(RANKING, ["a.py", "b.py"], 5) == pytest.approx(1.0)
    assert recall_at_k(RANKING, [], 5) == 0.0


# --- RRF ---------------------------------------------------------------------


def _result(method: str, paths: list[str]) -> RetrievalResult:
    return RetrievalResult(
        method=method,
        example_id="ex",
        ranked=[ScoredFile(path=p, score=1.0, blob_sha=f"sha-{p}") for p in paths],
        n_candidates=len(paths),
        seconds=0.0,
    )


def test_rrf_worked_example():
    """k=60. `b.py` is 2nd in one list and 1st in the other:
        1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
    `a.py` is 1st and 3rd:
        1/61 + 1/63 = 0.016393 + 0.015873 = 0.032266
    So b.py wins despite a.py holding the single highest rank.
    """
    fused = rrf_fuse(
        [_result("bm25", ["a.py", "b.py"]), _result("dense", ["b.py", "x.py", "a.py"])]
    )
    assert fused.ranked[0].path == "b.py"
    assert fused.ranked[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused.ranked[1].path == "a.py"
    assert fused.ranked[1].score == pytest.approx(1 / 61 + 1 / 63)


def test_rrf_ignores_score_magnitude():
    """The whole point of RRF: BM25's 35.6 and cosine's 0.47 aren't comparable,
    so only rank position may influence the result."""
    a = _result("bm25", ["a.py", "b.py"])
    b = _result("dense", ["a.py", "b.py"])
    for scored in a.ranked:
        object.__setattr__(scored, "score", 9999.0)
    for scored in b.ranked:
        object.__setattr__(scored, "score", 0.0001)
    fused = rrf_fuse([a, b])
    assert [s.path for s in fused.ranked] == ["a.py", "b.py"]


def test_rrf_agreement_beats_one_strong_opinion():
    """A file both lists rank 2nd should beat one that's 1st in a single list."""
    fused = rrf_fuse(
        [_result("bm25", ["only.py", "both.py"]), _result("dense", ["other.py", "both.py"])]
    )
    assert fused.ranked[0].path == "both.py"


def test_rrf_second_in_both_always_beats_first_in_one():
    """A structural property of RRF, independent of k.

    1st-in-one scores 1/(k+1); 2nd-in-both scores 2/(k+2). The second is larger
    for every k >= 0, since k+2 > 2(k+1) has no non-negative solution. So no
    choice of k lets a single retriever's confident top pick outrank a file both
    retrievers liked. Pinned because it is the mechanism by which hybrid is
    supposed to help, and it should not be tunable away by accident.
    """
    lists = [_result("bm25", ["top.py", "both.py"]), _result("dense", ["other.py", "both.py"])]
    for k in (1, 10, 60, 600):
        assert rrf_fuse(lists, k=k).ranked[0].path == "both.py", k


def test_rrf_k_controls_how_steeply_rank_decays():
    """k does matter, just further down the list.

    `top.py` is 1st in one list; `both.py` is 4th in both.
      k=1  -> top 1/2 = 0.500 vs both 2/5  = 0.400  -> top wins
      k=60 -> top 1/61 ≈ 0.0164 vs both 2/64 ≈ 0.0313 -> both wins
    Small k makes a high rank worth much more than agreement further down.
    """
    lists = [
        _result("bm25", ["top.py", "p.py", "q.py", "both.py"]),
        _result("dense", ["r.py", "s.py", "t.py", "both.py"]),
    ]
    # Compare the two scores directly: asserting on the winner would be decided
    # by the path tiebreak, since `r.py` also holds a rank-1 slot.
    for k, top_should_win in ((1, True), (60, False)):
        by_path = {s.path: s.score for s in rrf_fuse(lists, k=k).ranked}
        assert (by_path["top.py"] > by_path["both.py"]) is top_should_win, k


def test_rrf_union_of_both_lists():
    fused = rrf_fuse([_result("bm25", ["a.py"]), _result("dense", ["b.py"])])
    assert {s.path for s in fused.ranked} == {"a.py", "b.py"}


def test_rrf_is_deterministic_on_ties():
    a = _result("bm25", ["b.py", "a.py"])
    b = _result("dense", ["a.py", "b.py"])
    runs = [[s.path for s in rrf_fuse([a, b]).ranked] for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]
    assert runs[0] == ["a.py", "b.py"]  # exact tie broken by path


def test_rrf_empty_input():
    assert rrf_fuse([]).ranked == []


# --- paired significance -----------------------------------------------------


def test_mcnemar_ignores_agreement():
    """Examples both methods get right (or both wrong) carry no information.

    This is the whole point of pairing: adding 1,000 examples that both methods
    handle identically must not change the verdict.
    """
    a = [True, True, False, False]
    b = [True, False, True, False]
    base = mcnemar(a, b)
    padded = mcnemar([*a, *[True] * 1000], [*b, *[True] * 1000])
    assert base["a_only"] == padded["a_only"] == 1
    assert base["b_only"] == padded["b_only"] == 1
    assert base["z"] == padded["z"] == 0.0
    # The *delta* does shrink, since it is a difference of accuracies over n.
    assert abs(padded["delta"]) < abs(base["delta"]) or base["delta"] == 0


def test_mcnemar_direction_and_magnitude():
    """A wins 9 disagreements, B wins 1, over 100 examples."""
    a = [True] * 9 + [False] * 1 + [True] * 90
    b = [False] * 9 + [True] * 1 + [True] * 90
    s = mcnemar(a, b)
    assert s["a_only"] == 9 and s["b_only"] == 1
    assert s["delta"] == pytest.approx(0.08)
    assert s["se"] == pytest.approx(math.sqrt(10) / 100)
    assert s["z"] == pytest.approx(8 / math.sqrt(10))
    assert s["z"] > 1.96


def test_mcnemar_paired_se_beats_unpaired_when_methods_agree():
    """The reason to pair: agreement shrinks the error bar on the difference."""
    a = [True] * 60 + [False] * 40
    b = [True] * 58 + [False] * 2 + [False] * 40
    paired = mcnemar(a, b)["se"]
    assert paired < unpaired_se(0.60, 100)


def test_mcnemar_identical_methods():
    s = mcnemar([True, False, True], [True, False, True])
    assert s["delta"] == 0.0 and s["z"] == 0.0 and s["se"] == 0.0


def test_mcnemar_empty_and_mismatched():
    assert mcnemar([], [])["z"] == 0.0
    with pytest.raises(ValueError):
        mcnemar([True], [True, False])


def test_unpaired_se_shrinks_with_n():
    assert unpaired_se(0.5, 100) == pytest.approx(0.05)
    assert unpaired_se(0.5, 2235) == pytest.approx(0.0106, abs=1e-4)
    assert unpaired_se(0.5, 0) == 0.0
