"""Tests for score accumulation, composition reporting, and the peek ledger.

These cover the layer between the metrics (tested in test_metrics.py) and the
printed table — where an averaging or grouping mistake would produce a number
that looks entirely plausible.
"""

from __future__ import annotations

import json

import pytest

from buglocalizer.config import Config
from buglocalizer.eval.harness import EvalRun, MethodScores
from buglocalizer.eval.results import (
    peek_count,
    record_heldout_peek,
    render_markdown,
    run_to_dict,
)


def test_method_scores_accumulates_top_k():
    s = MethodScores()
    s.add(["a.py", "x.py"], ["a.py"], 0.1)  # gold at rank 1
    s.add(["x.py", "a.py"], ["a.py"], 0.1)  # gold at rank 2
    s.add(["x.py", "y.py"], ["a.py"], 0.1)  # not found

    assert s.n == 3
    assert s.top_k(1) == pytest.approx(1 / 3)
    assert s.top_k(5) == pytest.approx(2 / 3)
    # MRR = (1 + 0.5 + 0) / 3
    assert s.mrr == pytest.approx(0.5)


def test_method_scores_empty_is_zero_not_a_crash():
    s = MethodScores()
    assert s.top_k(10) == 0.0 and s.mrr == 0.0 and s.map == 0.0
    assert s.as_dict()["n"] == 0


def test_method_scores_dict_shape():
    s = MethodScores()
    s.add(["a.py"], ["a.py"], 0.25)
    d = s.as_dict()
    assert set(d) == {"n", "top1", "top5", "top10", "mrr", "map", "mean_seconds"}
    assert d["top1"] == 1.0
    assert d["mean_seconds"] == 0.25


def test_composition_shares_sum_to_one():
    run = EvalRun(include_tests=False, n_examples=10)
    run.repo_counts = {"pandas": 7, "flask": 2, "requests": 1}
    comp = run.composition()
    assert [r for r, _, _ in comp] == ["flask", "pandas", "requests"]  # sorted
    assert sum(share for _, _, share in comp) == pytest.approx(1.0)
    assert dict((r, share) for r, _, share in comp)["pandas"] == pytest.approx(0.7)


def test_composition_handles_empty_run():
    assert EvalRun(include_tests=False, n_examples=0).composition() == []


def _payload(cfg: Config, include_tests: bool, counts: dict[str, int]) -> dict:
    run = EvalRun(include_tests=include_tests, n_examples=sum(counts.values()))
    run.repo_counts = counts
    for repo in counts:
        scores = MethodScores()
        scores.add(["a.py"], ["a.py"], 0.1)
        run.per_repo[repo] = {"bm25": scores, "dense": scores, "hybrid": scores}
    for method in ("bm25", "dense", "hybrid"):
        s = MethodScores()
        s.add(["a.py"], ["a.py"], 0.1)
        run.overall[method] = s
    return run_to_dict(cfg, run, "heldout", "test")


def test_run_to_dict_records_the_reproducibility_fields():
    """A published number is only defensible if the settings behind it are stored."""
    payload = _payload(Config(), False, {"flask": 2})
    for key in ("timestamp", "git_sha", "split", "config", "composition", "overall", "per_repo"):
        assert key in payload
    for key in ("include_tests", "chunk_max_chars", "embedding_model", "rrf_k", "seed"):
        assert key in payload["config"]


def test_render_markdown_includes_composition_and_both_scopes():
    cfg = Config()
    payloads = [
        _payload(cfg, False, {"pandas": 300, "flask": 85}),
        _payload(cfg, True, {"pandas": 300, "flask": 85}),
    ]
    md = render_markdown(payloads)
    assert "tests excluded" in md
    assert "tests INCLUDED" in md
    assert "77.9%" in md  # 300 / 385, so the pandas share is stated outright
    assert "Scope comparison" in md


def test_peek_ledger_counts_every_heldout_run(tmp_path):
    cfg = Config()
    cfg.paths.results_dir = tmp_path
    assert peek_count(cfg) == 0

    for i in range(3):
        payload = _payload(cfg, False, {"flask": 1})
        n = record_heldout_peek(cfg, payload, tmp_path / f"{i}.json")
        assert n == i + 1

    assert peek_count(cfg) == 3
    lines = (tmp_path / "heldout_log.jsonl").read_text().strip().split("\n")
    assert len(lines) == 3
    entry = json.loads(lines[0])
    assert {"timestamp", "git_sha", "include_tests", "n_examples", "result_file"} <= set(entry)
