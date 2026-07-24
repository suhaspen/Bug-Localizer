"""End-to-end mining against a real git repository built in a temp dir.

The filter unit tests use fabricated `CommitInfo` objects; this file builds an
actual repo with `git`, so it also covers the parts that could silently break
without any filter being wrong: the `git log` output parser, the parent-commit
existence check, and query construction.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from buglocalizer.config import Config, MiningConfig, RepoConfig
from buglocalizer.dataset import DEV, HELDOUT, assign_temporal_split
from buglocalizer.mining.miner import (
    build_query_text,
    extract_issue_refs,
    iter_commits,
    mine_repo,
    paths_present_at,
)

MINING = MiningConfig(
    max_files_per_commit=5,
    exclude_path_globs=["docs/**", "**/tests/**", "**/test_*.py", "*.md"],
    source_extensions=[".py"],
    fix_patterns=[r"(?i)^BUG[:\s]", r"(?i)\bfix(e[sd])?\b", r"(?i)fixes? #\d+"],
    exclude_message_patterns=[r"(?i)^DOC[:\s]", r"(?i)\btypos?\b"],
)


def git(path: Path, *args: str, **kwargs) -> str:
    env = {
        "GIT_AUTHOR_NAME": "T",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "T",
        "GIT_COMMITTER_EMAIL": "t@example.com",
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        **kwargs.pop("env", {}),
    }
    return subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True,
        text=True,
        check=True,
        env=env,
        **kwargs,
    ).stdout


def write(path: Path, rel: str, content: str) -> None:
    target = path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)


def commit_all(path: Path, message: str, when: str) -> None:
    git(path, "add", "-A")
    git(path, "commit", "-m", message, env={"GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when})


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    """A tiny repo containing one of each commit shape the filters care about."""
    repo = tmp_path / "toy"
    repo.mkdir()
    git(repo, "init", "-b", "main")

    # 1. root commit — no parent, so it can never be an example
    write(repo, "toy/core.py", "def parse(s):\n    return s.split(',')\n")
    write(repo, "README.md", "# toy\n")
    commit_all(repo, "Initial commit", "2020-01-01T00:00:00+00:00")

    # 2. a feature commit — not a fix
    write(repo, "toy/api.py", "def get():\n    return 1\n")
    commit_all(repo, "Add the api module", "2020-02-01T00:00:00+00:00")

    # 3. a clean bug fix touching source + its test  → EXAMPLE (gold = core.py)
    write(repo, "toy/core.py", "def parse(s):\n    return [x.strip() for x in s.split(',')]\n")
    write(repo, "tests/test_core.py", "def test_parse():\n    assert True\n")
    commit_all(
        repo, "BUG: parse leaves whitespace on tokens\n\nFixes #12", "2020-03-01T00:00:00+00:00"
    )

    # 4. docs-only fix — nothing to localize to
    write(repo, "docs/guide.md", "fixed the wording\n")
    commit_all(repo, "Fix the guide wording", "2020-04-01T00:00:00+00:00")

    # 5. a DOC:-prefixed commit that edits source — excluded by message
    write(repo, "toy/api.py", "def get():\n    '''Return one.'''\n    return 1\n")
    commit_all(repo, "DOC: fix the get() docstring", "2020-05-01T00:00:00+00:00")

    # 6. a mega commit (6 files > max 5)
    for i in range(6):
        write(repo, f"toy/mod{i}.py", f"X = {i}\n")
    commit_all(repo, "Fix everything everywhere", "2020-06-01T00:00:00+00:00")

    # 7. a fix that CREATES its gold file — unreachable from the parent state
    write(repo, "toy/brand_new.py", "def helper():\n    return 2\n")
    commit_all(repo, "BUG: add missing helper", "2020-07-01T00:00:00+00:00")

    # 8. a fix whose message names the file verbatim → leakage to scrub
    write(repo, "toy/api.py", "def get():\n    return 2\n")
    commit_all(repo, "BUG: wrong return value in toy/api.py", "2020-08-01T00:00:00+00:00")

    return repo


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(mining=MINING, split={"strategy": "temporal", "dev_fraction": 0.5})


def test_mine_fixture_repo_yields_expected_examples(fixture_repo, cfg):
    examples, funnel = mine_repo(RepoConfig(name="toy", url="local"), cfg, fixture_repo)

    by_subject = {e.query_text.split("\n")[0]: e for e in examples}

    # Exactly the two localizable fixes survive: the clean one and the scrubbed one.
    assert len(examples) == 2, [e.query_text.split("\n")[0] for e in examples]

    clean = by_subject["BUG: parse leaves whitespace on tokens"]
    assert clean.gold_files == ["toy/core.py"]
    assert clean.issue_refs == [12]
    assert clean.n_files_changed == 2  # core.py + its test
    assert clean.parent_sha != clean.fix_sha

    # Every rejection is attributed, and the funnel accounts for every commit.
    assert funnel["no_parent"] == 1
    assert funnel["not_a_fix"] == 1
    assert funnel["excluded_message"] == 1
    assert funnel["mega_commit"] == 1
    assert funnel["no_source_files"] == 1
    assert funnel["gold_missing_at_parent"] == 1
    assert funnel["kept"] == 2
    assert (
        sum(funnel[k] for k in funnel if k not in {"scanned", "kept"}) + funnel["kept"]
        == (funnel["scanned"])
    )


def test_gold_file_created_by_the_fix_is_dropped(fixture_repo, cfg):
    """A file the fix created does not exist at the parent, so it is unretrievable."""
    examples, _ = mine_repo(RepoConfig(name="toy", url="local"), cfg, fixture_repo)
    assert all("brand_new.py" not in f for e in examples for f in e.gold_files)


def test_query_scrubbing_removes_the_gold_path(fixture_repo, cfg):
    examples, _ = mine_repo(RepoConfig(name="toy", url="local"), cfg, fixture_repo)
    leaky = next(e for e in examples if e.query_scrubbed)
    assert "toy/api.py" not in leaky.query_text
    # The rest of the description must survive the scrub.
    assert "wrong return value" in leaky.query_text


def test_parent_sha_really_is_the_buggy_state(fixture_repo, cfg):
    """The single most important correctness property in the project."""
    examples, _ = mine_repo(RepoConfig(name="toy", url="local"), cfg, fixture_repo)
    clean = next(e for e in examples if e.gold_files == ["toy/core.py"])

    at_parent = git(fixture_repo, "show", f"{clean.parent_sha}:toy/core.py")
    at_fix = git(fixture_repo, "show", f"{clean.fix_sha}:toy/core.py")

    assert "strip()" not in at_parent, "parent must still contain the bug"
    assert "strip()" in at_fix, "the fix must not be what we index"


# --- parsing helpers ---------------------------------------------------------


def test_iter_commits_parses_multiline_messages(fixture_repo):
    commits = iter_commits(fixture_repo, "main")
    assert len(commits) == 8
    multiline = next(c for c in commits if c.message.startswith("BUG: parse leaves"))
    assert "Fixes #12" in multiline.message
    assert set(multiline.changed_paths) == {"toy/core.py", "tests/test_core.py"}
    # Newest first, as `git log` emits them.
    assert commits[0].authored_at > commits[-1].authored_at


def test_paths_present_at_batches_correctly(fixture_repo):
    commits = iter_commits(fixture_repo, "main")
    head = commits[0].sha
    result = paths_present_at(fixture_repo, [(head, "toy/core.py"), (head, "nope/missing.py")])
    assert result == [True, False]


def test_paths_present_at_empty():
    assert paths_present_at(Path("."), []) == []


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("BUG: thing\n\nCo-authored-by: X <x@y.z>", "BUG: thing"),
        ("BUG: thing\n\nSigned-off-by: X <x@y.z>", "BUG: thing"),
        ("BUG: thing\n\n(cherry picked from commit abc1234)", "BUG: thing"),
    ],
)
def test_build_query_text_strips_trailers(message, expected):
    text, _ = build_query_text(message, (), scrub=False)
    assert text == expected


def test_build_query_text_strips_pandas_squash_trailer():
    """Old pandas PRs append the review history; it is noise, not bug description."""
    message = (
        "BUG: reindex_like after shape comparison\n\n"
        "the former code reindexed before comparing shapes.\n\n"
        "Author: jojomdt <z@example.com>\n\n"
        "Closes #15496 from jojomdt/master and squashes the following commits:\n\n"
        "7b3437b  fix test_frame_equal_message error\n"
        "0340b5c  change check_like description\n"
    )
    text, _ = build_query_text(message, (), scrub=False)
    assert "squashes the following commits" not in text
    assert "7b3437b" not in text
    assert "Author:" not in text
    # The actual bug description survives.
    assert "reindexed before comparing shapes" in text
    assert text.startswith("BUG: reindex_like after shape comparison")


def test_build_query_text_does_not_scrub_bare_module_names():
    """Scrubbing `groupby` here would delete the bug description itself."""
    text, scrubbed = build_query_text(
        "BUG: groupby.apply raises on empty frames", ("pandas/core/groupby/groupby.py",), scrub=True
    )
    assert "groupby.apply" in text
    assert scrubbed is False


def test_extract_issue_refs():
    assert extract_issue_refs("BUG: x\n\nCloses #123, also #456 and #123") == [123, 456]
    assert extract_issue_refs("no refs here") == []


# --- split -------------------------------------------------------------------


def test_temporal_split_is_per_repo_and_ordered(fixture_repo, cfg):
    from datetime import UTC, datetime

    from buglocalizer.dataset import Example

    def ex(repo: str, day: int) -> Example:
        return Example(
            example_id=f"{repo}@{day}",
            repo=repo,
            fix_sha=f"{day:040d}",
            parent_sha="0" * 40,
            authored_at=datetime(2020, 1, day, tzinfo=UTC),
            query_text="q",
            gold_files=["a.py"],
            n_files_changed=1,
        )

    # 10 examples for repo A, 4 for repo B, interleaved in time.
    examples = [ex("a", d) for d in range(1, 11)] + [ex("b", d) for d in range(1, 5)]
    out = assign_temporal_split(examples, dev_fraction=0.7)

    a = sorted([e for e in out if e.repo == "a"], key=lambda e: e.authored_at)
    b = sorted([e for e in out if e.repo == "b"], key=lambda e: e.authored_at)

    # Each repo is split independently — a global cutoff would leave b empty.
    assert [e.split for e in a] == [DEV] * 7 + [HELDOUT] * 3
    assert [e.split for e in b] == [DEV] * 2 + [HELDOUT] * 2

    # Held-out is strictly newer than dev within a repo.
    for group in (a, b):
        newest_dev = max(e.authored_at for e in group if e.split == DEV)
        oldest_heldout = min(e.authored_at for e in group if e.split == HELDOUT)
        assert newest_dev < oldest_heldout


def test_split_is_deterministic(fixture_repo, cfg):
    examples, _ = mine_repo(RepoConfig(name="toy", url="local"), cfg, fixture_repo)
    first = [e.split for e in assign_temporal_split(examples, 0.5)]
    second = [e.split for e in assign_temporal_split(examples, 0.5)]
    assert first == second
