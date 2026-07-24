"""Tests for the commit filters.

These are the highest-value tests in Milestone 1. A wrong filter does not crash
— it silently changes what the dataset contains, and every accuracy number
downstream inherits the error with no visible symptom.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from buglocalizer.config import MiningConfig
from buglocalizer.mining.filters import (
    AT_MEGA_THRESHOLD,
    EXCLUDED_MESSAGE,
    LONE_SOURCE_FILE,
    MANY_GOLD_FILES,
    MEGA_COMMIT,
    MERGE_COMMIT,
    NO_PARENT,
    NO_SOURCE_FILES,
    NOT_A_FIX,
    SHORT_QUERY,
    WEAK_FIX_SIGNAL,
    CommitInfo,
    classify,
    compute_gold_files,
    path_matches,
)

CFG = MiningConfig(
    max_files_per_commit=10,
    exclude_path_globs=[
        "docs/**",
        "doc/**",
        "**/tests/**",
        "**/test_*.py",
        "**/conftest.py",
        "*.md",
        "*.rst",
    ],
    source_extensions=[".py"],
    # Mirrors the shipped config.yaml — a fixture that drifts from the real
    # patterns tests the wrong thing.
    fix_patterns=[r"(?i)^BUG[:\s]", r"(?i)\bfix(e[sd])?\b", r"(?i)\bbug\b", r"(?i)fixes? #\d+"],
    exclude_message_patterns=[r"(?i)^(DOC|TST|CLN)[:\s]", r"(?i)^revert\b", r"(?i)\btypos?\b"],
)


def commit(**kwargs) -> CommitInfo:
    defaults = dict(
        sha="a" * 40,
        parent_shas=("b" * 40,),
        authored_at=datetime(2023, 1, 1, tzinfo=UTC),
        message="BUG: send_file returns the wrong mimetype for extensionless names",
        changed_paths=("src/flask/helpers.py",),
    )
    return CommitInfo(**{**defaults, **kwargs})


# --- glob matching -----------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "pattern", "expected"),
    [
        ("docs/index.rst", "docs/**", True),
        ("docs/a/b/c.py", "docs/**", True),
        ("pandas/docs.py", "docs/**", False),
        # `**/` must be optional, so the pattern matches at the repo root too.
        ("tests/test_a.py", "**/tests/**", True),
        ("pandas/tests/frame/test_a.py", "**/tests/**", True),
        ("pandas/core/frame.py", "**/tests/**", False),
        ("pandas/tests/test_x.py", "**/test_*.py", True),
        ("test_x.py", "**/test_*.py", True),
        # A slash-free pattern matches the basename, gitignore-style.
        ("README.md", "*.md", True),
        ("docs/notes.md", "*.md", True),
        # `*` must not cross a directory boundary, or `*.py` would match everything.
        ("a/b.py", "*.py", True),
        ("src/flask/helpers.py", "src/*.py", False),
    ],
)
def test_path_matches(path, pattern, expected):
    assert path_matches(path, pattern) is expected


def test_compute_gold_files_keeps_only_non_test_sources():
    changed = (
        "src/flask/helpers.py",
        "tests/test_helpers.py",
        "docs/api.rst",
        "CHANGES.md",
        "setup.py",
    )
    # setup.py is a .py outside any excluded path, so it legitimately survives.
    assert compute_gold_files(changed, CFG) == ("src/flask/helpers.py", "setup.py")


def test_test_files_are_never_gold():
    """A fix usually edits its test too, but the test is evidence, not location."""
    gold = compute_gold_files(("pandas/core/frame.py", "pandas/tests/test_frame.py"), CFG)
    assert gold == ("pandas/core/frame.py",)


# --- the funnel --------------------------------------------------------------


def test_keeps_a_clean_bug_fix():
    decision = classify(commit(), CFG)
    assert decision.kept
    assert decision.gold_files == ("src/flask/helpers.py",)


def test_root_commit_rejected():
    assert classify(commit(parent_shas=()), CFG).reason == NO_PARENT


def test_merge_commit_rejected():
    d = classify(commit(parent_shas=("b" * 40, "c" * 40)), CFG)
    assert d.reason == MERGE_COMMIT


def test_merge_commit_kept_when_disabled():
    cfg = CFG.model_copy(update={"skip_merge_commits": False})
    assert classify(commit(parent_shas=("b" * 40, "c" * 40)), cfg).kept


def test_non_fix_commit_rejected():
    assert classify(commit(message="Add support for async views"), CFG).reason == NOT_A_FIX


def test_doc_prefix_rejected_before_fix_pattern_matches():
    """`DOC: fix wording in docstring` touches source but is not a bug fix.

    This is the filter that matters most for pandas: without it, docstring
    edits become labeled examples pointing at files that were never at fault.
    """
    d = classify(
        commit(message="DOC: fix wording in DataFrame.merge docstring"),
        CFG,
    )
    assert d.reason == EXCLUDED_MESSAGE


def test_lowercase_and_plural_doc_prefixes_are_excluded():
    """pandas drifted to conventional-commit style; `docs:` must not slip through."""
    cfg = CFG.model_copy(
        update={"exclude_message_patterns": [r"(?i)^(DOCS?|STY|STYLE|MAINT|ASV|PEP)[:\s]"]}
    )
    for subject in ("docs: include delete_rows option", "DOC: fixed wording", "style: fix spacing"):
        assert classify(commit(message=subject), cfg).reason == EXCLUDED_MESSAGE, subject


def test_leading_whitespace_does_not_defeat_anchored_patterns():
    """pandas has commits written `" DOC: ..."`.

    A leading space makes every `^`-anchored exclusion miss, which is how five
    docstring commits got labeled as bug fixes on the first mining run.
    """
    cfg = CFG.model_copy(update={"exclude_message_patterns": [r"(?i)^(DOCS?|STY)[:\s]"]})
    assert classify(commit(message=" DOC: fix some sphinx syntax warnings"), cfg).reason == (
        EXCLUDED_MESSAGE
    )
    assert classify(commit(message="\n\tDOC: fix wording"), cfg).reason == EXCLUDED_MESSAGE


def test_subject_is_the_first_line_after_stripping():
    assert commit(message="  BUG: thing\n\nbody").subject == "BUG: thing"


def test_combined_doc_bug_prefix_survives():
    """`DOC/BUG:` really does fix a bug — the terminator must be `[:\\s]`, not `[:/]`."""
    cfg = CFG.model_copy(update={"exclude_message_patterns": [r"(?i)^(DOCS?|STY)[:\s]"]})
    assert classify(commit(message="DOC/BUG: wrong result in merge"), cfg).kept


def test_build_and_packaging_files_are_not_gold():
    """setup.py was the sole label for 108 examples before this exclusion."""
    cfg = CFG.model_copy(
        update={
            "exclude_path_globs": [
                *CFG.exclude_path_globs,
                "setup.py",
                "asv_bench/**",
                "scripts/**",
            ]
        }
    )
    assert compute_gold_files(("pandas/core/frame.py", "setup.py"), cfg) == (
        "pandas/core/frame.py",
    )
    assert compute_gold_files(("asv_bench/benchmarks/groupby.py",), cfg) == ()
    # A commit touching only packaging has nothing localizable left.
    assert classify(commit(changed_paths=("setup.py",)), cfg).reason == NO_SOURCE_FILES


def test_revert_and_typo_commits_rejected():
    assert classify(commit(message="Revert 'BUG: fix the thing'"), CFG).reason == EXCLUDED_MESSAGE
    assert classify(commit(message="Fix typo in comment"), CFG).reason == EXCLUDED_MESSAGE


def test_mega_commit_rejected():
    paths = tuple(f"src/flask/mod{i}.py" for i in range(11))
    assert classify(commit(changed_paths=paths), CFG).reason == MEGA_COMMIT


def test_mega_commit_boundary_is_inclusive():
    """Exactly at the threshold is kept; one more is dropped."""
    at = tuple(f"src/flask/mod{i}.py" for i in range(10))
    over = tuple(f"src/flask/mod{i}.py" for i in range(11))
    assert classify(commit(changed_paths=at), CFG).kept
    assert not classify(commit(changed_paths=over), CFG).kept


def test_docs_only_commit_rejected():
    d = classify(commit(changed_paths=("docs/a.rst", "README.md")), CFG)
    assert d.reason == NO_SOURCE_FILES


def test_test_only_commit_rejected():
    d = classify(commit(changed_paths=("tests/test_helpers.py",)), CFG)
    assert d.reason == NO_SOURCE_FILES


# --- borderline markers ------------------------------------------------------


def test_borderline_at_mega_threshold():
    paths = tuple(f"src/flask/mod{i}.py" for i in range(10))
    assert AT_MEGA_THRESHOLD in classify(commit(changed_paths=paths), CFG).borderline


def test_borderline_weak_fix_signal_when_keyword_only_in_body():
    msg = "Improve header handling\n\nThis also fixes the casing problem."
    d = classify(commit(message=msg), CFG)
    assert d.kept
    assert WEAK_FIX_SIGNAL in d.borderline


def test_strong_signal_in_subject_is_not_flagged_weak():
    d = classify(commit(message="BUG: mimetype wrong\n\nlong explanation here"), CFG)
    assert WEAK_FIX_SIGNAL not in d.borderline


def test_borderline_lone_source_file_among_many():
    changed = (
        "src/flask/helpers.py",
        "tests/test_a.py",
        "tests/test_b.py",
        "docs/x.rst",
        "CHANGES.md",
    )
    assert LONE_SOURCE_FILE in classify(commit(changed_paths=changed), CFG).borderline


def test_borderline_short_query():
    assert SHORT_QUERY in classify(commit(message="fix bug"), CFG).borderline


def test_borderline_many_gold_files():
    paths = tuple(f"src/flask/mod{i}.py" for i in range(6))
    assert MANY_GOLD_FILES in classify(commit(changed_paths=paths), CFG).borderline


def test_clean_example_has_no_borderline_flags():
    assert classify(commit(), CFG).borderline == []
