"""Which commits become labeled examples, and which get thrown away.

This module is deliberately pure: it takes a `CommitInfo` dataclass and returns
a decision. Nothing here touches git, the filesystem, or the network, which is
what makes the filter rules directly testable against fabricated commits.

The filters matter more than they look. Every one of them exists because letting
the commits through would *inflate* our reported accuracy for a reason unrelated
to retrieval quality. See docs/01_dataset.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache

from buglocalizer.config import MiningConfig

# --- rejection reasons -------------------------------------------------------
# Ordered as the funnel applies them; `mine` logs a count per reason.
NO_PARENT = "no_parent"
MERGE_COMMIT = "merge_commit"
EXCLUDED_MESSAGE = "excluded_message"
NOT_A_FIX = "not_a_fix"
MEGA_COMMIT = "mega_commit"
NO_SOURCE_FILES = "no_source_files"
GOLD_MISSING_AT_PARENT = "gold_missing_at_parent"

FUNNEL_ORDER = [
    NO_PARENT,
    MERGE_COMMIT,
    EXCLUDED_MESSAGE,
    NOT_A_FIX,
    MEGA_COMMIT,
    NO_SOURCE_FILES,
    GOLD_MISSING_AT_PARENT,
]

# --- borderline markers ------------------------------------------------------
# Not rejections. These flag examples that *nearly* got dropped, so that reviewing
# a sample of them shows what the filter boundaries actually let through.
AT_MEGA_THRESHOLD = "at_mega_commit_threshold"
WEAK_FIX_SIGNAL = "weak_fix_signal"
LONE_SOURCE_FILE = "lone_source_file_among_many"
SHORT_QUERY = "short_query"
MANY_GOLD_FILES = "many_gold_files"
PARTIAL_GOLD_AT_PARENT = "some_gold_files_absent_at_parent"

SHORT_QUERY_CHARS = 60
MANY_GOLD_THRESHOLD = 5


@dataclass(frozen=True)
class CommitInfo:
    """Everything the filters need to know about one commit."""

    sha: str
    parent_shas: tuple[str, ...]
    authored_at: datetime
    message: str
    changed_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        # Normalize here rather than at the call site. Real commit messages
        # sometimes carry leading whitespace -- pandas has several written
        # `" DOC: ..."` -- which makes every `^`-anchored exclusion pattern
        # silently miss, letting docs commits through as bug fixes. Stripping in
        # the dataclass means no construction path can skip it.
        object.__setattr__(self, "message", self.message.strip())

    @property
    def subject(self) -> str:
        return self.message.split("\n", 1)[0]


@dataclass
class Decision:
    kept: bool
    reason: str | None = None
    gold_files: tuple[str, ...] = ()
    borderline: list[str] = field(default_factory=list)


# --- glob matching -----------------------------------------------------------


def _translate(pattern: str) -> str:
    """Convert a glob to a regex, with `**` crossing directory boundaries.

    We can't use `fnmatch` because its `*` happily matches `/`, which would make
    `**/test_*.py` and `*.py` mean the same thing. We can't use
    `PurePath.full_match` because it is Python 3.13+.
    """
    out: list[str] = []
    i, n = 0, len(pattern)
    while i < n:
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return "".join(out)


@lru_cache(maxsize=512)
def _compiled(pattern: str) -> re.Pattern[str]:
    return re.compile(_translate(pattern) + r"\Z")


def path_matches(path: str, pattern: str) -> bool:
    """Match a repo-relative path against one glob.

    Follows gitignore's convention that a pattern containing no `/` is matched
    against the basename, so `*.md` catches `docs/notes.md`.
    """
    rx = _compiled(pattern)
    if rx.match(path):
        return True
    return "/" not in pattern and bool(rx.match(path.rsplit("/", 1)[-1]))


def is_excluded_path(path: str, globs: list[str]) -> bool:
    return any(path_matches(path, g) for g in globs)


def is_source_path(path: str, extensions: list[str]) -> bool:
    return any(path.endswith(ext) for ext in extensions)


def compute_gold_files(changed_paths: tuple[str, ...], cfg: MiningConfig) -> tuple[str, ...]:
    """The subset of a commit's changed files that could plausibly hold the bug.

    Test files are excluded even though a fix often edits them: the test is
    evidence of the bug, not its location. If we counted `tests/test_send.py` as
    a correct answer, a retriever could score well by matching the bug report's
    vocabulary against test names and never find the actual defect.
    """
    return tuple(
        p
        for p in changed_paths
        if is_source_path(p, cfg.source_extensions)
        and not is_excluded_path(p, cfg.exclude_path_globs)
    )


# --- message matching --------------------------------------------------------


def matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text) for p in patterns)


# --- the funnel --------------------------------------------------------------


def classify(commit: CommitInfo, cfg: MiningConfig) -> Decision:
    """Decide whether one commit becomes a labeled example.

    Rejection order is meaningful: cheap structural checks run before message
    regexes, and the funnel counts attribute each dropped commit to the *first*
    rule that rejected it, so the logged numbers sum to the total scanned.
    """
    if not commit.parent_shas:
        # The root commit has no buggy state to index against.
        return Decision(kept=False, reason=NO_PARENT)

    if cfg.skip_merge_commits and len(commit.parent_shas) > 1:
        # A merge's file list is an artifact of merging, not of fixing. The real
        # fix is an individual commit somewhere in the branch, which we see
        # separately.
        return Decision(kept=False, reason=MERGE_COMMIT)

    if matches_any(commit.message, cfg.exclude_message_patterns):
        return Decision(kept=False, reason=EXCLUDED_MESSAGE)

    if not matches_any(commit.message, cfg.fix_patterns):
        return Decision(kept=False, reason=NOT_A_FIX)

    if len(commit.changed_paths) > cfg.max_files_per_commit:
        # A 200-file commit is a refactor or a release. Calling all 200 files
        # "the bug" would make top-10 accuracy climb for free.
        return Decision(kept=False, reason=MEGA_COMMIT)

    gold = compute_gold_files(commit.changed_paths, cfg)
    if len(gold) < cfg.min_gold_files:
        # Docs-only or test-only commit: nothing to localize to.
        return Decision(kept=False, reason=NO_SOURCE_FILES)

    return Decision(kept=True, gold_files=gold, borderline=_borderline_flags(commit, gold, cfg))


def _borderline_flags(commit: CommitInfo, gold: tuple[str, ...], cfg: MiningConfig) -> list[str]:
    """Mark examples that only just survived, so a reviewer can inspect the edges."""
    flags: list[str] = []

    if len(commit.changed_paths) == cfg.max_files_per_commit:
        flags.append(AT_MEGA_THRESHOLD)

    # The fix keyword appears only in the body, never the subject line — a much
    # weaker signal than a `BUG:` prefix or a `fixes #123` in the summary.
    if not matches_any(commit.subject, cfg.fix_patterns):
        flags.append(WEAK_FIX_SIGNAL)

    if len(gold) == 1 and len(commit.changed_paths) >= 4:
        flags.append(LONE_SOURCE_FILE)

    if len(commit.message.strip()) < SHORT_QUERY_CHARS:
        flags.append(SHORT_QUERY)

    if len(gold) >= MANY_GOLD_THRESHOLD:
        flags.append(MANY_GOLD_FILES)

    return flags
