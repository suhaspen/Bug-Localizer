"""Walk a repository's history and turn its bug fixes into labeled examples.

Reading strategy: one `git log` subprocess for the entire history, rather than
GitPython's per-commit diff API. GitPython's `commit.stats` shells out once per
commit, which is roughly 40,000 subprocesses on pandas and takes tens of
minutes; a single `git log --name-only` does the same work in seconds. GitPython
is still used for cloning, where its API is genuinely nicer and the performance
does not matter.
"""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from buglocalizer.config import Config, RepoConfig
from buglocalizer.dataset import Example
from buglocalizer.logging_setup import get_logger
from buglocalizer.mining.filters import (
    FUNNEL_ORDER,
    GOLD_MISSING_AT_PARENT,
    PARTIAL_GOLD_AT_PARENT,
    CommitInfo,
    classify,
)

log = get_logger(__name__)

# Field/record separators chosen from the ASCII control block because they
# cannot occur in a commit message or a file path.
_REC = "\x01"
_FIELD = "\x1f"
_END = "\x1e"

# `%at` (unix epoch seconds), not `%aI` (ISO-8601). Real history contains commits
# with malformed timezone offsets -- flask has one written `+518:00` -- which
# `datetime.fromisoformat` rejects outright. The epoch form is always valid, and
# an absolute instant is what temporal ordering actually needs anyway.
_LOG_FORMAT = f"{_REC}%H{_FIELD}%P{_FIELD}%at{_FIELD}%B{_END}"

_ISSUE_RE = re.compile(r"#(\d{1,7})\b")
_TRAILER_RE = re.compile(
    r"(?im)^(co-authored-by|signed-off-by|reviewed-by|acked-by|tested-by|cc)\s*:.*$"
)
_CHERRY_PICK_RE = re.compile(r"(?im)^\(cherry picked from commit [0-9a-f]{7,40}\)\s*$")
# Old pandas merged PRs with a squash trailer that appends the author line and
# every squashed sub-commit subject with its SHA. That block is pure noise: it
# describes the *review history*, not the bug, and it drags in unrelated
# vocabulary plus a dozen hex SHAs. Affects ~360 examples.
_SQUASH_TRAILER_RE = re.compile(
    r"(?ims)^Closes #\d+ from \S+ and squashes the following commits:.*\Z"
)
_AUTHOR_LINE_RE = re.compile(r"(?im)^Author:\s.*$")


def _run_git(repo_path: Path, args: list[str], stdin: str | None = None) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def resolve_ref(repo_path: Path, preferred: str) -> str:
    """Use the configured branch if it exists, else fall back to HEAD."""
    try:
        _run_git(repo_path, ["rev-parse", "--verify", f"{preferred}^{{commit}}"])
        return preferred
    except subprocess.CalledProcessError:
        log.warning("branch %r not found in %s; falling back to HEAD", preferred, repo_path.name)
        return "HEAD"


def iter_commits(repo_path: Path, ref: str) -> list[CommitInfo]:
    """Read the whole history of `ref` in one subprocess."""
    raw = _run_git(
        repo_path,
        ["log", ref, "--name-only", f"--pretty=format:{_LOG_FORMAT}"],
    )

    commits: list[CommitInfo] = []
    for chunk in raw.split(_REC):
        if not chunk.strip():
            continue
        header, _, files_blob = chunk.partition(_END)
        parts = header.split(_FIELD)
        if len(parts) != 4:
            continue
        sha, parents, date, message = parts
        commits.append(
            CommitInfo(
                sha=sha,
                parent_shas=tuple(parents.split()),
                authored_at=datetime.fromtimestamp(int(date), tz=UTC),
                message=message,
                changed_paths=tuple(p for p in files_blob.splitlines() if p.strip()),
            )
        )
    return commits


def paths_present_at(repo_path: Path, pairs: list[tuple[str, str]]) -> list[bool]:
    """Check `(commit_sha, path)` existence in one batch.

    Guards the rule that a gold file must exist in the corpus we index. A fix
    that *creates* a file leaves a gold label that no retriever could ever
    return from the buggy state, which would put a silent ceiling on every
    accuracy number we report.

    `git cat-file --batch-check` emits exactly one line per input, in order, so
    results map back positionally.
    """
    if not pairs:
        return []
    stdin = "".join(f"{sha}:{path}\n" for sha, path in pairs)
    out = _run_git(repo_path, ["cat-file", "--batch-check"], stdin=stdin)
    lines = out.splitlines()
    if len(lines) != len(pairs):
        raise RuntimeError(f"cat-file returned {len(lines)} lines for {len(pairs)} inputs")
    return [not line.endswith(("missing", "ambiguous")) for line in lines]


def build_query_text(message: str, gold_files: tuple[str, ...], scrub: bool) -> tuple[str, bool]:
    """Turn a commit message into the query a retriever will see.

    Two kinds of thing are stripped. Author trailers, cherry-pick markers and
    squash blocks are noise that would dilute the term statistics. More
    importantly, literal gold file paths are removed when `scrub` is on: a
    message reading "fix mimetype in
    src/flask/helpers.py" hands the retriever its own answer, and counting that
    as a hit would measure leakage rather than localization.

    Only full paths are scrubbed, never bare module names. Removing `groupby`
    from "BUG: groupby.apply raises" because the gold file happens to be
    `pandas/core/groupby/groupby.py` would delete the actual bug description.
    """
    text = _SQUASH_TRAILER_RE.sub("", message)
    text = _TRAILER_RE.sub("", text)
    text = _AUTHOR_LINE_RE.sub("", text)
    text = _CHERRY_PICK_RE.sub("", text)

    scrubbed = False
    if scrub:
        for path in sorted(gold_files, key=len, reverse=True):
            if path in text:
                text = text.replace(path, " ")
                scrubbed = True

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), scrubbed


def extract_issue_refs(message: str) -> list[int]:
    seen: list[int] = []
    for match in _ISSUE_RE.finditer(message):
        num = int(match.group(1))
        if num not in seen:
            seen.append(num)
    return seen


def mine_repo(repo_cfg: RepoConfig, cfg: Config, repo_path: Path) -> tuple[list[Example], Counter]:
    """Mine one repository into examples, plus the funnel counts for logging."""
    ref = resolve_ref(repo_path, repo_cfg.default_branch)
    commits = iter_commits(repo_path, ref)
    log.info("[%s] scanned %d commits on %s", repo_cfg.name, len(commits), ref)

    funnel: Counter = Counter()
    funnel["scanned"] = len(commits)

    kept: list[tuple[CommitInfo, list[str], list[str]]] = []
    for commit in commits:
        decision = classify(commit, cfg.mining)
        if not decision.kept:
            funnel[decision.reason] += 1
            continue
        kept.append((commit, list(decision.gold_files), decision.borderline))

    # Batch-verify that every gold file exists at the parent commit.
    examples: list[Example] = []
    if cfg.mining.require_gold_in_parent and kept:
        pairs: list[tuple[str, str]] = []
        for commit, gold, _ in kept:
            pairs.extend((commit.parent_shas[0], path) for path in gold)
        present = paths_present_at(repo_path, pairs)

        cursor = 0
        surviving: list[tuple[CommitInfo, list[str], list[str]]] = []
        for commit, gold, borderline in kept:
            flags = present[cursor : cursor + len(gold)]
            cursor += len(gold)
            filtered = [p for p, ok in zip(gold, flags, strict=True) if ok]
            if not filtered:
                funnel[GOLD_MISSING_AT_PARENT] += 1
                continue
            if len(filtered) != len(gold):
                borderline = [*borderline, PARTIAL_GOLD_AT_PARENT]
            surviving.append((commit, filtered, borderline))
        kept = surviving

    for commit, gold, borderline in kept:
        query_text, scrubbed = build_query_text(
            commit.message, tuple(gold), cfg.mining.scrub_gold_paths_from_query
        )
        examples.append(
            Example(
                example_id=f"{repo_cfg.name}@{commit.sha[:12]}",
                repo=repo_cfg.name,
                fix_sha=commit.sha,
                parent_sha=commit.parent_shas[0],
                authored_at=commit.authored_at,
                query_text=query_text,
                gold_files=gold,
                n_files_changed=len(commit.changed_paths),
                issue_refs=extract_issue_refs(commit.message),
                borderline=borderline,
                query_scrubbed=scrubbed,
            )
        )

    funnel["kept"] = len(examples)
    _log_funnel(repo_cfg.name, funnel)
    return examples, funnel


def _log_funnel(repo: str, funnel: Counter) -> None:
    scanned = funnel["scanned"]
    for reason in FUNNEL_ORDER:
        count = funnel.get(reason, 0)
        if count:
            log.info("[%s]   -%-24s %6d (%.1f%%)", repo, reason, count, 100 * count / scanned)
    log.info("[%s]   = kept %d of %d commits", repo, funnel["kept"], scanned)
