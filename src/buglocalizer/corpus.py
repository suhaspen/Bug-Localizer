"""The searchable corpus at a given commit.

One idea underpins the whole indexing design: **a corpus entry is content, not a
file**. Git already content-addresses every file version with a blob hash, and
two commits that didn't touch a file share its blob exactly. So we key
everything on blob hash rather than (commit, path), and near-identical trees
collapse onto the same stored rows.

Measured on this dataset: 5,398,769 (commit, path) file-instances across pandas
reduce to 93,848 distinct blobs — a 57x saving. Without that, embedding the
corpus would be arithmetically impossible on a laptop.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from buglocalizer.config import Config
from buglocalizer.mining.filters import is_excluded_path, is_source_path


@dataclass(frozen=True)
class CorpusFile:
    path: str
    blob_sha: str
    n_bytes: int


def repo_path(cfg: Config, repo: str) -> Path:
    return cfg.paths.cache_dir / repo


def _run_git(repo_dir: Path, args: list[str], stdin: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        input=stdin,
        capture_output=True,
        check=True,
    ).stdout


def list_corpus(repo_dir: Path, commit_sha: str, cfg: Config) -> list[CorpusFile]:
    """The files a retriever is allowed to return for a query at this commit.

    By default the corpus is exactly the set of paths that could be a gold label
    — the same exclusions the miner applies. Corpus and label space agree, which
    is what makes a "wrong" answer actually wrong: if tests were searchable, a
    method that ranked `tests/test_frame.py` first would be scored as a miss even
    though the test is genuinely about the bug. That would penalise sensible
    behaviour for a reason that is an artifact of our labelling convention.

    `corpus.include_tests: true` restores the harder, more realistic setting.
    """
    out = _run_git(repo_dir, ["ls-tree", "-r", "-l", commit_sha]).decode("utf-8", errors="replace")
    m = cfg.mining
    files: list[CorpusFile] = []
    for line in out.splitlines():
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) != 4 or parts[1] != "blob":
            continue
        if not is_source_path(path, m.source_extensions):
            continue
        if not cfg.corpus.include_tests and is_excluded_path(path, m.exclude_path_globs):
            continue
        size = int(parts[3]) if parts[3].isdigit() else 0
        files.append(CorpusFile(path=path, blob_sha=parts[2], n_bytes=size))
    return files


def read_blobs(repo_dir: Path, blob_shas: list[str]) -> dict[str, str]:
    """Read many blob contents in a single `git cat-file --batch` process.

    One process for a whole tree rather than one per file: measured at 0.2s for
    a 1,400-file pandas tree, versus minutes if spawned per file.
    """
    if not blob_shas:
        return {}
    stdin = "".join(sha + "\n" for sha in blob_shas).encode()
    out = _run_git(repo_dir, ["cat-file", "--batch"], stdin=stdin)

    contents: dict[str, str] = {}
    pos = 0
    for sha in blob_shas:
        nl = out.find(b"\n", pos)
        if nl == -1:
            break
        header = out[pos:nl].decode()
        parts = header.split()
        if len(parts) != 3:
            # "<sha> missing" — skip it rather than desynchronising the stream.
            pos = nl + 1
            continue
        size = int(parts[2])
        contents[sha] = out[nl + 1 : nl + 1 + size].decode("utf-8", errors="replace")
        pos = nl + 1 + size + 1  # trailing newline after the payload
    return contents
