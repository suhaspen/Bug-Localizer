"""Tests for corpus construction and BM25 ranking against a real git repo.

No Postgres and no embedding model here — these cover the parts that decide
*what is searchable* and *how sparse ranking behaves*, both of which silently
change every downstream number if wrong.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from buglocalizer.config import Config, MiningConfig
from buglocalizer.corpus import list_corpus, read_blobs
from buglocalizer.retrieval.sparse import TokenCache, bm25_search

MINING = MiningConfig(
    exclude_path_globs=["docs/**", "**/tests/**", "**/test_*.py", "*.md", "setup.py"],
    source_extensions=[".py"],
)


def git(path: Path, *args: str, env_extra: dict | None = None) -> str:
    env = {
        "GIT_AUTHOR_NAME": "T",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "T",
        "GIT_COMMITTER_EMAIL": "t@example.com",
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        **(env_extra or {}),
    }
    return subprocess.run(
        ["git", "-C", str(path), *args], capture_output=True, text=True, check=True, env=env
    ).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "toy"
    r.mkdir()
    git(r, "init", "-b", "main")

    files = {
        "toy/parser.py": (
            "def parse_csv(text):\n"
            "    '''Split a CSV line into fields.'''\n"
            "    return text.split(',')\n"
        ),
        "toy/writer.py": "def write_json(obj):\n    return repr(obj)\n",
        "toy/empty.py": "",
        "tests/test_parser.py": "def test_parse_csv():\n    assert parse_csv('a,b')\n",
        "docs/guide.md": "# guide\n",
        "setup.py": "from setuptools import setup\n",
        "README.md": "# toy\n",
    }
    for rel, content in files.items():
        p = r / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    git(r, "add", "-A")
    git(
        r,
        "commit",
        "-m",
        "init",
        env_extra={
            "GIT_AUTHOR_DATE": "2020-01-01T00:00:00+00:00",
            "GIT_COMMITTER_DATE": "2020-01-01T00:00:00+00:00",
        },
    )
    return r


@pytest.fixture
def cfg() -> Config:
    return Config(mining=MINING)


def head(repo: Path) -> str:
    return git(repo, "rev-parse", "HEAD").strip()


def test_corpus_equals_label_space(repo, cfg):
    """Default corpus is exactly the files that could be a gold label."""
    files = list_corpus(repo, head(repo), cfg)
    assert {f.path for f in files} == {"toy/parser.py", "toy/writer.py", "toy/empty.py"}


def test_include_tests_widens_the_corpus(repo, cfg):
    cfg.corpus.include_tests = True
    files = list_corpus(repo, head(repo), cfg)
    paths = {f.path for f in files}
    assert "tests/test_parser.py" in paths
    assert "setup.py" in paths
    # Still Python only — a .md file is never a candidate.
    assert not any(p.endswith(".md") for p in paths)


def test_corpus_entries_carry_blob_sha_and_size(repo, cfg):
    files = list_corpus(repo, head(repo), cfg)
    parser = next(f for f in files if f.path == "toy/parser.py")
    assert len(parser.blob_sha) == 40
    assert parser.n_bytes > 0


def test_read_blobs_batches_and_roundtrips(repo, cfg):
    files = list_corpus(repo, head(repo), cfg)
    contents = read_blobs(repo, [f.blob_sha for f in files])
    assert len(contents) == len(files)
    parser = next(f for f in files if f.path == "toy/parser.py")
    assert "def parse_csv" in contents[parser.blob_sha]


def test_read_blobs_empty_input():
    assert read_blobs(Path("."), []) == {}


def test_read_blobs_survives_a_missing_sha(repo, cfg):
    """A bad sha must not desynchronise the stream and corrupt later files."""
    files = list_corpus(repo, head(repo), cfg)
    good = [f.blob_sha for f in files]
    contents = read_blobs(repo, [*good, "0" * 40])
    assert len(contents) == len(good)


# --- BM25 --------------------------------------------------------------------


def test_bm25_ranks_the_relevant_file_first(repo, cfg):
    files = list_corpus(repo, head(repo), cfg)
    contents = read_blobs(repo, [f.blob_sha for f in files])
    result = bm25_search(cfg, "ex1", "parse_csv drops whitespace between fields", files, contents)
    assert result.ranked[0].path == "toy/parser.py"
    assert result.n_candidates == 3


def test_bm25_ranking_is_deterministic_on_ties(repo, cfg):
    """Zero-score files must come back in a stable order, or top-k jitters."""
    files = list_corpus(repo, head(repo), cfg)
    contents = read_blobs(repo, [f.blob_sha for f in files])
    runs = [
        bm25_search(cfg, "ex", "nomatchingtermatall", files, contents).paths() for _ in range(3)
    ]
    assert runs[0] == runs[1] == runs[2]


def test_bm25_returns_every_candidate_not_just_matches(repo, cfg):
    """The eval needs a full ranking, so non-matching files still get a rank."""
    files = list_corpus(repo, head(repo), cfg)
    contents = read_blobs(repo, [f.blob_sha for f in files])
    result = bm25_search(cfg, "ex", "parse_csv", files, contents)
    assert len(result.ranked) == len(files)


def test_bm25_top_k_truncates(repo, cfg):
    files = list_corpus(repo, head(repo), cfg)
    contents = read_blobs(repo, [f.blob_sha for f in files])
    assert len(bm25_search(cfg, "ex", "parse", files, contents, top_k=2).ranked) == 2


def test_bm25_empty_corpus_is_not_an_error(cfg):
    result = bm25_search(cfg, "ex", "anything", [], {})
    assert result.ranked == []
    assert result.n_candidates == 0


def test_token_cache_hits_on_repeated_blobs(repo, cfg):
    files = list_corpus(repo, head(repo), cfg)
    contents = read_blobs(repo, [f.blob_sha for f in files])
    cache = TokenCache()
    bm25_search(cfg, "a", "parse", files, contents, token_cache=cache)
    assert cache.hit_rate == 0.0
    bm25_search(cfg, "b", "parse", files, contents, token_cache=cache)
    assert cache.hit_rate == 0.5  # second pass is all hits


def test_token_cache_evicts_beyond_maxsize():
    cache = TokenCache(maxsize=2)
    for i in range(5):
        cache.get(f"sha{i}", "def f(): pass")
    assert len(cache._data) == 2
