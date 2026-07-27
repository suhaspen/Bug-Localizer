"""Postgres + pgvector storage for the corpus.

Schema shape, and why:

    blob   (repo, blob_sha) -> the file's text, stored exactly once
    chunk  (repo, blob_sha, chunk_idx) -> a (start, end) window + its embedding

`repo` is part of every primary key even though a blob hash is already globally
unique by construction. That is deliberate denormalisation: the project reports
per-repo results, so every operation — index, drop, count, vector search — needs
to be cheaply scopeable to one repository.

Note what is *not* stored: paths. A blob is content, and the same content can
live at different paths in different commits. Paths come from `git ls-tree` at
query time, which keeps the store free of commit-specific duplication.
"""

from __future__ import annotations

from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector

from buglocalizer.config import Config
from buglocalizer.logging_setup import get_logger

log = get_logger(__name__)

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS blob (
    repo      text   NOT NULL,
    blob_sha  text   NOT NULL,
    n_bytes   int    NOT NULL,
    PRIMARY KEY (repo, blob_sha)
);

-- File text was stored here originally, then found to be dead weight: every
-- caller reads contents from git via `read_blobs` (one `cat-file --batch` per
-- tree, ~80 ms), so the column was a second copy of the corpus that nothing
-- queried. Dropping it frees ~20% of the database.
ALTER TABLE blob DROP COLUMN IF EXISTS content;

CREATE TABLE IF NOT EXISTS chunk (
    repo       text NOT NULL,
    blob_sha   text NOT NULL,
    chunk_idx  int  NOT NULL,
    start_char int  NOT NULL,
    end_char   int  NOT NULL,
    embedding  vector(VECTOR_DIM) NOT NULL,
    PRIMARY KEY (repo, blob_sha, chunk_idx),
    FOREIGN KEY (repo, blob_sha) REFERENCES blob(repo, blob_sha) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS chunk_repo_blob_idx ON chunk (repo, blob_sha);

CREATE TABLE IF NOT EXISTS indexed_commit (
    repo       text NOT NULL,
    commit_sha text NOT NULL,
    n_files    int  NOT NULL,
    indexed_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (repo, commit_sha)
);

-- Coverage depends on the corpus scope it was built at. include_tests=true
-- embeds a superset of the default scope, so one wide build serves both evals --
-- but without recording which scope was used, a later wider run would see the
-- commit marked "indexed" and skip it, silently leaving test files unembedded
-- and dense retrieval scoring them all at -1.
ALTER TABLE indexed_commit
    ADD COLUMN IF NOT EXISTS include_tests boolean NOT NULL DEFAULT false;
"""


@contextmanager
def connect(cfg: Config):
    with psycopg.connect(cfg.db.dsn) as conn:
        register_vector(conn)
        yield conn


def init_schema(cfg: Config) -> None:
    # Plain string replacement rather than %-formatting: the schema carries
    # explanatory comments, and a stray "20% of" in one of them is a valid
    # `%o` conversion that blows up at load time.
    ddl = SCHEMA.replace("VECTOR_DIM", str(cfg.retrieval.embedding_dim))
    with connect(cfg) as conn:
        conn.execute(ddl)
        conn.commit()
    log.info("schema ready (embedding dim %d)", cfg.retrieval.embedding_dim)


def existing_blobs(conn, repo: str, blob_shas: list[str]) -> set[str]:
    """Which of these blobs are already embedded — the cache check."""
    if not blob_shas:
        return set()
    rows = conn.execute(
        "SELECT blob_sha FROM blob WHERE repo = %s AND blob_sha = ANY(%s)",
        (repo, blob_shas),
    ).fetchall()
    return {r[0] for r in rows}


def insert_blob_with_chunks(
    conn, repo: str, blob_sha: str, content: str, chunks, embeddings
) -> None:
    conn.execute(
        "INSERT INTO blob (repo, blob_sha, n_bytes) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
        (repo, blob_sha, len(content.encode())),
    )
    with (
        conn.cursor() as cur,
        cur.copy(
            "COPY chunk (repo, blob_sha, chunk_idx, start_char, end_char, embedding) FROM STDIN "
            "WITH (FORMAT BINARY)"
        ) as cp,
    ):
        cp.set_types(["text", "text", "int4", "int4", "int4", "vector"])
        for ch, emb in zip(chunks, embeddings, strict=True):
            cp.write_row([repo, blob_sha, ch.idx, ch.start, ch.end, emb])


def mark_commit_indexed(
    conn, repo: str, commit_sha: str, n_files: int, include_tests: bool
) -> None:
    """Record coverage, never narrowing it.

    `include_tests` is OR-ed rather than overwritten: once a commit has been
    indexed at the wide scope it stays covered for both, even if a later
    default-scope run touches it.
    """
    conn.execute(
        "INSERT INTO indexed_commit (repo, commit_sha, n_files, include_tests) "
        "VALUES (%s, %s, %s, %s) "
        "ON CONFLICT (repo, commit_sha) DO UPDATE SET n_files = EXCLUDED.n_files, "
        "indexed_at = now(), "
        "include_tests = indexed_commit.include_tests OR EXCLUDED.include_tests",
        (repo, commit_sha, n_files, include_tests),
    )


def is_commit_indexed(conn, repo: str, commit_sha: str, include_tests: bool) -> bool:
    """Covered only if indexed at a scope at least as wide as the one requested."""
    row = conn.execute(
        "SELECT 1 FROM indexed_commit WHERE repo = %s AND commit_sha = %s "
        "AND (include_tests OR NOT %s)",
        (repo, commit_sha, include_tests),
    ).fetchone()
    return row is not None


def covered_commits(conn, include_tests: bool) -> set[tuple[str, str]]:
    rows = conn.execute(
        "SELECT repo, commit_sha FROM indexed_commit WHERE (include_tests OR NOT %s)",
        (include_tests,),
    ).fetchall()
    return {(r[0], r[1]) for r in rows}


def index_stats(conn) -> list[tuple]:
    return conn.execute(
        """
        SELECT b.repo,
               count(DISTINCT b.blob_sha)                       AS blobs,
               coalesce(sum(b.n_bytes), 0)                       AS bytes,
               (SELECT count(*) FROM chunk c WHERE c.repo = b.repo) AS chunks,
               (SELECT count(*) FROM indexed_commit i WHERE i.repo = b.repo) AS commits
        FROM blob b GROUP BY b.repo ORDER BY b.repo
        """
    ).fetchall()


def drop_repo(conn, repo: str) -> None:
    conn.execute("DELETE FROM blob WHERE repo = %s", (repo,))
    conn.execute("DELETE FROM indexed_commit WHERE repo = %s", (repo,))
