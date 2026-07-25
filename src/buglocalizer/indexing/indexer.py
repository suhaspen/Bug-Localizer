"""Build the corpus index for a set of examples.

The loop is: for each example's parent commit, list the corpus files, work out
which blobs are not already stored, embed only those, and record the commit as
covered. Everything hangs off the blob-hash cache — on this dataset it is the
difference between 5.4 million embedding jobs and 94 thousand.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from buglocalizer.config import Config
from buglocalizer.corpus import list_corpus, read_blobs, repo_path
from buglocalizer.dataset import Example
from buglocalizer.indexing.chunking import chunk_offsets
from buglocalizer.indexing.embedder import get_embedder
from buglocalizer.indexing.store import (
    connect,
    existing_blobs,
    insert_blob_with_chunks,
    is_commit_indexed,
    mark_commit_indexed,
)
from buglocalizer.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class IndexStats:
    commits_seen: int = 0
    commits_skipped: int = 0
    file_instances: int = 0
    blobs_new: int = 0
    blobs_cached: int = 0
    chunks_embedded: int = 0
    seconds: float = 0.0
    per_repo: dict[str, int] = field(default_factory=dict)

    @property
    def dedup_factor(self) -> float:
        """File instances seen per blob actually embedded — the cache's payoff.

        Deliberately NOT divided by (new + cached): `blobs_cached` counts a hit
        every time a blob is seen again, so that denominator is roughly
        `file_instances` by construction and always reports ~1.0x.
        """
        return self.file_instances / self.blobs_new if self.blobs_new else 0.0


def index_examples(cfg: Config, examples: list[Example], force: bool = False) -> IndexStats:
    """Index the parent commit of every given example."""
    stats = IndexStats()
    embedder = get_embedder(cfg)
    t_start = time.perf_counter()

    # Group by repo so each repo's git process and DB scoping stay together, and
    # process oldest-first so consecutive commits share nearly all their blobs.
    by_repo: dict[str, list[Example]] = {}
    for ex in examples:
        by_repo.setdefault(ex.repo, []).append(ex)

    with connect(cfg) as conn:
        for repo, group in sorted(by_repo.items()):
            repo_dir = repo_path(cfg, repo)
            group = sorted(group, key=lambda e: e.authored_at)
            log.info("[%s] indexing %d commits", repo, len(group))

            for n, ex in enumerate(group, 1):
                stats.commits_seen += 1
                if not force and is_commit_indexed(conn, repo, ex.parent_sha):
                    stats.commits_skipped += 1
                    continue

                files = list_corpus(repo_dir, ex.parent_sha, cfg)
                stats.file_instances += len(files)

                wanted = sorted({f.blob_sha for f in files})
                have = existing_blobs(conn, repo, wanted)
                todo = [sha for sha in wanted if sha not in have]
                stats.blobs_cached += len(have)

                if todo:
                    contents = read_blobs(repo_dir, todo)
                    _embed_and_store(conn, cfg, embedder, repo, contents, stats)

                mark_commit_indexed(conn, repo, ex.parent_sha, len(files))
                conn.commit()

                if n % 25 == 0 or n == len(group):
                    log.info(
                        "[%s] %d/%d commits | %d new blobs | %d chunks | %.0f%% cache hit",
                        repo,
                        n,
                        len(group),
                        stats.blobs_new,
                        stats.chunks_embedded,
                        100 * stats.blobs_cached / max(stats.blobs_cached + stats.blobs_new, 1),
                    )
            stats.per_repo[repo] = stats.blobs_new

    stats.seconds = time.perf_counter() - t_start
    return stats


def _embed_and_store(
    conn, cfg: Config, embedder, repo: str, contents: dict[str, str], stats
) -> None:
    """Chunk a batch of blobs, embed all their chunks together, then store.

    Batched across blobs rather than per blob: a small file yields one chunk, and
    encoding one chunk at a time wastes most of the GPU's throughput.
    """
    pending: list[tuple[str, str, list]] = []
    batch_texts: list[str] = []

    for sha, content in contents.items():
        chunks = chunk_offsets(
            content, cfg.retrieval.chunk_max_chars, cfg.retrieval.chunk_overlap_chars
        )
        if not chunks:
            continue
        pending.append((sha, content, chunks))
        batch_texts.extend(content[c.start : c.end] for c in chunks)

    if not batch_texts:
        return

    vectors = embedder.encode(batch_texts)

    cursor = 0
    for sha, content, chunks in pending:
        vecs = vectors[cursor : cursor + len(chunks)]
        cursor += len(chunks)
        insert_blob_with_chunks(conn, repo, sha, content, chunks, vecs)
        stats.blobs_new += 1
        stats.chunks_embedded += len(chunks)
