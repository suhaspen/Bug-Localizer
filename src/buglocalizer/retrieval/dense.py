"""Dense retrieval — embeddings and cosine similarity in pgvector.

Where BM25 matches literal words, dense retrieval matches meaning: query and
code are both mapped to 384-dimensional vectors positioned so that related
things land near each other. It can connect "upload breaks on large files" to
code about `chunk_size` with no shared vocabulary at all — the thing sparse
retrieval fundamentally cannot do.

Two implementation choices worth defending:

**Exact search, not an ANN index.** pgvector offers HNSW for approximate
nearest neighbours, which matters when scanning millions of vectors. Here every
query is restricted to one commit's corpus — a few hundred blobs, ~10k chunks —
and an exact scan over that is already fast. Using HNSW would add recall error
to a measurement whose entire purpose is measuring recall.

**A file scores as the max over its chunks.** A file is relevant if *any* part
of it is relevant; averaging would punish a large file with one highly relevant
function, which is exactly the case we care about.
"""

from __future__ import annotations

import time

from buglocalizer.config import Config
from buglocalizer.corpus import CorpusFile
from buglocalizer.retrieval.base import RetrievalResult, ScoredFile

# `<=>` is pgvector's cosine distance. Vectors are stored normalised, so
# similarity is exactly 1 - distance.
_SQL = """
SELECT blob_sha, MIN(embedding <=> %(q)s) AS best_distance
FROM chunk
WHERE repo = %(repo)s AND blob_sha = ANY(%(blobs)s)
GROUP BY blob_sha
"""


def dense_search(
    cfg: Config,
    conn,
    repo: str,
    example_id: str,
    query_text: str,
    files: list[CorpusFile],
    embedder,
    top_k: int | None = None,
    query_vec=None,
) -> RetrievalResult:
    t0 = time.perf_counter()
    if not files:
        return RetrievalResult("dense", example_id, [], 0, time.perf_counter() - t0)

    # Reranking needs the same vector to pick each candidate's best windows, so
    # the caller may pass it in rather than paying for a second forward pass.
    if query_vec is None:
        query_vec = embedder.encode_one(query_text)
    blob_shas = sorted({f.blob_sha for f in files})

    rows = conn.execute(_SQL, {"q": query_vec, "repo": repo, "blobs": blob_shas}).fetchall()
    best: dict[str, float] = {sha: 1.0 - float(dist) for sha, dist in rows}

    ranked = sorted(
        (
            ScoredFile(path=f.path, score=best.get(f.blob_sha, -1.0), blob_sha=f.blob_sha)
            for f in files
        ),
        key=lambda s: (-s.score, s.path),
    )
    if top_k:
        ranked = ranked[:top_k]
    return RetrievalResult("dense", example_id, ranked, len(files), time.perf_counter() - t0)
