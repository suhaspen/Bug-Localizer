"""Source files → chunks → embeddings → Postgres/pgvector."""

from buglocalizer.indexing.chunking import Chunk, chunk_offsets
from buglocalizer.indexing.indexer import IndexStats, index_examples

__all__ = ["Chunk", "chunk_offsets", "index_examples", "IndexStats"]
