"""Splitting a source file into indexable units.

Chunking exists because embedding models have a hard token limit, and because
one vector for a 400 KB file is too blurry to discriminate anything. The size is
not a free parameter — it is dictated by the model. See `docs/02_retrieval.md`
for the alternatives that were measured and rejected.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    idx: int
    start: int
    end: int

    def text(self, content: str) -> str:
        return content[self.start : self.end]


def chunk_offsets(content: str, max_chars: int, overlap_chars: int) -> list[Chunk]:
    """Fixed-size sliding windows over the raw file text.

    Offsets rather than strings: the file content is stored once, and a chunk is
    a (start, end) view into it. That avoids storing an overlapping second copy
    of the entire corpus.

    The overlap matters. A hard split at 700 characters will sometimes land in
    the middle of the one function that answers the query, leaving neither half
    with enough context to match. Overlapping windows mean every region of the
    file appears intact in at least one chunk.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    if not content.strip():
        return []
    if len(content) <= max_chars:
        return [Chunk(idx=0, start=0, end=len(content))]

    step = max_chars - overlap_chars
    chunks: list[Chunk] = []
    start = 0
    while start < len(content):
        end = min(start + max_chars, len(content))
        if content[start:end].strip():
            chunks.append(Chunk(idx=len(chunks), start=start, end=end))
        if end == len(content):
            break
        start += step
    return chunks
