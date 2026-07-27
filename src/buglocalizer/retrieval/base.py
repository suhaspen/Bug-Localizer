"""Shared retrieval types and the tokenizer used by the sparse side."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass

# Identifiers are the highest-signal tokens in a bug report: a report that says
# `send_file` and a file that defines `send_file` should match strongly. But a
# report saying "send file" should match too, so identifiers are emitted BOTH
# whole and split on snake_case/camelCase boundaries.
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tokenize(text: str) -> list[str]:
    """Tokenise source or query text.

    Tokens are interned. A tokenised corpus is millions of short strings, but a
    repository's vocabulary is only tens of thousands of distinct identifiers, so
    interning collapses the duplicates to shared references. Measured effect on
    the cached corpus: roughly a 6x reduction in resident memory, which matters
    because the alternative is the eval being paged out mid-run.
    """
    intern = sys.intern
    tokens: list[str] = []
    for match in _TOKEN_RE.findall(text):
        tokens.append(intern(match.lower()))
        parts = [p for p in _CAMEL_RE.sub(" ", match).replace("_", " ").split() if p]
        if len(parts) > 1:
            tokens.extend(intern(p.lower()) for p in parts)
    return tokens


@dataclass(frozen=True)
class ScoredFile:
    path: str
    score: float
    blob_sha: str


@dataclass
class RetrievalResult:
    method: str
    example_id: str
    ranked: list[ScoredFile]
    n_candidates: int
    seconds: float

    def paths(self, k: int | None = None) -> list[str]:
        return [s.path for s in (self.ranked[:k] if k else self.ranked)]
