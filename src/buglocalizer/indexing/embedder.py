"""The embedding model wrapper.

Imports `sentence_transformers` lazily. The mining, filtering and metric code
must stay importable without the ML stack — that boundary is what keeps
`make test` fast and the dataset logic independent of torch.
"""

from __future__ import annotations

from buglocalizer.config import Config
from buglocalizer.logging_setup import get_logger

log = get_logger(__name__)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Embedder:
    def __init__(self, cfg: Config):
        from sentence_transformers import SentenceTransformer

        self.cfg = cfg
        self.device = resolve_device(cfg.retrieval.device)
        log.info("loading %s on %s", cfg.retrieval.embedding_model, self.device)
        self.model = SentenceTransformer(cfg.retrieval.embedding_model, device=self.device)
        self.max_tokens = self.model.max_seq_length

        get_dim = getattr(self.model, "get_embedding_dimension", None) or (
            self.model.get_sentence_embedding_dimension
        )
        dim = get_dim()
        if dim != cfg.retrieval.embedding_dim:
            raise ValueError(
                f"model {cfg.retrieval.embedding_model} emits {dim}-d vectors but config "
                f"declares embedding_dim={cfg.retrieval.embedding_dim}. The pgvector column "
                f"is fixed-width, so this must match — update the config and re-index."
            )

    def encode(self, texts: list[str], normalize: bool = True):
        """Embed a batch.

        Normalised to unit length so cosine similarity is a plain dot product,
        and so pgvector's `<=>` cosine distance is exactly `1 - similarity`.
        """
        return self.model.encode(
            texts,
            batch_size=self.cfg.retrieval.embedding_batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def encode_one(self, text: str):
        return self.encode([text])[0]


_EMBEDDER: Embedder | None = None


def get_embedder(cfg: Config) -> Embedder:
    """Process-wide singleton — loading the model takes ~3s and it is stateless."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Embedder(cfg)
    return _EMBEDDER
