"""PyTorch sentence-embedding wrapper (lazy-loads the pretrained model)."""

from __future__ import annotations

from gigi.cpucompat import configure_cpu, configure_torch, needs_sse_cap

configure_cpu()

import torch

configure_torch(torch)

# BGE retrieval quality depends on this instruction being applied to queries only.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# E5-family models (e.g. intfloat/multilingual-e5-small) are trained with explicit
# ``query:`` / ``passage:`` prefixes applied to both sides at inference time.
_E5_QUERY_PREFIX = "query: "
_E5_PASSAGE_PREFIX = "passage: "


def _prefixes(model_name: str) -> tuple[str, str]:
    """Return (query_prefix, passage_prefix) appropriate for the model family."""
    name = model_name.lower()
    if "e5" in name:
        return _E5_QUERY_PREFIX, _E5_PASSAGE_PREFIX
    if "bge" in name:
        return _BGE_QUERY_INSTRUCTION, ""
    return "", ""


class Embedder:
    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self._model = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            extra = {}
            if needs_sse_cap():
                # tokenizers.abi3.so is built with AVX2; the Python tokenizer is SSE-safe.
                extra["processor_kwargs"] = {"use_fast": False}
            try:
                self._model = SentenceTransformer(
                    self.model_name, device=self.device, local_files_only=True, **extra
                )
            except OSError:
                self._model = SentenceTransformer(self.model_name, device=self.device, **extra)
        return self._model

    def format(self, text: str, *, passage: bool = False) -> str:
        """Apply the model family's query/passage prefix (idempotent)."""
        prefix = _prefixes(self.model_name)[1 if passage else 0]
        if prefix and not text.startswith(prefix):
            return prefix + text
        return text

    def format_query(self, text: str) -> str:
        return self.format(text, passage=False)

    def format_passage(self, text: str) -> str:
        return self.format(text, passage=True)

    def encode(self, texts: list[str], *, passage: bool = False) -> torch.Tensor:
        """Return L2-normalized embeddings of shape (len(texts), dim)."""
        formatted = [self.format(text, passage=passage) for text in texts]
        return self._load().encode(formatted, convert_to_tensor=True, normalize_embeddings=True)

    def encode_one(self, text: str) -> torch.Tensor:
        """Embed a retrieval query (applies the model family's query prefix)."""
        return self.encode([text], passage=False)[0]

    def encode_chunks(self, texts: list[str]) -> torch.Tensor:
        """Embed document passages (applies the model family's passage prefix)."""
        return self.encode(texts, passage=True)