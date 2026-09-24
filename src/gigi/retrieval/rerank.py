"""Cross-encoder reranking over the top candidates (PyTorch scoring layer)."""

from __future__ import annotations

from gigi.cpucompat import needs_sse_cap
from gigi.indexing.loader import Chunk


class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            extra = {}
            if needs_sse_cap():
                extra["processor_kwargs"] = {"use_fast": False}
            try:
                self._model = CrossEncoder(self.model_name, local_files_only=True, **extra)
            except OSError:
                self._model = CrossEncoder(self.model_name, **extra)
        return self._model

    def rerank(self, query: str, chunks: list[Chunk], top_k: int | None = None) -> list[tuple[Chunk, float]]:
        pairs = [(query, c.text) for c in chunks]
        scores = self._load().predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]
        return [(chunk, float(score)) for chunk, score in ranked]