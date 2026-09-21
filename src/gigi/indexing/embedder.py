"""PyTorch sentence-embedding wrapper (lazy-loads the pretrained model)."""

from __future__ import annotations

import torch


class Embedder:
    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self._model = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            try:
                self._model = SentenceTransformer(
                    self.model_name, device=self.device, local_files_only=True
                )
            except OSError:
                self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: list[str]) -> torch.Tensor:
        """Return L2-normalized embeddings of shape (len(texts), dim)."""
        return self._load().encode(texts, convert_to_tensor=True, normalize_embeddings=True)

    def encode_one(self, text: str) -> torch.Tensor:
        return self.encode([text])[0]