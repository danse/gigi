"""Runtime configuration for gigi.

All values can be overridden via environment variables, which makes the CLI
easy to script and the LLM backend easy to swap.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    index_dir: Path = Path.cwd() / ".index"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_enabled: bool = True
    top_k: int = 16
    rerank_top_k: int = 8
    n_clusters: int = 16
    chunk_size: int = 800
    chunk_overlap: int = 100
    embed_batch_size: int = 32
    grade_threshold: float | None = None
    llm_kind: str = "ollama"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"
    openai_url: str = ""
    openai_api_key: str = ""
    openai_model: str = ""
    max_attempts: int = 2
    llm_timeout: float = 1800.0

    def resolve_grade_threshold(self) -> float | None:
        """None means keep every retrieved chunk (reranker already truncated)."""
        if self.grade_threshold is not None:
            return self.grade_threshold
        if self.rerank_enabled:
            return None
        return 0.3

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            index_dir=Path(os.environ.get("GIGI_INDEX_DIR", str(Path.cwd() / ".index"))),
            embed_model=os.environ.get("GIGI_EMBED_MODEL", cls.embed_model),
            rerank_model=os.environ.get("GIGI_RERANK_MODEL", cls.rerank_model),
            rerank_enabled=_env_bool("GIGI_RERANK", cls.rerank_enabled),
            top_k=int(os.environ.get("GIGI_TOP_K", cls.top_k)),
            rerank_top_k=int(os.environ.get("GIGI_RERANK_TOP_K", cls.rerank_top_k)),
            n_clusters=int(os.environ.get("GIGI_N_CLUSTERS", cls.n_clusters)),
            chunk_size=int(os.environ.get("GIGI_CHUNK_SIZE", cls.chunk_size)),
            chunk_overlap=int(os.environ.get("GIGI_CHUNK_OVERLAP", cls.chunk_overlap)),
            embed_batch_size=int(os.environ.get("GIGI_EMBED_BATCH_SIZE", cls.embed_batch_size)),
            grade_threshold=(
                float(os.environ["GIGI_GRADE_THRESHOLD"])
                if os.environ.get("GIGI_GRADE_THRESHOLD")
                else cls.grade_threshold
            ),
            llm_kind=os.environ.get("GIGI_LLM", cls.llm_kind),
            ollama_url=os.environ.get("GIGI_OLLAMA_URL", cls.ollama_url),
            ollama_model=os.environ.get("GIGI_OLLAMA_MODEL", cls.ollama_model),
            openai_url=os.environ.get("GIGI_OPENAI_URL", cls.openai_url),
            openai_api_key=os.environ.get("GIGI_OPENAI_API_KEY", cls.openai_api_key),
            openai_model=os.environ.get("GIGI_OPENAI_MODEL", cls.openai_model),
            max_attempts=int(os.environ.get("GIGI_MAX_ATTEMPTS", cls.max_attempts)),
            llm_timeout=float(os.environ.get("GIGI_LLM_TIMEOUT", cls.llm_timeout)),
        )