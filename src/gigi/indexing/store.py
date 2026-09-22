"""Persistence for the embedding index."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from gigi.indexing.cluster import ClusterRecord, build_clusters
from gigi.indexing.loader import Chunk


def _read_json_records(text: str) -> list[dict]:
    """Decode a JSONL stream (or a single JSON array) via ``JSONDecoder``.

    Do not use ``str.splitlines()``: it splits on U+2028/U+2029/U+0085, which
    ``json.dumps(..., ensure_ascii=False)`` does not escape inside strings.
    """
    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    records: list[dict] = []
    while idx < n:
        if text[idx].isspace():
            idx += 1
            continue
        obj, idx = decoder.raw_decode(text, idx)
        records.append(obj)
    if len(records) == 1 and isinstance(records[0], list):
        return records[0]
    return records


@dataclass
class IndexManifest:
    model_name: str
    dimension: int
    created_at: str
    n_chunks: int

    def as_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "created_at": self.created_at,
            "n_chunks": self.n_chunks,
        }

    @classmethod
    def from_dict(cls, d: dict) -> IndexManifest:
        return cls(
            model_name=str(d["model_name"]),
            dimension=int(d["dimension"]),
            created_at=str(d["created_at"]),
            n_chunks=int(d["n_chunks"]),
        )


class IndexStore:
    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.chunks_file = self.index_dir / "chunks.jsonl"
        self.embeddings_file = self.index_dir / "embeddings.npy"
        self.manifest_file = self.index_dir / "manifest.json"
        self.clusters_file = self.index_dir / "clusters.json"

    def exists(self) -> bool:
        return all(p.is_file() for p in (self.chunks_file, self.embeddings_file, self.manifest_file))

    def save(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        model_name: str,
        n_clusters: int = 16,
    ) -> IndexManifest:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with self.chunks_file.open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(json.dumps(chunk.as_dict(), ensure_ascii=False) + "\n")
        np.save(self.embeddings_file, embeddings)
        clusters = build_clusters(chunks, embeddings, n_clusters=n_clusters)
        self.clusters_file.write_text(
            json.dumps({"n_clusters": len(clusters), "clusters": [c.as_dict() for c in clusters]}, indent=2),
            encoding="utf-8",
        )
        manifest = IndexManifest(
            model_name=model_name,
            dimension=int(embeddings.shape[1]),
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            n_chunks=len(chunks),
        )
        self.manifest_file.write_text(json.dumps(manifest.as_dict(), indent=2))
        return manifest

    def load_clusters(self) -> list[ClusterRecord]:
        if not self.clusters_file.is_file():
            return []
        data = json.loads(self.clusters_file.read_text(encoding="utf-8"))
        return [ClusterRecord.from_dict(d) for d in data.get("clusters", [])]

    def load(self) -> tuple[list[Chunk], np.ndarray, IndexManifest]:
        if not self.exists():
            raise FileNotFoundError(f"no index found at {self.index_dir}; run `gigi index <dir>` first")
        try:
            records = _read_json_records(self.chunks_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(
                f"corrupt index at {self.chunks_file}: {e}; re-run `gigi index <dir>`"
            ) from e
        chunks = [Chunk.from_dict(d) for d in records]
        embeddings = np.load(self.embeddings_file)
        manifest = IndexManifest.from_dict(json.loads(self.manifest_file.read_text()))
        return chunks, embeddings, manifest

    def clear(self) -> None:
        if self.index_dir.exists():
            for p in self.index_dir.iterdir():
                if p.is_file():
                    p.unlink()