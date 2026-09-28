"""Shared index-building pipeline (used by `gigi index` and `gigi eval`)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from gigi.config import Settings
from gigi.indexing.embedder import Embedder
from gigi.indexing.loader import load_documents
from gigi.indexing.store import IndexManifest, IndexStore

# ``on_scan(n_chunks)`` fires after scanning, ``on_batch(n_embedded, last_source)``
# after each embedding batch — the CLI uses them to drive a live progress bar.
ScanCallback = Callable[[int], None]
BatchCallback = Callable[[int, str], None]


def _excludes(root: Path, index_dir: Path) -> set[Path]:
    """Directories an index build must never read.

    The index's own output directory first of all; plus any ``.index`` folder
    inside the corpus root, so corpora that contain an older gigi index (e.g.
    after ``gigi index .``) are not slowly self-polluted.
    """
    excluded = {index_dir.resolve()}
    for candidate in Path(root).rglob(".index"):
        if candidate.is_dir():
            excluded.add(candidate.resolve())
    return excluded


def build_index(
    root: Path,
    settings: Settings,
    index_dir: Path | None = None,
    on_scan: ScanCallback | None = None,
    on_batch: BatchCallback | None = None,
) -> tuple[IndexStore, IndexManifest]:
    """Scan, chunk, embed and persist an index in one pass.

    ``gigi index`` and ``gigi eval`` share this so that evaluation measures the
    exact pipeline used in production (same chunking, same embedder, same store).
    """
    chunks = load_documents(
        root,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        exclude=_excludes(root, Path(index_dir or settings.index_dir)),
    )
    if not chunks:
        raise ValueError(f"no supported documents found under {root}")
    if on_scan is not None:
        on_scan(len(chunks))

    embedder = Embedder(settings.embed_model)
    embedded = []
    for start in range(0, len(chunks), settings.embed_batch_size):
        batch = chunks[start : start + settings.embed_batch_size]
        embedded.append(embedder.encode_chunks([c.text for c in batch]))
        if on_batch is not None:
            on_batch(len(batch), batch[-1].source if batch else "")
    embeddings = torch.cat(embedded).cpu().numpy().astype(np.float32)

    store = IndexStore(index_dir or settings.index_dir)
    store.clear()
    manifest = store.save(
        chunks,
        embeddings,
        settings.embed_model,
        n_clusters=settings.n_clusters,
    )
    return store, manifest