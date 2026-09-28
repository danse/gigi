"""Spherical k-means (no model downloads)."""

from __future__ import annotations

import numpy as np

from gigi.indexing.cluster import build_clusters
from gigi.indexing.embedder import Embedder
from gigi.indexing.loader import Chunk


def test_build_clusters_separates_two_groups():
    chunks = [
        Chunk(source="a.md", text="alpha", heading="Alpha", idx=0),
        Chunk(source="a.md", text="alpha 2", heading="Alpha", idx=1),
        Chunk(source="b.md", text="beta", heading="Beta", idx=2),
        Chunk(source="b.md", text="beta 2", heading="Beta", idx=3),
    ]
    embeddings = np.array(
        [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]],
        dtype=np.float32,
    )
    records = build_clusters(chunks, embeddings, n_clusters=2, seed=0)
    assert len(records) == 2
    headings = {r.heading for r in records}
    assert headings == {"Alpha", "Beta"}
    assert sum(r.size for r in records) == 4
    assert all(0 <= r.centroid_idx < 4 for r in records)


def test_kmeans_pp_accepts_float32_weights_that_do_not_sum_to_one():
    """Large float32 distance vectors often fail numpy.choice's p.sum()==1 check."""
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(4000, 32)).astype(np.float32)
    chunks = [Chunk(source=f"{i}.md", text=str(i), heading=str(i), idx=i) for i in range(4000)]
    records = build_clusters(chunks, embeddings, n_clusters=16, seed=0)
    assert records
    assert sum(r.size for r in records) == 4000


def test_bge_query_instruction_without_loading_model():
    embedder = Embedder("BAAI/bge-small-en-v1.5")
    formatted = embedder.format_query("how do I deploy?")
    assert formatted.startswith("Represent this sentence for searching relevant passages:")
    assert embedder.format_query(formatted) == formatted
