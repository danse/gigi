"""Cluster-representative selection used by `gigi summarise` (no model downloads)."""

from __future__ import annotations

from gigi.indexing.cluster import ClusterRecord
from gigi.indexing.loader import Chunk
from gigi.retrieval.overview import cluster_representatives


def _chunk(source: str, text: str, idx: int) -> Chunk:
    return Chunk(source=source, text=text, heading="h", idx=idx)


def test_representatives_cover_one_chunk_per_cluster():
    chunks = [_chunk("a.md", "a", 0), _chunk("b.md", "b", 1), _chunk("c.md", "c", 2)]
    clusters = [
        ClusterRecord(id=0, centroid_idx=0, size=2, heading="Alpha", source="a.md"),
        ClusterRecord(id=1, centroid_idx=2, size=1, heading="Gamma", source="c.md"),
    ]
    reps = cluster_representatives(chunks, clusters)
    assert [r["source"] for r in reps] == ["a.md", "c.md"]
    assert [r["heading"] for r in reps] == ["Alpha", "Gamma"]
    assert reps[0]["score"] == 2 / 3  # cluster share of the corpus = coverage
    assert reps[1]["score"] == 1 / 3


def test_representatives_respect_cluster_order():
    chunks = [_chunk("a.md", "a", 0), _chunk("b.md", "b", 1)]
    clusters = [
        ClusterRecord(id=0, centroid_idx=1, size=1, heading="Small", source="b.md"),
        ClusterRecord(id=1, centroid_idx=0, size=9, heading="Big", source="a.md"),
    ]
    reps = cluster_representatives(chunks, clusters)
    # cluster_representatives keeps cluster order: index-time clusters are
    # pre-sorted by size, so big topics stay first.
    assert [r["source"] for r in reps] == ["b.md", "a.md"]


def test_representatives_skip_missing_centroids():
    chunks = [_chunk("a.md", "a", 0)]
    clusters = [
        ClusterRecord(id=0, centroid_idx=0, size=1, heading="A", source="a.md"),
        ClusterRecord(id=1, centroid_idx=99, size=1, heading="Ghost", source="x.md"),
    ]
    reps = cluster_representatives(chunks, clusters)
    assert [r["source"] for r in reps] == ["a.md"]


def test_representatives_empty_index():
    assert cluster_representatives([], []) == []