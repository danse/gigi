"""Topic-cluster representatives for `gigi summarise` and the eval harness.

The production `summarise` command and `gigi eval`'s overview cases both select
documents through this function, so the harness measures exactly the selection
logic the command uses.
"""

from __future__ import annotations

from gigi.indexing.cluster import ClusterRecord
from gigi.indexing.loader import Chunk


def cluster_representatives(
    chunks: list[Chunk],
    clusters: list[ClusterRecord],
    per_cluster: int = 1,
) -> list[dict]:
    """The *per_cluster* chunks nearest each cluster centroid, largest topics first.

    Every returned chunk carries a ``cluster`` key (the record id) so the
    overview prompt can group passages by topic. A chunk's score is its
    cluster's share of the corpus. Clusters recorded without member lists
    (older indexes) fall back to their single representative, so ``summarise``
    stays correct until the index is reclustered.
    """
    n = max(len(chunks), 1)
    per_cluster = max(per_cluster, 1)
    retrieved: list[dict] = []
    for cluster in clusters:
        idxs = [i for i in (cluster.members or []) if 0 <= i < len(chunks)]
        if not idxs and 0 <= cluster.centroid_idx < len(chunks):
            idxs = [cluster.centroid_idx]
        for idx in idxs[:per_cluster]:
            chunk = chunks[idx]
            retrieved.append(
                {
                    "source": chunk.source,
                    "text": chunk.text,
                    "heading": cluster.heading or "",
                    "score": cluster.size / n,
                    "cluster": cluster.id,
                }
            )
    return retrieved