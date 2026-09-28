"""Topic-cluster representatives for `gigi summarise` and the eval harness.

The production `summarise` command and `gigi eval`'s overview cases both select
documents through this function, so the harness measures exactly the selection
logic the command uses.
"""

from __future__ import annotations

from gigi.indexing.cluster import ClusterRecord
from gigi.indexing.loader import Chunk


def cluster_representatives(chunks: list[Chunk], clusters: list[ClusterRecord]) -> list[dict]:
    """One representative chunk per topic cluster, with a coverage score (0..1).

    Chunks are ordered by cluster size (large topics first). A representative
    is the chunk nearest its cluster's centroid; its score is the cluster's
    share of the corpus, so `gigi summarise` answers with the main topics, not
    a nearest-neighbor query.
    """
    n = max(len(chunks), 1)
    retrieved: list[dict] = []
    for cluster in clusters:
        if cluster.centroid_idx < 0 or cluster.centroid_idx >= len(chunks):
            continue
        chunk = chunks[cluster.centroid_idx]
        retrieved.append(
            {
                "source": chunk.source,
                "text": chunk.text,
                "heading": cluster.heading or "",
                "score": cluster.size / n,
            }
        )
    return retrieved