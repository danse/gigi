# gigi — agent notes

Usage, env vars, eval harness and architecture are documented in `README.md`;
this file only records things an agent cannot discover directly from the code.

## Gotchas

- Don't worry about what's committed — users handle commits, agents don't.
- Only the embed model is in the index manifest: changing it forces a full
  reindex (`check_embed_model`); changing the rerank model does not.
- The store records chunk `source` as full paths; `gigi eval` normalizes to
  bare file names before scoring.
- `gigi index <root>` skips the index output dir (any `.index` folder), so a
  corpus never contains its own index files.
- `clusters.json` records each cluster's `members` (chunk ids, nearest-centroid
  first); `gigi summarise` feeds `overview_per_cluster` of them per topic.
  Indexes built before `members` existed need `gigi index --recluster` (no
  re-embed) to light that up — until then summaries use one passage per topic.
- Graph convention: conditional edges own control flow; nodes are
  single-purpose.
- The retrieval defaults (reranker off, `top_k=16`, `mmr_lambda=0.7`,
  `n_clusters=16`) were tuned with `gigi eval --grid` on the fixture corpus —
  the sweep had reranking hurting mrr/basis. Re-tune deliberately, not by hand.