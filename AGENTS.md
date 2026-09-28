# gigi — agent notes

Usage, env vars, eval harness and architecture are documented in `README.md`;
this file only records things an agent cannot discover directly from the code.

## Gotchas

- Don't worry about what's committed — users handle commits, agents don't.
- Only the embed model is in the index manifest: changing it forces a full
  reindex (`check_embed_model`); changing the rerank model does not.
- The store records chunk `source` as full paths; `gigi eval` normalizes to
  bare file names before scoring.
- Graph convention: conditional edges own control flow; nodes are
  single-purpose.