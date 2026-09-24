# gigi

Local document Q&A assistant built on **PyTorch** + **LangGraph**.

Ask questions over a folder of documents. A pretrained PyTorch sentence-transformer
embeds your docs and query, torch does cosine-similarity retrieval, a PyTorch
cross-encoder reranks the candidates, and a LangGraph agent retrieves → grades →
generates a grounded answer (with a self-correction loop) via a local Ollama LLM.

`gigi index` also clusters the embeddings into topics. Overview questions such as
`gigi ask "what are these documents about?"` use those cluster representatives
instead of nearest-neighbor search, so a large index can still be summarized.

```
gigi index examples/docs     # build the embedding index + topic clusters
gigi ask "How do I deploy the app?"
gigi ask "what are these documents about?"
```

## What makes it tick

- **PyTorch retrieval stack**
  - Embeddings: `BAAI/bge-small-en-v1.5` (sentence-transformers; BGE query instruction applied)
  - Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (CrossEncoder)
  - Top-k via cosine similarity + MMR, computed with `torch.matmul`
  - Spherical k-means over the index (`clusters.json`) for corpus-overview questions
- **LangGraph agent** (`src/gigi/agent/graph.py`)
  - After `retrieve`, a conditional edge sends overview questions to
    `generate_overview` (cluster representatives) and everything else through
    `grade → generate`. Grade bails out to a "no answer" node when nothing
    clears the relevance threshold. Generate retries (max `GIGI_MAX_ATTEMPTS`)
    when grounding fails.
  - State machine is compiled with an in-memory checkpointer.
- **Pluggable LLM** (`src/gigi/agent/llm.py`): Ollama (default), any
  OpenAI-compatible endpoint, or an offline stub for tests.

## Setup

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

There are two ways to use gigi:

### Option A — install it as a command (recommended)

This installs gigi into a managed environment and puts `gigi` on your `PATH`:

```bash
uv tool install .
```

After that, `gigi index <dir>`, `gigi ask "..."` and `gigi status` work from
anywhere. Skip the `--extra dev` unless you plan to edit the code and run the
tests.

The index is stored in `./.index` relative to the directory you run gigi from,
so each project gets its own index — `gigi index` a folder, then `ask` from the
same directory. Override the location with `GIGI_INDEX_DIR`.

### Option B — run from the repo (development)

```bash
uv sync --extra dev      # deps + editable install + pytest/ruff
uv run gigi --help       # run without a global install
```

### Ollama

Install Ollama and pull the default model (the first `ask` will also
auto-download the embedding + reranker models from Hugging Face):

```bash
curl -fsSL https://ollama.com/install.sh | sh   # or download the binary
ollama serve &
ollama pull llama3.2
```

## Usage

```bash
# Index a folder of .md/.txt/.rst/.pdf files
gigi index examples/docs

# Rebuild topic clusters only (keeps embeddings; uses GIGI_N_CLUSTERS)
gigi index --recluster

# Ask a question (answer + cited sources)
gigi ask "What is the deployment process?"

# Summarize the index (uses topic clusters, not query nearest-neighbors)
gigi ask "what are these documents about?"

# Inspect the index
gigi status
```

## Configuration (env vars)

| Var                   | Default                              | Description                          |
| --------------------- | ------------------------------------ | ------------------------------------ |
| `GIGI_INDEX_DIR`      | `./.index` (in the current dir)      | Where the embedding index lives; one per directory |
| `GIGI_EMBED_MODEL`    | `BAAI/bge-small-en-v1.5`             | Sentence-transformer for embeddings  |
| `GIGI_RERANK_MODEL`   | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker            |
| `GIGI_RERANK`         | `1`                                  | Enable the reranker (`0` to disable) |
| `GIGI_TOP_K`          | `16`                                 | Candidates retrieved before rerank   |
| `GIGI_RERANK_TOP_K`   | `8`                                  | Candidates kept after rerank         |
| `GIGI_N_CLUSTERS`     | `16`                                 | Topic clusters built at index time   |
| `GIGI_GRADE_THRESHOLD`| auto (off when rerank / `0.3` cosine)| Relevance cutoff; unset keeps reranked hits |
| `GIGI_CHUNK_SIZE`     | `800`                                | Chunk size in characters             |
| `GIGI_CHUNK_OVERLAP`  | `100`                                | Chunk overlap in characters          |
| `GIGI_EMBED_BATCH_SIZE`| `32`                                 | Chunks embedded per progress step   |
| `GIGI_LLM`            | `ollama`                             | `ollama` \| `openai` \| `stub`       |
| `GIGI_OLLAMA_URL`     | `http://localhost:11434`             | Ollama server                        |
| `GIGI_OLLAMA_MODEL`   | `llama3.2`                           | Ollama model                         |
| `GIGI_OPENAI_URL/API_KEY/MODEL` | —                          | For OpenAI-compatible endpoints      |
| `GIGI_MAX_ATTEMPTS`   | `2`                                  | Max self-correction retries          |

To try the pipeline without Ollama: `GIGI_LLM=stub gigi ask "..."`.

## Tests

```bash
uv run pytest
uv run ruff check src tests
```

The LangGraph tests run fully offline with fake services and a stub LLM — no
model downloads required.