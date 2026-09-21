# gigi

Local document Q&A assistant built on **PyTorch** + **LangGraph**.

Ask questions over a folder of documents. A pretrained PyTorch sentence-transformer
embeds your docs and query, torch does cosine-similarity retrieval, a PyTorch
cross-encoder reranks the candidates, and a LangGraph agent retrieves → grades →
generates a grounded answer (with a self-correction loop) via a local Ollama LLM.

```
gigi index examples/docs     # build the embedding index
gigi ask "How do I deploy the app?"
```

## What makes it tick

- **PyTorch retrieval stack**
  - Embeddings: `BAAI/bge-small-en-v1.5` (sentence-transformers)
  - Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (CrossEncoder)
  - Top-k via cosine similarity computed with `torch.matmul`
- **LangGraph agent** (`src/gigi/agent/graph.py`)
  - `retrieve → grade → generate`, with a conditional edge that bails out to a
    "no answer" node when nothing clears the relevance threshold, and a retry
    edge that re-generates (max `GIGI_MAX_ATTEMPTS` times) when grounding fails.
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

# Ask a question (answer + cited sources)
gigi ask "What is the deployment process?"

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
| `GIGI_TOP_K`          | `6`                                  | Candidates retrieved before rerank   |
| `GIGI_RERANK_TOP_K`   | `3`                                  | Candidates kept after rerank         |
| `GIGI_GRADE_THRESHOLD`| auto (`1.0` rerank / `0.4` cosine)   | Relevance cutoff for grading         |
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