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

- **PyTorch retrieval stack** (multilingual; index is often Italian/Spanish/Catalan
  prose, English or Dutch in the docs)
  - Embeddings: `intfloat/multilingual-e5-small` (sentence-transformers; e5
    `query:` / `passage:` prefixes applied on each side, BGE instruction for `bge-*`)
  - Reranker: `cross-encoder/mmarco-mMiniLMv2-L6-H384-v1` (multilingual CrossEncoder)
  - Top-k via cosine similarity + MMR, computed with `torch.matmul`
  - Spherical k-means over the index (`clusters.json`) for corpus-overview questions
    (detection speaks English, Italian, Spanish and Catalan)
- **LangGraph agent** (`src/gigi/agent/graph.py`)
  - After `retrieve`, a conditional edge sends overview questions to
    `generate_overview` (cluster representatives) and everything else through
    `grade → generate`. Grade bails out to a "no answer" node when nothing
    clears the relevance threshold. Generate retries (max `GIGI_MAX_ATTEMPTS`)
    when grounding fails.
  - Consecutive `gigi ask` calls continue the conversation: grounded Q&A pairs
    are stored in a SQLite checkpointer under `GIGI_INDEX_DIR` (`checkpoints.sqlite`,
    thread id in `thread_id`), and replayed into the prompt. `--reset` starts over.
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
ollama pull gemma3:270m
```

## Usage

```bash
# Index a folder of .md/.txt/.rst/.pdf files
gigi index examples/docs

# Rebuild topic clusters only (keeps embeddings; uses GIGI_N_CLUSTERS)
gigi index --recluster

# Ask a question (answer + cited sources)
gigi ask "What is the deployment process?"

# Continue the conversation: consecutive asks remember earlier Q&A pairs
gigi ask "And how do I roll it back?"
gigi ask --reset "Start a fresh conversation"

# Summarize the index (uses topic clusters, not query nearest-neighbors)
gigi ask "what are these documents about?"

# Inspect the index
gigi status
```

## Configuration (env vars)

| Var                   | Default                              | Description                          |
| --------------------- | ------------------------------------ | ------------------------------------ |
| `GIGI_INDEX_DIR`      | `./.index` (in the current dir)      | Where the embedding index lives; one per directory |
| `GIGI_EMBED_MODEL`    | `intfloat/multilingual-e5-small`    | Sentence-transformer for embeddings (multilingual) |
| `GIGI_RERANK_MODEL`   | `cross-encoder/mmarco-mMiniLMv2-L6-H384-v1` | Multilingual cross-encoder reranker |
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
| `GIGI_OLLAMA_MODEL`   | `gemma3:270m`                       | Ollama model                         |
| `GIGI_OPENAI_URL/API_KEY/MODEL` | —                          | For OpenAI-compatible endpoints      |
| `GIGI_MAX_ATTEMPTS`   | `2`                                  | Max self-correction retries          |
| `GIGI_LLM_TIMEOUT`    | `1800`                               | LLM HTTP read timeout in seconds (`0` waits forever) |

To try the pipeline without Ollama: `GIGI_LLM=stub gigi ask "..."`.

Changing `GIGI_EMBED_MODEL` (or the default) requires a full re-index — the
embeddings live in a different vector space, so `gigi ask` refuses to run
against an index built with another model and prints the rebuild command:

```bash
gigi index .     # full rebuild; also bulk-downloads the new HF models on first run
```

## Tests

```bash
uv run pytest
uv run ruff check src tests
```

The LangGraph tests run fully offline with fake services and a stub LLM — no
model downloads required.