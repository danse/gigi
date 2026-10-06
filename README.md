# gigi

Local document Q&A assistant built on **PyTorch** + **LangGraph**.

Ask questions over a folder of documents. A pretrained PyTorch sentence-transformer
embeds your docs and query, torch does cosine-similarity retrieval (optional
cross-encoder reranking is off by default — the eval grid showed it demotes the
right document on needle questions; enable with `GIGI_RERANK=1`), and a LangGraph
agent retrieves → grades → generates an answer via a local Ollama LLM (one
generation per turn; no LLM-side "grounding" verification layer).

`gigi index` also clusters the embeddings into topics. `gigi summarise` uses those
clusters' representative passages — several per topic — instead of
nearest-neighbor search, so a large index can still be summarized:

```
gigi index examples/docs     # build the embedding index + topic clusters
gigi ask "How do I deploy the app?"
gigi summarise               # summarize the corpus via its topic clusters
```

## What makes it tick

- **PyTorch retrieval stack** (multilingual; index is often Italian/Spanish/Catalan
  prose, English or Dutch in the docs)
  - Embeddings: `intfloat/multilingual-e5-small` (sentence-transformers; e5
    `query:` / `passage:` prefixes applied on each side, BGE instruction for `bge-*`)
  - Reranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingual
    CrossEncoder; the L6 sibling is no longer public) — *off by default*, kept
    for `GIGI_RERANK=1` (the `--grid` sweep shows it hurting mmr/basis on the
    fixture corpus)
  - Top-k via cosine similarity + MMR, computed with `torch.matmul`
  - Spherical k-means over the index (`clusters.json`, with per-cluster member
    chunks) for `gigi summarise` — each topic contributes its nearest passages,
    so even a journal-like corpus summarizes sensibly
- **LangGraph agent** (`src/gigi/agent/graph.py`)
  - After `retrieve`, a conditional edge sends `overview`-seeded turns (only
    `gigi summarise` takes that branch) to `generate_overview` (per-topic
    representative passages) and everything else through `grade → generate`.
    Grade bails out to a "no answer" node when nothing clears the relevance
    threshold, and `generate` routes there too when the output is empty or the
    canonical refusal (`"I don't know."`). Otherwise answers are accepted
    as-is: the cosine grade is the only relevance gate, and the source list
    printed under every answer is there for the human to verify.
  - Consecutive `gigi ask` calls continue the conversation: question/answer
    turns are stored in a SQLite checkpointer under `GIGI_INDEX_DIR`
    (`checkpoints.sqlite`, thread id in `thread_id`), and replayed into the
    prompt. `--reset` starts over. Declined turns are never remembered.
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
gigi summarise

# Inspect the index
gigi status

# Measure retrieval quality over the committed multilingual golden set (offline)
gigi eval
gigi eval --grid          # sweep retrieval knobs, show the best configs
```

## Configuration (env vars)

| Var                   | Default                              | Description                          |
| --------------------- | ------------------------------------ | ------------------------------------ |
| `GIGI_INDEX_DIR`      | `./.index` (in the current dir)      | Where the embedding index lives; one per directory |
| `GIGI_EMBED_MODEL`    | `intfloat/multilingual-e5-small`    | Sentence-transformer for embeddings (multilingual) |
| `GIGI_RERANK_MODEL`   | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Multilingual cross-encoder reranker (only loaded when enabled) |
| `GIGI_RERANK`         | `0`                                  | Enable the reranker (`1`); off by default — the eval grid showed it demoting the correct document on the fixture |
| `GIGI_TOP_K`          | `16`                                 | Candidates retrieved by MMR, fed to the LLM |
| `GIGI_RERANK_TOP_K`   | `8`                                  | Candidates kept after rerank (only when `GIGI_RERANK=1`) |
| `GIGI_MMR_LAMBDA`     | `0.7`                                | MMR diversity: 1 = pure relevance, 0 = pure diversity |
| `GIGI_N_CLUSTERS`     | `16`                                 | Topic clusters built at index time   |
| `GIGI_OVERVIEW_PER_CLUSTER` | `2`                           | Passages per topic fed to `gigi summarise` |
| `GIGI_GRADE_THRESHOLD`| auto (`0.3` cosine with no reranker; off when reranking) | Relevance cutoff for the kept candidates |
| `GIGI_CHUNK_SIZE`     | `800`                                | Chunk size in characters             |
| `GIGI_CHUNK_OVERLAP`  | `100`                                | Chunk overlap in characters          |
| `GIGI_EMBED_BATCH_SIZE`| `32`                                 | Chunks embedded per progress step   |
| `GIGI_LLM`            | `ollama`                             | `ollama` \| `openai` \| `stub`       |
| `GIGI_OLLAMA_URL`     | `http://localhost:11434`             | Ollama server                        |
| `GIGI_OLLAMA_MODEL`   | `gemma3:270m`                       | Ollama model                         |
| `GIGI_OPENAI_URL/API_KEY/MODEL` | —                          | For OpenAI-compatible endpoints      |
| `GIGI_LLM_TIMEOUT`    | `1800`                               | LLM HTTP read timeout in seconds (`0` waits forever) |

To try the pipeline without Ollama: `GIGI_LLM=stub gigi ask "..."`.

Changing `GIGI_EMBED_MODEL` (or the default) requires a full re-index — the
embeddings live in a different vector space, so `gigi ask` refuses to run
against an index built with another model and prints the rebuild command:

```bash
gigi index .     # full rebuild; also bulk-downloads the new HF models on first run
```

`gigi index <dir>` never reads its own `.index/` output directory (or any
`.index` folder inside the corpus), so a re-run doesn't accumulate its own
index files as documents.

## Evaluation (`gigi eval`)

`gigi eval` measures the retrieval stage with no LLM at all: it embeds the
committed multilingual fixture corpus ([`src/gigi/eval/corpus/`](src/gigi/eval/corpus/),
23 docs across en/it/es/ca) into a throwaway index, runs 20 golden questions
([`src/gigi/eval/golden.py`](src/gigi/eval/golden.py)) through the same
pipeline `ask` uses, and reports:

| Metric | What it measures |
| ------ | ---------------- |
| `recall@8` | Is the answer's source retrieved at all (pre-grade candidates)? |
| `mrr` | How high is the answer's source ranked (post-grade)? |
| `answer-basis` | Is the top kept source the expected one? |
| `bail rate` | How often nothing survives the relevance threshold? |
| `overview cov` | What fraction of the corpus surfaces in the `summarise` passages? |
| `score` | Weighted composite: `0.4·recall@8 + 0.3·mrr + 0.2·basis + 0.1·(1−bail)` for specific cases, `coverage` for overview cases |

```bash
gigi eval                 # baseline report, one line per golden question
gigi eval --grid          # sweep top_k, rerank_top_k, rerank, grade, λ, n_clusters
gigi eval --json          # machine-readable report
gigi eval --min-score 0.8 # exit 1 when the baseline score drops below 0.8
gigi eval --bootstraps 0  # skip the confidence intervals
gigi eval --seed 42       # other RNG seed for the CIs
```

Every metric in the baseline report carries a 95% percentile-bootstrap
confidence interval: the per-case values are resampled with replacement
(`--bootstraps` times, default 2000) and the `alpha/2`–`1−alpha/2` percentiles
of the resampled means are quoted as `[lo, hi]`. With only 20 golden cases the
intervals are wide and coarse — that is the point: it reads as "score 0.824,
95% CI [0.745, 0.900]" instead of a false-precise point estimate. The
`--json` report carries the same intervals under `ci`; a fixed `--seed`
(default 0) makes every run reproducible, and the grid table shows each
config's score interval so near-ties stop looking like real differences.

The grid is fast because the cross-encoder scores every `(question, chunk)`
pair once and each config only re-selects from those fixed scores. The
chunking knobs (`GIGI_CHUNK_SIZE`/`_OVERLAP`) are intentionally excluded from
the sweep — they change the index itself, mixing signals. First run downloads
the two HF models (if not cached); afterwards everything is offline. The
shipped defaults were tuned with `--grid`: reranking is **off by default**
because the sweep showed the cross-encoder demoting the correct document on
this fixture (mean mrr 1.000 off vs 0.644 on, answer-basis 1.000 vs 0.604,
bail 0.228 on). The baseline with the tuned defaults is ~0.965 — 16/16
specific cases perfect, overview coverage ~0.83.

## Tests

```bash
uv run pytest
uv run ruff check src tests
```

The LangGraph tests run fully offline with fake services and a stub LLM — no
model downloads required.