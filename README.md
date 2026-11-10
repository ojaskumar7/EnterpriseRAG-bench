# Enterprise Knowledge Intelligence Platform

A production-oriented RAG platform for searching and reasoning over enterprise knowledge sources with grounded answers, citations, and explicit abstention.

This project is being built incrementally as an engineering learning project. The focus is on retrieval quality, evaluation, reliability, and backend architecture rather than hiding the system behind a high-level RAG framework.

## Current Status

**Blocks 1, 2, and 3 are complete. The ingestion pipeline, dense retrieval stack, and
PostgreSQL FTS + hybrid retrieval stack are implemented, verified on the real subset,
and ready for Block 4.**

The approved plan is in [`plan.md`](plan.md). The project currently provides:

- Reproducible dataset subset acquisition and manifest generation.
- Zip and parquet document loaders.
- Deterministic normalization with content hashing.
- Token-aware chunking with metadata and BGE tokenization.
- A validated corpus of 34,378 documents and 136,383 chunks.
- BGE embeddings loaded into PostgreSQL with pgvector.
- An idempotent embedding/load pipeline and an HNSW cosine index.
- Dense retrieval with a leakage-safe interface and development recall/MRR metrics.
- PostgreSQL full-text lexical retrieval with a generated `search_vector` and GIN index.
- Hybrid retrieval with Reciprocal Rank Fusion (RRF) over dense and lexical candidates.
- A unified development evaluator for dense, lexical, and hybrid modes.
- Unit, integration, and real-subset tests.

The generated corpus is in `data/processed/chunks.jsonl` (gitignored). The designs are
documented in [`docs/architecture/ingestion.md`](docs/architecture/ingestion.md),
[`docs/architecture/embeddings-and-retrieval.md`](docs/architecture/embeddings-and-retrieval.md),
and [`docs/architecture/hybrid-retrieval.md`](docs/architecture/hybrid-retrieval.md).

Exact BM25, reranking, generation, the full benchmark evaluation harness,
authentication, and the API are deliberately deferred to later blocks.

## Problem

Organizations keep knowledge in documents, emails, tickets, wikis, meeting transcripts, and internal discussions. Keyword search cannot reliably handle semantic questions, while general-purpose LLMs can produce answers that are not supported by the available evidence.

This platform aims to:

1. Ingest and organize enterprise documents.
2. Isolate documents by user and knowledge base.
3. Retrieve evidence with dense, sparse, hybrid, and reranked strategies.
4. Generate answers grounded in retrieved context.
5. Cite source documents and chunks.
6. Abstain when the evidence is insufficient.
7. Support multi-document and conflicting-information questions.
8. Benchmark retrieval and generation configurations reproducibly.

## Approved Baseline Decisions

| Decision | Choice |
|---|---|
| Dataset | Official [`onyx-dot-app/EnterpriseRAG-Bench`](https://github.com/onyx-dot-app/EnterpriseRAG-Bench) |
| Block 1 corpus | 10 release slices for Confluence, Jira, GitHub, and HubSpot |
| Block 1 documents | 34,378 document members; 34,375 unique `doc_id` values |
| Block 1 chunks | 136,383 chunks from 34,378 documents |
| Initial embedding model | `BAAI/bge-large-en-v1.5` (1024 dimensions) |
| Embedding runtime | PyTorch 2.8.0 (CUDA 12.8), `float16` for the measured bulk load |
| Vector store | PostgreSQL 16 + pgvector, HNSW cosine index |
| Chunking baseline | 400 tokens, 64-token overlap, 50-token minimum |
| Evaluation split | Stratified 80/20 development/holdout, seed 42 |

The `SJChen02/EnterpriseRAG-Bench` Hugging Face repository is a mirror. The official repository is used for provenance.

## Target Architecture

```text
Document source
  -> Loader
  -> Normalizer
  -> Chunker
  -> EmbeddingProvider
  -> PostgreSQL documents/chunks + pgvector
  -> PostgreSQL full-text search index (GIN)

User question
  -> optional query rewriting/decomposition
  -> dense retriever + sparse retriever
  -> Reciprocal Rank Fusion
  -> cross-encoder reranker
  -> context builder
  -> Gemini generation
  -> answer + citations + abstention result
```

The backend is a modular monolith with independent functional modules:

- **API:** FastAPI routes, request validation, authentication, and response schemas.
- **Ingestion:** Loaders, preprocessing, chunking, embeddings, and indexing.
- **Retrieval:** Dense, sparse, hybrid, RRF, reranking, query rewriting, and decomposition.
- **Generation:** Context construction, prompts, Gemini integration, citations, and abstention.
- **Evaluation:** Retrieval metrics, generation metrics, benchmark execution, and result storage.
- **Monitoring:** Structured logs, latency, token usage, cost, and error tracking.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Backend | FastAPI |
| LLM | Google Gemini |
| Embeddings | BGE (`bge-large-en-v1.5` initially), optional Gemini embeddings |
| Vector database | PostgreSQL + pgvector |
| Sparse retrieval | PostgreSQL full-text search (GIN); exact BM25 later if justified |
| Fusion | Reciprocal Rank Fusion |
| Reranking | BGE cross-encoder |
| Orchestration | LangChain, used minimally at the LLM boundary |
| Evaluation | RAGAS for local development plus the official EnterpriseRAG-Bench protocol |
| Monitoring | Structured logging first; Langfuse later |
| Authentication | JWT, added in a later block |
| Caching | Redis, added when needed |
| File storage | MinIO/S3, added when needed |
| Frontend | React/Next.js, added after the API is stable |
| Containerization | Docker and Docker Compose |

### Stack notes

- Retrieval remains independent of the LLM provider.
- Embedding generation remains independent of retrieval implementation.
- Only one embedding model and dimension should be used per vector index.
- PostgreSQL full-text search is not BM25.
- Many cross-encoders cap input at 512 tokens, so chunk size must remain compatible.
- Langfuse self-hosting is operationally heavy and is deferred.

## Dataset

EnterpriseRAG-Bench simulates an enterprise called Redwood Inference and contains:

- **511,962 documents**
- **500 benchmark questions**
- **9 source types:** Slack, Gmail, Linear, Google Drive, HubSpot, Fireflies, GitHub, Jira, and Confluence
- **10 question categories**, including basic, semantic, multi-document, conflicting-information, high-level, and information-not-found questions

The dataset has two configurations with a single `test` split each:

- `documents`: `doc_id`, `source_type`, `title`, `content`
- `questions`: `question_id`, `question_type`, `source_types`, `question`, `expected_doc_ids`, `gold_answer`, `answer_facts`

Documents are UTF-8 text exports, not PDFs. The Hugging Face parquet has four core columns. The GitHub repository also contains richer source-specific JSON metadata, which is not present in the parquet.

### Evaluation caveats

- The dataset has no development split. This project uses an internal stratified 80/20 split. The holdout set must remain untouched until final evaluation.
- `source_types`, `expected_doc_ids`, `question_type`, and `gold_answer` are evaluation metadata. They must never be passed to the retriever.
- `high_level` and `info_not_found` questions have no gold documents and require separate abstention evaluation.
- The official benchmark uses an LLM judge and different metrics from RAGAS. Local RAGAS results and official benchmark results must be labeled separately.
- The dataset is synthetic. It does not validate PDF/OCR ingestion, tenant isolation, or document-version comparison.

### Block 1 subset and measured results

Block 1 downloads every release slice for Confluence, Jira, GitHub, and HubSpot: 10
zip slices and 84.5 MiB. The measured ingestion run produced:

| Metric | Value |
|---|---:|
| Document members | 34,378 |
| Unique `doc_id` values | 34,375 |
| Chunks | 136,383 |
| Unique chunk IDs | 136,383 |
| Documents chunked | 34,378 |
| Documents skipped or failed | 0 |
| Empty documents | 0 |
| Malformed document IDs | 0 |

| Source | Documents | Chunks |
|---|---:|---:|
| Confluence | 5,189 | 41,190 |
| GitHub | 8,052 | 29,797 |
| HubSpot | 15,017 | 39,059 |
| Jira | 6,120 | 26,337 |

Question selection produced 211 eligible retrieval questions: 169 development, 42
holdout, and 30 abstention. The remaining 259 questions require at least one source
outside the Block 1 subset. No eligible question had a missing gold document.

The chunk corpus has 400-token maximum chunks with a median of 3 chunks per document.
Two content hashes each appear twice across different documents, and three document
members share an existing `doc_id`. Document storage in later blocks must therefore
use `(source_type, doc_id)` as the document identity, while retrieval evaluation
should deduplicate by `doc_id`.

See [`docs/architecture/ingestion.md`](docs/architecture/ingestion.md) for the full
EDA summary and pipeline design.

## Repository Layout

Implemented Blocks 1-3 structure:

```text
productionRAG/
├── backend/
│   ├── app/
│   │   ├── config.py
│   │   ├── settings.py
│   │   ├── ingestion/
│   │   │   ├── models.py
│   │   │   ├── manifest.py
│   │   │   ├── pipeline.py
│   │   │   ├── loaders/
│   │   │   ├── preprocessing/
│   │   │   └── chunking/
│   │   ├── embeddings/
│   │   │   ├── base.py
│   │   │   ├── bge.py
│   │   │   ├── fake.py
│   │   │   └── loader.py
│   │   ├── db/
│   │   │   ├── base.py
│   │   │   ├── models.py
│   │   │   ├── repositories.py
│   │   │   └── session.py
│   │   ├── evaluation/
│   │   │   └── dev.py
│   │   └── retrieval/
│   │       ├── dense.py
│   │       ├── fusion.py
│   │       ├── hybrid.py
│   │       ├── lexical.py
│   │       ├── metrics.py
│   │       └── types.py
│   ├── alembic.ini
│   ├── alembic/
│   └── tests/
│       ├── unit/
│       ├── integration/
│       └── fixtures/
├── scripts/
│   ├── download_subset.py
│   ├── build_subset_manifest.py
│   ├── ingest_subset.py
│   ├── check_embedding_runtime.py
│   ├── embed_and_load.py
│   ├── dense_search.py
│   ├── eval_dense_dev.py
│   ├── lexical_search.py
│   ├── hybrid_search.py
│   └── eval_retrieval_dev.py
├── data/
│   ├── raw/          # gitignored
│   ├── processed/    # gitignored
│   └── sample/       # committed question splits and manifest
├── docs/
│   ├── architecture/
│   │   ├── ingestion.md
│   │   ├── embeddings-and-retrieval.md
│   │   └── hybrid-retrieval.md
│   └── design-decisions/
│       ├── 0001-dataset-subset.md
│       ├── 0002-vector-store-and-embedding-runtime.md
│       └── 0003-lexical-backend-and-fusion.md
├── compose.yml
├── .env.example
├── pyproject.toml
├── uv.lock
├── .gitignore
├── README.md
└── plan.md
```

Future modules will be added only when their implementation block begins.

## Getting Started

### Prerequisites

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- Git
- Docker and Docker Compose (for PostgreSQL/pgvector)

### Installation

```bash
uv sync
```

### Block 1 workflow

```bash
# Download the approved release slices and questions
uv run python scripts/download_subset.py

# Build the question manifest and splits
uv run python scripts/build_subset_manifest.py

# Ingest and chunk the subset
uv run python scripts/ingest_subset.py --raw-dir data/raw --output-dir data/processed
```

For a quick development run, add `--sources confluence --max-docs 20`.

Outputs:

- `data/raw/`: downloaded slices and questions (gitignored)
- `data/processed/chunks.jsonl`: 136,383-chunk corpus, approximately 242 MiB (gitignored)
- `data/processed/stats.json`: deterministic ingestion statistics (gitignored)
- `data/processed/eda.json`: generated EDA statistics (gitignored)
- `data/sample/*.jsonl`: committed question splits
- `data/sample/subset_manifest.json`: committed manifest

### Block 2 workflow

```bash
# Start PostgreSQL with pgvector
docker compose up -d db

# Apply the schema and HNSW index
uv run alembic -c backend/alembic.ini upgrade head

# Check the CUDA/model runtime before the full load
uv run python scripts/check_embedding_runtime.py --device cuda --dtype float16

# Embed and load all 136,383 chunks (idempotent; reruns skip existing embeddings)
uv run python scripts/embed_and_load.py \
  --chunks data/processed/chunks.jsonl \
  --device cuda \
  --dtype float16 \
  --batch-size 128

# Query the index
uv run python scripts/dense_search.py "How do I rotate an API key?" \
  --top-k 5 --device cuda --dtype float16

# Evaluate only the development split
uv run python scripts/eval_dense_dev.py --top-k 20 --device cuda --dtype float16
```

The development metrics file is written to
`data/processed/dense_dev_metrics.json` (gitignored). Do not run the holdout split
during development.

### Block 3 workflow

```bash
# Start PostgreSQL and apply the Block 3 lexical schema (generated search_vector + GIN)
docker compose up -d db
uv run alembic -c backend/alembic.ini upgrade head

# Optional: refresh planner statistics after the migration
docker exec production-rag-db psql -U rag -d rag -c "VACUUM ANALYZE chunks;"

# Query the PostgreSQL full-text index
uv run python scripts/lexical_search.py "How do I rotate an API key?" --top-k 5

# Query dense + lexical with Reciprocal Rank Fusion
uv run python scripts/hybrid_search.py "How do I rotate an API key?" \
  --top-k 5 --device cuda --dtype float16

# Evaluate dense, lexical, and hybrid on the 169 development questions
uv run python scripts/eval_retrieval_dev.py \
  --mode all --top-k 20 --candidate-k 50 --rrf-k 60 \
  --device cuda --dtype float16
```

The unified metrics file is written to
`data/processed/retrieval_dev_metrics.json` (gitignored). The lexical backend is
PostgreSQL full-text search with `ts_rank_cd`, not exact BM25. Do not run the holdout
split during development.

### Testing

```bash
uv run pytest -q
uv run pytest -q -m integration
uv run ruff check .
uv run ruff format --check .
```

### Block 1 verification evidence

- `uv run pytest -q`: 56 passed.
- `uv run pytest -q -m integration`: real-subset smoke passed; 55 deselected.
- `uv run ruff check .`: passed.
- `uv run ruff format --check .`: 26 files already formatted.
- Chunk corpus: 136,383 chunks, 136,383 unique chunk IDs, maximum 400 tokens.
- Ingestion counts: 34,378 loaded, 34,378 chunked, 0 skipped, 0 duplicate, 0 failed.
- Question coverage: 0 missing gold documents across 169 development and 42 holdout questions.
- Answerability: all 30 abstention questions have no expected documents; all 259 excluded questions require an unsupported source type.
- Git: no raw or processed dataset files are tracked.

### Block 2 verification evidence

- Full load: 34,378 documents and 136,383 chunks, all with 1024-dimensional embeddings.
- Unique chunk IDs: 136,383; orphan chunks: 0; wrong embedding dimensions: 0.
- HNSW cosine index exists and is used by an `ORDER BY embedding <=> ... LIMIT` plan.
- Loader rerun: 0 rows written, 0 chunks embedded, 136,383 chunks skipped.
- Development metrics: recall@1 0.2840, recall@5 0.4643, recall@10 0.5362,
  recall@20 0.5836, MRR 0.4330 over 169 questions.
- `uv run pytest -q`: 93 passed.
- `uv run ruff check .`: passed.
- `uv run ruff format --check .`: passed.
- Holdout set: not used.

### Block 3 verification evidence

- Generated `search_vector` exists on `chunks` with title weight `A` and content weight `B`.
- GIN index `ix_chunks_search_vector` exists (about 75 MB) and lexical queries use it
  (`Bitmap Index Scan`).
- `LexicalRetriever`, `HybridRetriever`, and RRF are implemented with deterministic
  tie-breaking.
- `scripts/lexical_search.py`, `scripts/hybrid_search.py`, and
  `scripts/eval_retrieval_dev.py` work.
- Development metrics cover 169 questions for dense, lexical, and hybrid modes.
- Hybrid recall@10 and MRR exceed dense on the clean database.
- Final database state: 34,378 documents, 136,383 chunks, 136,383 embeddings, 0
  non-default rows.
- Holdout set: not used.

### Block 1 retrospective

- The release data contains duplicate `doc_id` values, so composite document identity is required. This was discovered from the real data rather than assumed during planning.
- Tokenizer offsets made character-preserving chunks straightforward, while balanced tail handling avoided sub-minimum trailing chunks for configs where the minimum exceeds the overlap.
- Atomic temporary-file writes plus sorted inputs and JSON keys produced byte-identical reruns.
- The fixed 400/64/50 chunking baseline is now measured but not tuned. Tuning belongs after retrieval evaluation exists.
- Block 2 started with the embedding provider and dense retrieval over the generated chunk corpus.

## Block 2 measured results

Block 2 embedded the full Block 1 corpus with `BAAI/bge-large-en-v1.5` and loaded it
into PostgreSQL/pgvector:

| Metric | Value |
|---|---:|
| Documents | 34,378 |
| Chunks | 136,383 |
| Non-null embeddings | 136,383 |
| Embedding dimensions | 1024 |
| Unique chunk IDs | 136,383 |
| Orphan chunks | 0 |
| HNSW index | present, cosine, `m=16`, `ef_construction=64` |
| `chunks` total size | 2.0 GB |
| HNSW index size | 1.1 GB |
| Full load time | about 46 minutes |
| Rerun after load | 0 written, 0 embedded, 136,383 skipped |

Development retrieval metrics on the 169 development questions (top-20, `float16`):

| Metric | Value |
|---|---:|
| recall@1 | 0.2840 |
| recall@5 | 0.4643 |
| recall@10 | 0.5362 |
| recall@20 | 0.5836 |
| MRR | 0.4330 |

These are diagnostics, not benchmark claims. The 42-question holdout set remains
untouched.

The local GPU thermally throttles under sustained load, so the bulk load used
`float16` weights. A 100-chunk comparison against `float32` produced a mean cosine
similarity of 1.0001 and a minimum of 0.9998. Pass `--dtype float16` when querying this
index.

## Block 3 measured results

Block 3 added a generated PostgreSQL `search_vector` column, a GIN index, a lexical
retriever, and hybrid RRF fusion. The final evaluation ran on a clean database
(`34,378` documents, `136,383` chunks, `136,383` embeddings, `0` non-default rows) over
the 169 development questions with `top_k=20`, `candidate_k=50`, `rrf_k=60`, and equal
weights:

| Mode | recall@1 | recall@5 | recall@10 | recall@20 | MRR |
|---|---:|---:|---:|---:|---:|
| dense | 0.2840 | 0.4761 | 0.5303 | 0.5777 | 0.4345 |
| lexical | 0.0118 | 0.0178 | 0.0178 | 0.0178 | 0.0178 |
| hybrid | 0.2929 | 0.4726 | 0.5348 | 0.5821 | 0.4404 |

Hybrid recall@10 (+0.0045) and MRR (+0.0059) exceed dense, so the Block 3 acceptance
criterion passes. Hybrid recall@5 is 0.0035 below dense, a non-binding trade-off.
Dense results versus the Block 2 baseline are mixed rather than uniformly better (for
example recall@10 0.5303 vs 0.5362).

The lexical path is effectively non-functional on this development split: 156 of 169
questions return zero lexical results because `websearch_to_tsquery` ANDs the terms of a
natural-language question. Hybrid gains therefore come from the minority of questions
where lexical returns matches. This is a documented limitation, not a hidden success.

These are diagnostics, not benchmark claims. The 42-question holdout set remains
untouched.

## Development Roadmap

| Block | Focus | Status |
|---|---|---|
| 1 | Project setup, dataset subset, loading, normalization, chunking | complete |
| 2 | Embeddings, PostgreSQL, pgvector, dense retrieval | complete |
| 3 | PostgreSQL FTS, hybrid retrieval, Reciprocal Rank Fusion | complete |
| 4 | Evidence selection and context construction | planned |
| 5 | Cross-encoder reranking and retrieval optimization | planned |
| 6 | Gemini integration, citations, and grounded generation | planned |
| 7 | Evaluation framework and retrieval benchmarking | planned |
| 8 | Hallucination control, abstention, multi-document reasoning | planned |
| 9 | FastAPI, authentication, and multi-tenancy | planned |
| 10 | Asynchronous ingestion and document lifecycle | planned |
| 11 | Observability, frontend, Docker, and deployment | planned |

## Engineering Principles

- Grounded answers take priority over fluent answers.
- Retrieval relevance, answer faithfulness, and answer correctness are distinct.
- High similarity scores do not guarantee correct answers.
- Evaluation must be reproducible and must not fabricate results.
- Retrieved content is untrusted input and must be treated accordingly.
- Tenant isolation is enforced in code and database queries, not only in the UI.
- External services are kept behind clean interfaces.
- Dependencies are introduced only when they solve a concrete problem.

## Known Limitations

- Block 3 stops at hybrid retrieval. There is no exact BM25, reranker, evidence selection, or generated answer yet.
- The lexical backend is PostgreSQL full-text search (`ts_rank_cd`), not exact BM25.
- Lexical retrieval is effectively non-functional on the development split: 156 of 169 questions return zero results because `websearch_to_tsquery` ANDs natural-language terms.
- Hybrid gains over dense are small and hybrid recall@5 is slightly below dense.
- Block 1 uses a four-source development subset, not the full 511,962-document corpus.
- PDF and OCR ingestion are not implemented or validated by this dataset.
- The corpus has 34,378 document members but 34,375 unique `doc_id` values. The database uses content-addressed identity `(tenant_id, knowledge_base_id, source_type, doc_id, content_hash)` to preserve duplicate IDs.
- Metadata-aware retrieval is not implemented.
- Document-version comparison is not supported by the dataset and requires a separate design.
- Chunking is fixed-size and token-aware, not heading-aware or semantic.
- Chunking defaults are a baseline and have not been experimentally tuned.
- Dense, lexical, and hybrid retrieval have development-split diagnostics only; there are no official benchmark results yet.
- English stemming and stopword removal are fixed by the PostgreSQL `english` text search configuration.
- RRF uses ranks only and discards score magnitude; candidate depth (`candidate_k`) bounds what fusion can recover.
- The measured index was built with `float16` weights because the laptop GPU thermally throttles; queries must use the same dtype.
- No multi-tenant authentication or authorization exists yet.

## Documentation

- [`plan.md`](plan.md): detailed Block 1-3 implementation plan.
- [`docs/architecture/ingestion.md`](docs/architecture/ingestion.md): pipeline, loaders, normalization, chunking, and measured statistics.
- [`docs/architecture/embeddings-and-retrieval.md`](docs/architecture/embeddings-and-retrieval.md): embeddings, pgvector schema, loading, dense retrieval, and measured results.
- [`docs/architecture/hybrid-retrieval.md`](docs/architecture/hybrid-retrieval.md): lexical schema, PostgreSQL FTS query flow, RRF fusion, and measured hybrid results.
- [`docs/design-decisions/0001-dataset-subset.md`](docs/design-decisions/0001-dataset-subset.md): dataset, split, chunking, and data-quality decisions.
- [`docs/design-decisions/0002-vector-store-and-embedding-runtime.md`](docs/design-decisions/0002-vector-store-and-embedding-runtime.md): pgvector, BGE, PyTorch/CUDA, and dtype decisions.
- [`docs/design-decisions/0003-lexical-backend-and-fusion.md`](docs/design-decisions/0003-lexical-backend-and-fusion.md): PostgreSQL FTS vs exact BM25, RRF, and measured results.
- `docs/evaluation/`: evaluation methodology and results, added later.

## License and Attribution

The EnterpriseRAG-Bench dataset and official tooling are released under the MIT license. This project will include attribution and the dataset citation when benchmark results are reported.

The dataset contains a benchmark canary string and should not be included in model training corpora.
