# Enterprise Knowledge Intelligence Platform

A production-oriented RAG platform for searching and reasoning over enterprise knowledge sources with grounded answers, citations, and explicit abstention.

This project is being built incrementally as an engineering learning project. The focus is on retrieval quality, evaluation, reliability, and backend architecture rather than hiding the system behind a high-level RAG framework.

## Current Status

**Block 1 planning is complete and approved. Application code has not been implemented yet.**

The approved Block 1 plan is in [`plan.md`](plan.md). The first implementation block covers:

- Reproducible dataset subset acquisition.
- Document loading and normalization.
- Token-aware chunking with metadata.
- A validated chunk corpus and EDA report.
- Unit, integration, and data-quality tests.

Embeddings, PostgreSQL/pgvector, retrieval, generation, evaluation, authentication, and the API are deliberately deferred to later blocks.

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
| Dataset | Official [`onyx-dot-app/EnterpriseRAG-Bench`](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench) |
| Block 1 corpus | All release slices for Confluence, Jira, GitHub, and HubSpot (~34k docs) |
| Initial embedding model | `BAAI/bge-large-en-v1.5` (1024 dimensions) |
| Chunking baseline | 400 tokens, 64-token overlap, 50-token minimum |
| Evaluation split | Stratified 80/20 development/holdout, fixed seed |

The `SJChen02/EnterpriseRAG-Bench` Hugging Face repository is a mirror. The official repository is used for provenance.

## Target Architecture

```text
Document source
  -> Loader
  -> Normalizer
  -> Chunker
  -> EmbeddingProvider
  -> PostgreSQL documents/chunks + pgvector
  -> BM25 index

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
| Sparse retrieval | BM25 (`bm25s` initially; ParadeDB/OpenSearch optional later) |
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

## Repository Layout

Planned Block 1 structure:

```text
productionRAG/
├── backend/
│   ├── app/
│   │   ├── config.py
│   │   └── ingestion/
│   │       ├── models.py
│   │       ├── loaders/
│   │       ├── preprocessing/
│   │       ├── chunking/
│   │       └── pipeline.py
│   └── tests/
│       ├── unit/
│       ├── integration/
│       └── fixtures/
├── scripts/
│   ├── download_subset.py
│   ├── build_subset_manifest.py
│   └── ingest_subset.py
├── data/
│   ├── raw/          # gitignored
│   ├── processed/    # gitignored
│   └── sample/       # tiny committed fixtures and manifests
├── docs/
│   ├── architecture/
│   └── design-decisions/
├── pyproject.toml
├── uv.lock
├── .env.example
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
- Docker (for later database and infrastructure blocks)

### Installation

The project is not implemented yet. Once Block 1 Task 1 is complete, the setup will be:

```bash
uv sync
```

### Block 1 workflow

After the Block 1 scripts are implemented:

```bash
# Download the approved release slices and questions
uv run python scripts/download_subset.py

# Build the question manifest and splits
uv run python scripts/build_subset_manifest.py

# Ingest and chunk the subset
uv run python scripts/ingest_subset.py --raw-dir data/raw --output-dir data/processed
```

Outputs:

- `data/raw/`: downloaded slices and questions (gitignored)
- `data/processed/chunks.jsonl`: chunk corpus (gitignored)
- `data/processed/stats.json`: ingestion statistics (gitignored)
- `data/sample/*.jsonl`: committed question splits
- `data/sample/subset_manifest.json`: committed manifest

### Testing

```bash
uv run pytest -q
uv run pytest -q -m integration
uv run ruff check .
uv run ruff format --check .
```

## Development Roadmap

| Block | Focus |
|---|---|
| 1 | Project setup, dataset subset, loading, normalization, chunking |
| 2 | Embeddings, PostgreSQL, pgvector, dense retrieval |
| 3 | BM25, hybrid retrieval, Reciprocal Rank Fusion |
| 4 | Cross-encoder reranking and retrieval optimization |
| 5 | Gemini integration, context construction, citations |
| 6 | Evaluation framework and retrieval benchmarking |
| 7 | Hallucination control, abstention, multi-document reasoning |
| 8 | FastAPI, authentication, and multi-tenancy |
| 9 | Asynchronous ingestion and document lifecycle |
| 10 | Observability, frontend, Docker, and deployment |

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

- No application code exists yet.
- Block 1 uses a four-source development subset, not the full 511,962-document corpus.
- PDF and OCR ingestion are not implemented or validated by this dataset.
- Metadata-aware retrieval is not implemented.
- Document-version comparison is not supported by the dataset and requires a separate design.
- Chunking defaults are a baseline and have not been experimentally tuned.
- No retrieval or generation benchmark results have been produced.
- No multi-tenant authentication or authorization exists yet.

## Documentation

- [`plan.md`](plan.md): detailed Block 1 implementation plan.
- `docs/architecture/`: system and ingestion design.
- `docs/design-decisions/`: architectural decision records.
- `docs/evaluation/`: evaluation methodology and results, added later.

## License and Attribution

The EnterpriseRAG-Bench dataset and official tooling are released under the MIT license. This project will include attribution and the dataset citation when benchmark results are reported.

The dataset contains a benchmark canary string and should not be included in model training corpora.
