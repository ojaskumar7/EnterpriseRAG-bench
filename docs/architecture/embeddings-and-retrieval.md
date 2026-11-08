# Embeddings and Dense Retrieval Architecture

- Status: implemented for Block 2
- Scope: BGE embeddings, pgvector storage, and dense retrieval
- Last measured run: 2026-11-07

## Scope

Block 2 turns the Block 1 chunk corpus into a searchable dense index. It covers the
embedding provider, PostgreSQL/pgvector schema, idempotent loading, dense retrieval,
and a development-split retrieval sanity check.

It deliberately excludes BM25, hybrid retrieval, reranking, generation, the API, and
the full benchmark evaluation harness.

## Pipeline Flow

```text
data/processed/chunks.jsonl
  -> batch reader
  -> document identity upsert
  -> embedded-chunk skip check
  -> BGE document embeddings
  -> chunks upsert with vector(1024)
  -> PostgreSQL + pgvector
  -> HNSW cosine index
  -> DenseRetriever
  -> development recall@k / MRR
```

The loader lives in `backend/app/embeddings/loader.py`; the CLI wrapper is
`scripts/embed_and_load.py`. The retriever lives in
`backend/app/retrieval/dense.py`; the query CLI is `scripts/dense_search.py`.

## Embedding Provider

`EmbeddingProvider` is a small protocol:

```python
class EmbeddingProvider(Protocol):
    model_name: str
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
```

`BGEEmbeddingProvider` uses `BAAI/bge-large-en-v1.5` through sentence-transformers.
It lazily loads the model, resolves `auto`/`cuda`/`cpu` device selection, validates the
model dimension, normalizes embeddings, and applies the BGE query instruction only to
queries. `FakeEmbeddingProvider` is deterministic and exists for tests and dry runs.

The provider also supports `float32` and `float16` weights. PyTorch is pinned to
`2.8.0` (CUDA 12.8) because the CUDA 13.2 build segfaulted on the local WSL driver.
The measured full load used `float16` on CUDA; a 100-chunk comparison produced a mean
cosine similarity of 1.0001 and a minimum of 0.9998 against `float32`.

## Data Model

### documents

| Column | Type | Notes |
|---|---|---|
| `id` | UUID | Generated primary key |
| `tenant_id` | UUID | Multi-tenant placeholder |
| `knowledge_base_id` | UUID | Knowledge-base placeholder |
| `doc_id` | text | Release document ID |
| `source_type` | text | Confluence, Jira, GitHub, or HubSpot |
| `title` | text | Document title |
| `content_hash` | text | Normalized document SHA256 |
| `doc_metadata` | JSONB | Source metadata |

Unique identity:
`(tenant_id, knowledge_base_id, source_type, doc_id, content_hash)`.

### chunks

| Column | Type | Notes |
|---|---|---|
| `id` | UUID | Generated primary key |
| `document_id` | UUID | FK to `documents.id` |
| `doc_id` | text | Denormalized for retrieval and evaluation |
| `tenant_id` | UUID | Tenant filter |
| `knowledge_base_id` | UUID | Knowledge-base filter |
| `chunk_id` | text | Block 1 deterministic ID |
| `chunk_index` | integer | Document order |
| `content`, `title` | text | Retrieval payload |
| `token_count`, offsets | integer | Provenance |
| `content_hash` | text | Chunk SHA256 |
| `doc_metadata` | JSONB | Source metadata |
| `embedding` | `vector(1024)` | Nullable until embedded |
| `embedding_model` | text | Model provenance |

Unique identity: `(tenant_id, knowledge_base_id, chunk_id)`. The HNSW index uses
cosine distance (`vector_cosine_ops`).

## Loading

The loader streams the JSONL corpus in batches, usually 256 chunks per database
transaction. For each batch it:

1. Deduplicates document identities within the batch.
2. Upserts document rows and maps the natural key to the generated document UUID.
3. Queries which chunk IDs already have embeddings.
4. Embeds only new or previously unembedded chunks.
5. Upserts chunks with the embedding and model name.
6. Commits the batch.

A second run over the same corpus adds no rows and skips already embedded chunks.
The loader also supports `--limit` for smoke runs.

The measured full run used 128-chunk embedding batches and 256-chunk database
transactions. The HNSW index remained enabled during the load.

## Dense Retrieval

`DenseRetriever.retrieve(query, top_k=10, source_types=None)` embeds the query with the
BGE query instruction, runs a pgvector cosine search, and returns chunk ID, document
ID, source type, title, chunk index, content, score, and metadata.

The interface accepts only the query text, a result limit, and optional user-selected
source filters. It cannot accept `expected_doc_ids`, `question_type`, `gold_answer`, or
the evaluation question's `source_types` field.

## Development Metrics

`scripts/eval_dense_dev.py` reads `data/sample/dev_questions.jsonl`. For each question
it retrieves top-k using only the question text, deduplicates retrieved chunks by
`doc_id`, and compares the document list with `expected_doc_ids` after retrieval.

Reported metrics:

- recall@1
- recall@5
- recall@10
- recall@20
- MRR

The report is written to `data/processed/dense_dev_metrics.json` (gitignored). The
holdout split is not used in Block 2.

## Measured Results

Full embedding run on 2026-11-07:

| Check | Result |
|---|---:|
| Documents | 34,378 |
| Chunks | 136,383 |
| Non-null embeddings | 136,383 |
| Embedding dimensions | 1024 |
| Unique chunk IDs | 136,383 |
| Orphan chunks | 0 |
| HNSW index | present, `vector_cosine_ops`, `m=16`, `ef_construction=64` |
| `chunks` total size | 2.0 GB |
| HNSW index size | 1.1 GB |
| Load time | about 46 minutes |
| Rerun after load | 0 written, 0 embedded, 136,383 skipped |

Development retrieval metrics (169 questions, top-20, `float16`):

| Metric | Value |
|---|---:|
| recall@1 | 0.2840 |
| recall@5 | 0.4643 |
| recall@10 | 0.5362 |
| recall@20 | 0.5836 |
| MRR | 0.4330 |

These are diagnostics on the development split, not benchmark claims.

## Commands

```bash
docker compose up -d db
uv run alembic -c backend/alembic.ini upgrade head
uv run python scripts/check_embedding_runtime.py --device cuda --dtype float16
uv run python scripts/embed_and_load.py --chunks data/processed/chunks.jsonl --device cuda --dtype float16 --batch-size 128
uv run python scripts/dense_search.py "How do I rotate an API key?" --top-k 5 --device cuda --dtype float16
uv run python scripts/eval_dense_dev.py --top-k 20 --device cuda --dtype float16
```

## Limitations

- No hybrid retrieval, reranking, or query rewriting yet.
- No API or authentication yet; tenant and knowledge-base IDs are placeholders.
- The HNSW index may need to be dropped and rebuilt around a full bulk load; it was not
  necessary for this 136,383-chunk run.
- CPU fallback is functionally correct but much slower than the GPU path.
- The laptop GPU thermally throttles under sustained load; `float16` was used to keep the
  full run practical.
- Queries must use the same dtype as the indexed documents. Pass `--dtype float16` when
  querying an index built with `float16` weights.
- Development metrics are diagnostics, not benchmark claims.
