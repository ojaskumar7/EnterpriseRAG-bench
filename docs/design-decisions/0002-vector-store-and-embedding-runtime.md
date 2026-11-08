# 0002: pgvector, BGE, and Dense Retrieval

- Status: accepted
- Date: 2026-11-07
- Scope: Block 2 embeddings, vector storage, and dense retrieval

## Context

Block 2 needs to embed the full Block 1 corpus and retrieve relevant chunks while
keeping evaluation metadata out of the retrieval path. The local machine has an
RTX 5050 Laptop GPU with 8 GB of VRAM, an NVIDIA driver with CUDA 13.1 support, and
PostgreSQL available through Docker. The host runs WSL, so the CUDA user-space driver
comes from `/usr/lib/wsl/lib`.

The Block 1 corpus has 136,383 chunks, 34,378 document members, and a known document
identity problem: `(source_type, doc_id)` is not unique.

## Decision

### Embedding model

Use `BAAI/bge-large-en-v1.5` with 1024-dimensional normalized embeddings. Use the
BGE query instruction only for queries. Documents are embedded without the instruction.

The model is loaded lazily through sentence-transformers. The provider exposes
`embed_documents` and `embed_query`; tests use a deterministic fake provider so unit
tests do not download model weights.

Pin `torch==2.8.0` (CUDA 12.8 build). The initial resolver selected PyTorch 2.15.0
with CUDA 13.2 packages; on this WSL host, `torch.cuda.is_available()` segfaulted
inside `cudaGetDeviceCount` because the installed driver reports CUDA 13.1. The
CUDA 12.8 build works on the same driver.

On CUDA, the bulk load uses `float16` weights. A 100-chunk comparison against
`float32` produced a mean cosine similarity of 1.0001 and a minimum of 0.9998, while
increasing measured throughput from about 7 chunks/second to about 48-67 chunks/second.
The default remains `float32`; the loader, search, and evaluation CLIs share a
`--dtype` flag so queries use the same precision as the indexed documents.

### Vector store

Use PostgreSQL 16 with the pgvector extension through the `pgvector/pgvector:pg16`
Docker image. Use cosine distance and an HNSW index with `m=16` and
`ef_construction=64`. The local Compose service maps host port 5433 because port 5432
was already occupied; the default `DATABASE_URL` uses 5433.

PostgreSQL is chosen over a separate vector database because the project already needs
relational metadata, filters, transactions, and idempotent upserts. pgvector keeps the
initial architecture simple and avoids a second stateful service.

### Database access

Use SQLAlchemy 2.0 async with asyncpg and Alembic migrations. The application exposes
an async session factory and repository functions rather than leaking SQL into the
retriever.

### Document identity

Store documents with a generated UUID primary key and a unique natural key of
`(tenant_id, knowledge_base_id, source_type, doc_id, document_content_hash)`.

`(source_type, doc_id)` alone is not sufficient because one Jira document ID appears
twice with different content. The content hash preserves both variants and remains
stable when the source record is unchanged.

Chunk identity is `(tenant_id, knowledge_base_id, chunk_id)`. The existing chunk ID
format is unchanged: `{doc_id}:{chunk_index}:{sha256(content)[:12]}`.

### Load strategy

Stream `chunks.jsonl` in batches. Upsert documents and chunks in the same transaction.
Embedding is nullable so a batch can be retried; already embedded chunk IDs are skipped
on reruns. This makes the loader idempotent without requiring a separate job table.

The measured bulk run used 128-chunk embedding batches and 256-chunk database
transactions. The HNSW index remained enabled; no drop/recreate was necessary.

### Evaluation scope

Block 2 reports recall@1/5/10/20 and MRR on the 169 development questions only. This is
a retrieval sanity check, not the Block 6 evaluation harness. The holdout set remains
untouched.

The retriever accepts only the query text, `top_k`, and user-selected source filters.
It never accepts `expected_doc_ids`, `question_type`, `gold_answer`, or evaluation
source metadata.

## Consequences

- A GPU is optional but preferred. If CUDA or the model download fails, the runtime
  check and provider can fall back to CPU, with much slower embedding throughput.
- The PyTorch version is pinned to the CUDA 12.8 build for WSL driver compatibility.
  Upgrading PyTorch requires re-running the CUDA smoke check first.
- `float16` document embeddings must be queried with `float16`; using a different
  precision changes vectors slightly, so the CLIs default to the configured dtype.
- The HNSW index adds build time during bulk loads. If loading becomes slow, the index
  can be dropped and recreated after the initial corpus load.
- The schema reserves tenant and knowledge-base IDs now, even though authentication
  and multi-tenant API behavior arrive later.
- Changing the embedding model requires a new dimension and a full re-embedding or a
  second embedding column/table.

## Alternatives Considered

| Alternative | Why not now |
|---|---|
| Chroma or FAISS | Adds a second store and separates vectors from relational metadata |
| Sync SQLAlchemy/psycopg | Simpler, but the API layer will be async in a later block |
| IVFFlat index | Requires training and tuning; HNSW is a better default for this corpus size |
| Store only `(source_type, doc_id)` | Collides on the duplicate Jira document |
| Embed at ingestion time | Couples chunking to model availability and makes reprocessing harder |
