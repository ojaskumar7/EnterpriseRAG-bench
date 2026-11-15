# BM25 Production Retrieval Architecture

- Status: implemented for Block 4
- Scope: versioned per-tenant/per-knowledge-base BM25 indexes, atomic publish, and retrieval
- Last measured run: 2026-11-15

## Scope

Block 4 turns the Block 3.5 BM25 benchmark into the production retrieval path. It covers:

- An immutable `bm25s` index built from PostgreSQL chunks.
- A versioned on-disk store with an atomic `current.json` publish pointer.
- A rebuild/check CLI, a query CLI, and a BM25 development evaluation mode.
- A `BM25Retriever` with adaptive overfetch for source filters and PostgreSQL content fetch.

It does not add API, generation, reranking, evidence selection, or incremental indexing.
The 42-question holdout split was not used.

### Why BM25-only replaced the hybrid continuation

Block 3 combined dense retrieval with PostgreSQL full-text search through Reciprocal Rank
Fusion. The lexical side was effectively non-functional: 156 of 169 development questions
returned no lexical results because `websearch_to_tsquery` ANDs natural-language terms.
Hybrid therefore beat dense only marginally (recall@10 0.5348 vs 0.5303, MRR 0.4404 vs
0.4345).

The Block 3.5 benchmark then evaluated exact BM25 over `title + "\n" + content` with a
stemmed tokenizer. BM25 beat both dense and hybrid on every headline metric (recall@20
0.8494, MRR 0.6771). Productionizing another round of hybrid tuning would have optimized a
weaker retriever, so Block 4 productionizes BM25 itself and keeps dense, lexical, and hybrid
code as documented experiments. Evidence selection and context construction move to Block 5.

## Corpus Construction

`scripts/build_bm25_index.py` loads the corpus through `load_bm25_corpus`:

```text
documents count for tenant/KB
  -> chunks ordered by chunk_id
  -> text = title + "\n" + content
  -> bm25s tokenization with English stopwords and PyStemmer
  -> bm25s.BM25(k1=settings.bm25_k1, b=settings.bm25_b, method="lucene")
  -> immutable version directory
```

The corpus is content-only: the title is prepended to the body and is not separately
weighted. `document_count` comes from the `documents` table; `row_count` is the number of
chunks actually indexed. Production code never hardcodes corpus counts.

| Parameter | Value |
|---|---|
| Text | `title + "\n" + content` |
| `k1` | `settings.bm25_k1` (default 1.5) |
| `b` | `settings.bm25_b` (default 0.75) |
| Method | `lucene` |
| Stopwords | `english` |
| Stemmer | `PyStemmer("english")` |
| `bm25s` version | 0.3.16 (recorded in the manifest) |

The build writes the active `k1`/`b` values into the manifest, and load verifies the stored
index parameters match the manifest. The tokenizer configuration is fixed and validated on
load. `corpus_fingerprint` is an order-independent SHA256 over sorted
`(chunk_id, doc_id, source_type, title, chunk_index, content_hash)` tuples, so it changes
when corpus membership, text, or retrieval-relevant metadata changes.

## On-Disk Layout

```text
data/indexes/bm25/
└── <tenant_id>/
    └── <knowledge_base_id>/
        ├── current.json
        ├── .build.lock
        └── versions/
            └── <version>/
                ├── manifest.json
                ├── catalog.jsonl
                └── index/
                    ├── data.csc.index.npy
                    ├── indices.csc.index.npy
                    ├── indptr.csc.index.npy
                    ├── params.index.json
                    └── vocab.index.json
```

A version name is `<UTC timestamp>-<fingerprint prefix>`, for example
`20261114T114342596160Z-9b59d71b6d36`. A published version directory is immutable; a new
build always creates a new version rather than modifying an existing one.

### Manifest

`manifest.json` records everything needed to validate and operate the version:

| Field | Purpose |
|---|---|
| `schema_version` | Index metadata schema; readers reject unsupported versions |
| `version` | Version directory name |
| `tenant_id`, `knowledge_base_id` | Index ownership |
| `created_at` | Build timestamp |
| `document_count`, `row_count` | Source document and indexed chunk counts |
| `corpus_fingerprint` | Order-independent corpus identity |
| `catalog_sha256` | Integrity check for `catalog.jsonl` |
| `k1`, `b`, `method` | BM25 scoring parameters |
| `stopwords`, `stemmer` | Tokenizer configuration |
| `bm25s_version`, `pystemmer_version` | Build-time library versions |

### Catalog

`catalog.jsonl` is one JSON object per indexed row:

```text
row_index, chunk_id, doc_id, source_type, title, chunk_index, content_hash
```

`catalog[row_index]` is the row identity returned by `bm25s`. The catalog is written in
`chunk_id` order, so `row_index` is deterministic. It stores identity and metadata only;
final chunk content is fetched from PostgreSQL at query time so the index never serves
stale text.

### Pointer, retention, and lock

`current.json` contains:

```json
{"version": "<version>", "manifest_sha256": "<sha256 of manifest.json>"}
```

Retention keeps the active version plus the previous two (`retention=2`, three total). The
cleanup runs after a successful pointer switch and never removes the active version.

A build holds an exclusive non-blocking `flock` on `.build.lock` for the tenant/KB. A
concurrent build for the same tenant/KB fails fast with
`another BM25 build is running for <root>`; builds for different tenant/KB pairs are
independent.

## Atomic Publish and Failure Behavior

The build writes into a temporary `.building-<uuid>` directory, then validates it by
reloading the saved index and retrieving one probe row. Publish is:

```text
write index/ + catalog.jsonl + manifest.json into .building-<uuid>
  -> reload and probe validation
  -> fsync every version file and directory
  -> rename .building-<uuid> to versions/<version>
  -> fsync versions/
  -> write and fsync a temporary pointer file
  -> os.replace(pointer, current.json)
  -> fsync the tenant/KB root
  -> best-effort retention cleanup (logged, never fails the publish)
```

The rename and pointer replace are atomic filesystem operations. Every version file and
directory is fsynced before the pointer swap, so a committed pointer never references
unflushed index data. A failure before the pointer replace leaves the previous
`current.json` untouched, so readers keep the old version. A failed build removes its
temporary directory. A failure after the version rename but before the pointer replace can
leave an inactive orphan version directory; it is never served, and only a later retention
pass that pushes it outside the retained window will remove it. A version that already
exists is never overwritten. Cleanup runs after the pointer swap and is best-effort: an
OSError is logged and the committed publish still succeeds. The manager clears its snapshot
cache in a `finally` block, so a post-swap failure cannot leave a stale cached version
behind the new pointer.

## Runtime Flow

```text
query
  -> reject top_k <= 0
  -> return [] for blank/whitespace-only query
  -> load cached snapshot for tenant/KB
  -> tokenize with the same English stemmer/stopwords
  -> return [] when no tokens remain
  -> bm25s search with candidate_k
  -> drop non-positive scores (out-of-vocabulary queries return no hits)
  -> map row indices to catalog entries and apply source_type filter
  -> adaptively overfetch when filtered results are short
  -> fetch final content from PostgreSQL by chunk_id
       constrained by tenant_id, knowledge_base_id, and optional source_types
  -> compare selected metadata against the catalog; raise BM25IndexStale on mismatch
  -> return top_k RetrievedChunk values with BM25 scores
```

Without a source filter, the retriever requests `top_k` rows. With a filter it starts at
`max(top_k * 4, top_k)` (bounded by index size) and doubles `candidate_k` until enough
filtered chunks are found or the index is exhausted. Filters are user-selected product
source types; evaluation metadata is never passed to the retriever.

The catalog is not trusted as the content source. `get_chunks_by_ids` loads only the
fields needed for hydration (no embeddings or generated `search_vector`) and re-applies
tenant and knowledge-base isolation, so a stale or cross-tenant catalog cannot leak
content. If any selected chunk is missing from PostgreSQL, or if its `doc_id`,
`source_type`, `title`, `chunk_index`, or `content_hash` differs from the catalog, the
query fails with `BM25IndexStale` and a tenant/KB-specific rebuild command rather than
returning a partial result.

### Error contract

| Condition | Error |
|---|---|
| `current.json` missing | `BM25IndexUnavailable` |
| pointer or manifest invalid / hash mismatch | `BM25IndexStale` |
| `catalog.jsonl` missing | `BM25IndexUnavailable` |
| catalog hash or count mismatch | `BM25IndexStale` |
| stored index parameters or row count differ from manifest | `BM25IndexStale` |
| index files missing or unreadable | `BM25IndexUnavailable` |
| selected chunk missing from PostgreSQL | `BM25IndexStale` |
| selected chunk metadata differs from catalog | `BM25IndexStale` |
| blank, stopword-only, or out-of-vocabulary query | no error; returns `[]` |
| `top_k <= 0` | `ValueError("top_k must be positive")` |

All actionable errors name the tenant/KB and include:

```text
uv run python scripts/build_bm25_index.py --tenant-id <tenant_id> --knowledge-base-id <kb_id>
```

There is no PostgreSQL full-text fallback. A missing or stale index is an operational
error, not a silent quality degradation.

## Caching and Reload Semantics

`BM25IndexManager` caches one loaded snapshot per `(tenant_id, knowledge_base_id)`.
Concurrent queries share the immutable snapshot safely. Query tokenization runs through a
thread-local PyStemmer instance because PyStemmer objects are not safe to call from multiple
threads; `asyncio.to_thread` workers therefore never share stemmer state. `reload()` drops
the cached entry and loads the current pointer again; `publish()` clears the cache in a
`finally` block so post-swap failures cannot leave a stale snapshot cached.

Short-lived CLI and evaluation processes therefore always see the latest index. A
long-running process keeps its snapshot until it calls `reload()`. After a rebuild, restart
the process or call `reload()` for that tenant/KB; a process using an old snapshot can also
detect that its selected chunks no longer match PostgreSQL and raise `BM25IndexStale`.

## Batch-Rebuild Limitation

Indexes are rebuilt in batch; there is no incremental update or change-data-capture. After
any ingestion that adds, removes, or modifies chunks, rebuild the affected tenant/KB index.
`--check` reads only pointer/manifest/catalog metadata and a metadata-only PostgreSQL
projection, then compares the stored fingerprint with the fresh fingerprint. It does not
load the `bm25s` arrays, so checking does not pay the ~181 MB index read cost. Runtime
`load()` still validates the full stored index, including its parameters and row count.
Ingestion is expected to be slow and infrequent relative to queries, so a batch rebuild is
acceptable for this block.

Each loaded index is held in process memory in addition to its on-disk footprint. The
measured on-disk version for the default tenant/KB is about **181 MB** for 136,383 chunks;
memory scales with the number of distinct tenant/KB indexes a process serves.

## Operational Commands

```bash
# Build and publish a new version
uv run python scripts/build_bm25_index.py \
  --tenant-id 00000000-0000-0000-0000-000000000001 \
  --knowledge-base-id 00000000-0000-0000-0000-000000000002

# Verify the active version matches PostgreSQL
uv run python scripts/build_bm25_index.py \
  --tenant-id 00000000-0000-0000-0000-000000000001 \
  --knowledge-base-id 00000000-0000-0000-0000-000000000002 \
  --check

# Query the active index
uv run python scripts/bm25_search.py "How do I rotate an API key?" --top-k 5
uv run python scripts/bm25_search.py "How do I rotate an API key?" \
  --top-k 5 --source-type confluence --json

# Evaluate the development split only
uv run python scripts/eval_retrieval_dev.py \
  --mode bm25 --top-k 20 \
  --output data/processed/retrieval_dev_metrics_block4.json
```

The build writes to `data/indexes/bm25/`, which is gitignored. Do not commit index
artifacts.

## Measured Results

Measured on 2026-11-15 for the default tenant/KB
(`00000000-0000-0000-0000-000000000001` /
`00000000-0000-0000-0000-000000000002`):

- Index version: `20261115T131732353721Z-f79d12de1fc6`
- Documents: 34,378
- Indexed chunks: 136,383
- `bm25s` 0.3.16, PyStemmer 3.1.0, `k1=1.5`, `b=0.75`, `method=lucene`
- On-disk version size: about 181 MB
- `--check`: `BM25 index is current`
- Latency measured over 169 queries at `top_k=50`: p50 8.5 ms, p95 12.9 ms, max 17.5 ms

Development metrics over the 169 development questions, `top_k=20`:

| Metric | BM25 | Block 3.5 target |
|---|---:|---:|
| recall@1 | 0.5138 | — |
| recall@5 | 0.7310 | 0.7310 |
| recall@10 | 0.7776 | 0.7776 |
| recall@20 | 0.8494 | 0.8494 |
| MRR | 0.6771 | 0.6771 |
| zero-result questions | 0 | 0 |
| top-50 p95 | 12.9 ms | `< 200 ms` |

The measured BM25 run reproduces the Block 3.5 benchmark and clears every Block 4 target.
These are development-split diagnostics, not benchmark claims. The metrics file
`data/processed/retrieval_dev_metrics_block4.json` is gitignored.

## Alternatives Considered

- **PostgreSQL FTS only.** Already present in Block 3; its AND semantics returned zero
  results for 156 of 169 development questions, so it cannot be the production retriever.
- **Hybrid RRF.** Measured below BM25 on every headline metric; productionizing it would
  add dense-model latency and fusion tuning without improving the result.
- **External search service.** Adds another process and index lifecycle to keep in sync
  with PostgreSQL; unnecessary at this corpus size.
- **Incremental BM25 updates.** More complex than this block needs; batch rebuilds are
  acceptable while ingestion is infrequent.

## Holdout Status

The 42-question holdout split was **not** run or tuned on during Block 4. All reported
numbers are development-split diagnostics.
