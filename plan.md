# Block 1 Implementation Plan — Reproducible Ingestion Foundation

> **For agentic workers:** Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible ingestion foundation that downloads a representative EnterpriseRAG-Bench subset, loads and normalizes documents, chunks them with rich metadata, and produces a validated chunk corpus plus an EDA report.

**Architecture:** Format-pluggable loaders produce validated `RawDocument` objects. A normalizer creates deterministic clean text and content hashes. A token-aware chunker creates stable `Chunk` records with document/chunk metadata and character offsets. A pipeline writes JSONL artifacts and statistics. No embeddings, database, retrieval, LLM, or API code is added in this block.

**Tech Stack:** Python 3.12, `uv`, Pydantic v2, PyArrow, Hugging Face `transformers` tokenizer, pytest, Ruff.

**Estimated effort:** 17.5 hours implementation + 2.5 hours review/buffer = 20 hours.

---

## Approved Baseline Decisions

1. Use the official dataset `onyx-dot-app/EnterpriseRAG-Bench`, not the `SJChen02` mirror.
2. Use a Block 1 development corpus made from all release slices for Confluence, Jira, GitHub, and HubSpot.
3. Use `BAAI/bge-large-en-v1.5` as the initial embedding model in Block 2. Dimension: 1024.
4. Use 400-token chunks with 64-token overlap. Minimum chunk size: 50 tokens.
5. Use a stratified 80/20 development/holdout split of benchmark questions. Keep the holdout untouched until Block 6.

---

## Global Constraints

- Python 3.12 initially. Fall back to Python 3.11 only if a required ML package is incompatible.
- All code must have type hints.
- Use Pydantic v2 for data validation.
- Use only UTF-8 text. Decode with strict mode and record failures.
- Treat all dataset content as untrusted input.
- Never use `source_types`, `expected_doc_ids`, `question_type`, or `gold_answer` as retrieval inputs. These fields are evaluation metadata only.
- Never index benchmark questions as documents.
- Do not commit raw or processed dataset files. Commit only tiny fixtures and manifests.
- Do not hardcode secrets. Use environment variables.
- Do not add embedding, database, retrieval, LLM, API, Redis, MinIO, Langfuse, or frontend code in this block.
- Run `uv run pytest` and `uv run ruff check .` before claiming completion.
- Do not fabricate dataset counts, metrics, or benchmark results.

---

## Scope

### In scope

- Git repository initialization and Python tooling.
- Reproducible download of selected official release slices.
- Question subset manifest with gold-document coverage validation.
- Document and chunk data models.
- Zip/text and parquet document loaders.
- Text normalization and content hashing.
- Token-aware chunking with metadata and character offsets.
- Ingestion pipeline and CLI.
- Unit tests, integration tests, and data-quality checks.
- EDA report and ingestion documentation.
- Block 1 completion review.

### Out of scope

- Embeddings and embedding-provider interfaces.
- PostgreSQL, pgvector, Alembic, and repositories.
- BM25, dense retrieval, hybrid retrieval, RRF, and reranking.
- Gemini, prompt construction, answer generation, citations, and abstention.
- FastAPI, JWT, multi-tenancy implementation, Redis, MinIO, Langfuse, and frontend.
- PDF/OCR ingestion. The dataset contains text exports, not PDFs.

---

## Corpus Subset

### Sources

| Source | Release slices | Approx. docs | Approx. compressed size |
|---|---:|---:|---:|
| Confluence | 2 | 5,189 | ~26 MB |
| Jira | 2 | 6,120 | ~17 MB |
| GitHub | 2 | 8,052 | ~20 MB |
| HubSpot | 4 | 15,017 | ~28 MB |
| **Total** | **10** | **~34,378** | **~90 MB** |

These sources provide a mix of long structured documents, support tickets, technical discussions, and CRM records. Slack and Gmail are excluded because they dominate corpus volume and would require many additional slices. They will be added when scaling to the full corpus.

### Question selection rules

For each question in `questions.jsonl`:

1. Parse `question_type`, `source_types`, `expected_doc_ids`, `question`, `gold_answer`, and `answer_facts`.
2. Keep the question only if every item in `source_types` is in the selected source set.
3. Keep the question only if every `expected_doc_id` is present in the downloaded document corpus.
4. Exclude questions with an empty `expected_doc_ids` list from retrieval evaluation. Route `high_level` and `info_not_found` questions to a separate abstention set.
5. Record excluded questions and the reason for exclusion in the manifest.
6. Produce a stratified 80/20 development/holdout split using `question_type` as the stratification key. Use a fixed random seed.

The exact number of eligible questions must be computed by `build_subset_manifest.py`. Do not hardcode it.

### Label leakage rule

The manifest may use `source_types` and `expected_doc_ids` for corpus construction and evaluation. Retrieval and generation code must receive only the question text and product-level filters chosen by the user. The manifest must never be passed to a retriever.

---

## Target Files

```text
productionRAG/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── ingestion/
│   │       ├── __init__.py
│   │       ├── models.py
│   │       ├── loaders/
│   │       │   ├── __init__.py
│   │       │   ├── base.py
│   │       │   ├── parquet_loader.py
│   │       │   └── zip_loader.py
│   │       ├── preprocessing/
│   │       │   ├── __init__.py
│   │       │   └── normalize.py
│   │       ├── chunking/
│   │       │   ├── __init__.py
│   │       │   └── token_chunker.py
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
│   │   └── ingestion.md
│   └── design-decisions/
│       └── 0001-dataset-subset.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
├── README.md
└── plan.md
```

---

## Interfaces

These signatures are the contract for Block 1. Later blocks depend on them.

### Data models

```python
from typing import Any, Literal
from pydantic import BaseModel, Field

SourceType = Literal[
    "slack",
    "gmail",
    "linear",
    "google_drive",
    "hubspot",
    "fireflies",
    "github",
    "jira",
    "confluence",
]


class RawDocument(BaseModel):
    doc_id: str
    source_type: SourceType
    title: str
    content: str
    source_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class NormalizedDocument(BaseModel):
    doc_id: str
    source_type: SourceType
    title: str
    content: str
    raw_content: str
    content_hash: str
    char_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    content: str
    token_count: int
    start_char: int
    end_char: int
    content_hash: str
    source_type: SourceType
    title: str
    metadata: dict[str, Any] = Field(default_factory=dict)
```

### Loader protocol

```python
from collections.abc import Iterator
from typing import Protocol


class DocumentLoader(Protocol):
    def iter_documents(self) -> Iterator[RawDocument]:
        ...
```

### Chunker

```python
class TokenChunker:
    def __init__(
        self,
        tokenizer_name: str = "BAAI/bge-large-en-v1.5",
        chunk_size: int = 400,
        chunk_overlap: int = 64,
        min_chunk_tokens: int = 50,
    ) -> None:
        ...

    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        ...
```

### Pipeline

```python
class IngestionConfig(BaseModel):
    chunk_size: int = 400
    chunk_overlap: int = 64
    min_chunk_tokens: int = 50
    tokenizer_name: str = "BAAI/bge-large-en-v1.5"
    output_dir: str = "data/processed"
    max_docs: int | None = None
    source_types: list[SourceType] | None = None


class IngestionResult(BaseModel):
    documents_loaded: int
    documents_skipped: int
    documents_failed: int
    chunks_created: int
    duplicate_documents: int
    errors: list[dict[str, str]]


class IngestionPipeline:
    def __init__(self, loader: DocumentLoader, config: IngestionConfig) -> None:
        ...

    def run(self) -> IngestionResult:
        ...
```

---

## Tasks

### Task 0: Lock Decisions and Create the Decision Record

**Files:**
- Create: `docs/design-decisions/0001-dataset-subset.md`
- Create: `docs/architecture/ingestion.md` (skeleton only)

**Interfaces:**
- Consumes: approved decisions from this plan.
- Produces: written rationale for the dataset source, subset sources, chunk defaults, and evaluation split.

- [x] **Step 1: Create the decision record**

Document:
- Official dataset URL and mirror note.
- Why Confluence, Jira, GitHub, and HubSpot are selected.
- Why Slack and Gmail are excluded for Block 1.
- The label-leakage rule.
- The 80/20 split and fixed seed.
- The chunk defaults: 400/64/50.

- [ ] **Step 2: Create the ingestion document skeleton**

Include headings for pipeline flow, data model, chunking, manifest rules, statistics, and limitations. Leave implementation details for Task 7.

- [ ] **Step 3: Commit**

```bash
git add docs/design-decisions/0001-dataset-subset.md docs/architecture/ingestion.md
git commit -m "docs: record block 1 dataset and ingestion decisions"
```

---

### Task 1: Initialize Repository and Python Tooling

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `uv.lock`
- Create: `README.md` (already created; update only if needed)
- Create: `backend/app/__init__.py`
- Create: `backend/tests/unit/test_smoke.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a working `uv` environment, Ruff configuration, pytest configuration, and importable `app` package.

- [ ] **Step 1: Initialize git**

```bash
git init
```

- [ ] **Step 2: Create `.gitignore`**

Include:

```text
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.mypy_cache/
.env
data/raw/
data/processed/
*.parquet
*.zip
```

- [ ] **Step 3: Create `pyproject.toml`**

Configure:
- Project name: `enterprise-knowledge-intelligence`
- Python requirement: `>=3.12`
- Dependencies: `pydantic>=2`, `pyarrow`, `transformers`
- Development dependencies: `pytest`, `pytest-cov`, `ruff`
- Ruff line length: 100
- Pytest test paths: `backend/tests`

- [ ] **Step 4: Install and lock**

```bash
uv lock
uv sync
```

Expected: `.venv` is created and `uv.lock` is generated.

- [ ] **Step 5: Write the failing smoke test**

`backend/tests/unit/test_smoke.py`:

```python
def test_app_package_imports() -> None:
    import app

    assert app is not None
```

- [ ] **Step 6: Run the test to confirm failure**

```bash
uv run pytest backend/tests/unit/test_smoke.py -v
```

Expected: FAIL because `backend/app/__init__.py` does not exist or is not importable.

- [ ] **Step 7: Create the package marker**

Create `backend/app/__init__.py` with a short module docstring.

- [ ] **Step 8: Run the test and lint**

```bash
uv run pytest backend/tests/unit/test_smoke.py -v
uv run ruff check .
```

Expected: PASS and no lint errors.

- [ ] **Step 9: Commit**

```bash
git add .gitignore pyproject.toml uv.lock backend/app/__init__.py backend/tests/unit/test_smoke.py
git commit -m "chore: initialize python project and test tooling"
```

---

### Task 2: Download the Subset and Build the Manifest

**Files:**
- Create: `scripts/download_subset.py`
- Create: `scripts/build_subset_manifest.py`
- Create: `backend/tests/unit/test_manifest.py`
- Create: `backend/tests/fixtures/questions.jsonl`
- Create: `data/sample/subset_manifest.json`
- Create: `data/sample/dev_questions.jsonl`
- Create: `data/sample/holdout_questions.jsonl`
- Create: `data/sample/abstention_questions.jsonl`

**Interfaces:**
- Consumes: official release URLs and `questions.jsonl`.
- Produces: `data/raw/slices/*.zip`, `data/raw/questions.jsonl`, document ID inventory, and the split files.

- [x] **Step 1: Write the failing manifest tests**

Test cases:

```python
def test_question_kept_when_all_sources_selected_and_gold_docs_present() -> None:
    ...

def test_question_excluded_when_source_type_not_selected() -> None:
    ...

def test_question_excluded_when_expected_doc_missing() -> None:
    ...

def test_zero_gold_question_routed_to_abstention_set() -> None:
    ...

def test_split_is_stratified_and_uses_fixed_seed() -> None:
    ...

def test_manifest_never_contains_question_text_as_document() -> None:
    ...
```

- [x] **Step 2: Run the tests to confirm failure**

```bash
uv run pytest backend/tests/unit/test_manifest.py -v
```

Expected: FAIL because the manifest module does not exist.

- [x] **Step 3: Implement `download_subset.py`**

Requirements:
- Download the approved release slices for the four sources.
- Download `questions.jsonl`.
- Store files under `data/raw/`.
- Verify zip integrity with `zipfile.ZipFile.testzip()`.
- Record SHA256, byte size, and file count in `data/raw/download_manifest.json`.
- Support `--force` and `--dry-run`.
- Never delete existing files unless `--force` is passed.

- [x] **Step 4: Implement `build_subset_manifest.py`**

Requirements:
- Read the downloaded zips without extracting them.
- Build an inventory of `doc_id` values and source types.
- Apply the question selection rules.
- Route zero-gold questions to the abstention set.
- Produce a stratified 80/20 split with a fixed seed.
- Write `data/sample/subset_manifest.json`, `dev_questions.jsonl`, `holdout_questions.jsonl`, and `abstention_questions.jsonl`.
- Fail loudly if a selected question has a missing gold document.
- Report excluded questions with reasons.

- [x] **Step 5: Run the tests**

```bash
uv run pytest backend/tests/unit/test_manifest.py -v
```

Expected: PASS.

- [x] **Step 6: Run the real download and manifest build**

```bash
uv run python scripts/download_subset.py
uv run python scripts/build_subset_manifest.py
```

Expected: ~90 MB downloaded; manifest files created; no missing gold documents.

- [x] **Step 7: Inspect and record actual counts**

Record the actual document count, eligible question count, split counts, and exclusion reasons in the decision record. Do not guess.

Actual: 34,375 unique doc IDs / 34,378 document members; 211 eligible; 169 dev; 42 holdout; 30 abstention; 259 excluded; 0 missing gold documents.

- [x] **Step 8: Commit**

```bash
git add scripts/download_subset.py scripts/build_subset_manifest.py backend/tests/unit/test_manifest.py backend/tests/fixtures/questions.jsonl data/sample/
git commit -m "feat: add subset download and question manifest pipeline"
```

---

### Task 3: Implement Data Models and Loaders

**Files:**
- Create: `backend/app/ingestion/models.py`
- Create: `backend/app/ingestion/loaders/base.py`
- Create: `backend/app/ingestion/loaders/zip_loader.py`
- Create: `backend/app/ingestion/loaders/parquet_loader.py`
- Create: `backend/tests/unit/test_models.py`
- Create: `backend/tests/unit/test_zip_loader.py`
- Create: `backend/tests/unit/test_parquet_loader.py`
- Create: `backend/tests/fixtures/sample_docs.zip`
- Create: `backend/tests/fixtures/sample_docs.parquet`

**Interfaces:**
- Consumes: `RawDocument`, `NormalizedDocument`, `Chunk`, `SourceType`, `DocumentLoader`.
- Produces: iterators over validated `RawDocument` objects.

- [x] **Step 1: Write the failing model tests**

Test cases:

```python
def test_valid_doc_id_is_accepted() -> None:
    ...

def test_invalid_doc_id_is_rejected() -> None:
    ...

def test_unknown_source_type_is_rejected() -> None:
    ...

def test_chunk_serializes_to_json() -> None:
    ...
```

- [x] **Step 2: Write the failing loader tests**

Test cases:

```python
def test_zip_loader_parses_doc_id_and_title() -> None:
    ...

def test_zip_loader_derives_source_type_from_path() -> None:
    ...

def test_zip_loader_skips_malformed_filename() -> None:
    ...

def test_zip_loader_rejects_path_traversal() -> None:
    ...

def test_parquet_loader_reads_batches() -> None:
    ...

def test_parquet_loader_rejects_missing_columns() -> None:
    ...
```

- [x] **Step 3: Run the tests to confirm failure**

```bash
uv run pytest backend/tests/unit/test_models.py backend/tests/unit/test_zip_loader.py backend/tests/unit/test_parquet_loader.py -v
```

Expected: FAIL because the modules do not exist.

- [x] **Step 4: Implement the models**

Validate `doc_id` with `^dsid_[0-9a-f]{32}$`. Use Pydantic v2. Keep models immutable where practical. Reject unknown source types.

- [x] **Step 5: Implement the zip loader**

Requirements:
- Read `.txt` members directly from the zip; do not extract to disk.
- Parse `dsid_<32 hex>__slug.txt`.
- Derive `source_type` from the first path component.
- Use the first non-empty line as the title.
- Join the remaining lines as content.
- Reject absolute paths, `..` traversal, and members that exceed a configured uncompressed size.
- Skip malformed filenames and record the reason.

- [x] **Step 6: Implement the parquet loader**

Requirements:
- Read the HF parquet schema: `doc_id`, `source_type`, `title`, `content`.
- Support batch iteration with a configurable batch size.
- Reject missing columns with a clear error.
- Skip rows with invalid `doc_id` values and record the reason.

- [x] **Step 7: Run the tests and lint**

```bash
uv run pytest backend/tests/unit/test_models.py backend/tests/unit/test_zip_loader.py backend/tests/unit/test_parquet_loader.py -v
uv run ruff check .
```

Expected: PASS.

- [x] **Step 8: Commit**

```bash
git add backend/app/ingestion/models.py backend/app/ingestion/loaders/ backend/tests/
git commit -m "feat: add document models and zip/parquet loaders"
```

---

### Task 4: Implement Normalization

**Files:**
- Create: `backend/app/ingestion/preprocessing/normalize.py`
- Create: `backend/tests/unit/test_normalize.py`

**Interfaces:**
- Consumes: `RawDocument`.
- Produces: `NormalizedDocument` with deterministic `content` and `content_hash`.

- [x] **Step 1: Write the failing tests**

Test cases:

```python
def test_unicode_is_nfkc_normalized() -> None:
    ...

def test_crlf_is_normalized_to_lf() -> None:
    ...

def test_null_bytes_are_removed() -> None:
    ...

def test_excess_blank_lines_are_collapsed() -> None:
    ...

def test_headings_and_lists_are_preserved() -> None:
    ...

def test_content_hash_is_stable_across_runs() -> None:
    ...

def test_empty_document_is_flagged_not_crashed() -> None:
    ...
```

- [x] **Step 2: Run the tests to confirm failure**

```bash
uv run pytest backend/tests/unit/test_normalize.py -v
```

Expected: FAIL because the module does not exist.

- [x] **Step 3: Implement normalization**

Requirements:
- Unicode NFKC normalization.
- Normalize `\r\n` and `\r` to `\n`.
- Remove null bytes.
- Strip trailing whitespace per line.
- Collapse three or more consecutive blank lines to two.
- Preserve headings, bullet points, code blocks, and punctuation.
- Do not lowercase content.
- Compute SHA256 over the UTF-8 encoded normalized content.
- Return `NormalizedDocument`.

- [x] **Step 4: Run the tests and lint**

```bash
uv run pytest backend/tests/unit/test_normalize.py -v
uv run ruff check .
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add backend/app/ingestion/preprocessing/normalize.py backend/tests/unit/test_normalize.py
git commit -m "feat: add deterministic document normalization"
```

---

### Task 5: Implement Token-Aware Chunking

**Files:**
- Create: `backend/app/ingestion/chunking/token_chunker.py`
- Create: `backend/tests/unit/test_chunking.py`

**Interfaces:**
- Consumes: `NormalizedDocument`, tokenizer name, chunk size, overlap, minimum chunk size.
- Produces: a deterministic `list[Chunk]`.

- [x] **Step 1: Write the failing tests**

Test cases:

```python
def test_short_document_produces_one_chunk() -> None:
    ...

def test_long_document_produces_multiple_chunks() -> None:
    ...

def test_no_chunk_exceeds_max_tokens() -> None:
    ...

def test_adjacent_chunks_overlap() -> None:
    ...

def test_chunk_ids_are_deterministic() -> None:
    ...

def test_character_offsets_map_to_original_content() -> None:
    ...

def test_trailing_small_chunk_is_merged() -> None:
    ...

def test_empty_document_produces_no_chunks() -> None:
    ...
```

- [x] **Step 2: Run the tests to confirm failure**

```bash
uv run pytest backend/tests/unit/test_chunking.py -v
```

Expected: FAIL because the module does not exist.

- [x] **Step 3: Implement the chunker**

Requirements:
- Load the tokenizer with `AutoTokenizer.from_pretrained(tokenizer_name)`.
- Use tokenizer offsets to map chunks back to character spans.
- Chunk by token windows with the configured overlap.
- Never exceed `chunk_size` tokens.
- Merge a trailing chunk smaller than `min_chunk_tokens` into the previous chunk when possible.
- Preserve short documents as a single chunk.
- Return no chunks for empty content.
- Generate `chunk_id` as `f"{doc_id}:{chunk_index}:{sha256(content)[:12]}"`.
- Populate `source_type`, `title`, and metadata from the document.
- Keep the tokenizer instance reusable; do not reload it per document.

- [x] **Step 4: Run the tests and lint**

```bash
uv run pytest backend/tests/unit/test_chunking.py -v
uv run ruff check .
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add backend/app/ingestion/chunking/token_chunker.py backend/tests/unit/test_chunking.py
git commit -m "feat: add token-aware deterministic chunking"
```

---

### Task 6: Implement the Pipeline and CLI

**Files:**
- Create: `backend/app/ingestion/pipeline.py`
- Create: `backend/app/config.py`
- Create: `scripts/ingest_subset.py`
- Create: `backend/tests/integration/test_pipeline.py`
- Create: `backend/tests/integration/test_real_subset.py`

**Interfaces:**
- Consumes: a `DocumentLoader`, `IngestionConfig`, normalizer, and chunker.
- Produces: `IngestionResult`, `data/processed/chunks.jsonl`, and `data/processed/stats.json`.

- [x] **Step 1: Write the failing pipeline tests**

Test cases:

```python
def test_pipeline_processes_fixture_zip_end_to_end() -> None:
    ...

def test_pipeline_writes_valid_jsonl() -> None:
    ...

def test_pipeline_counts_documents_and_chunks() -> None:
    ...

def test_pipeline_is_idempotent_on_rerun() -> None:
    ...

def test_pipeline_records_skipped_and_failed_documents() -> None:
    ...

def test_pipeline_does_not_drop_documents_silently() -> None:
    ...
```

- [x] **Step 2: Run the tests to confirm failure**

```bash
uv run pytest backend/tests/integration/test_pipeline.py -v
```

Expected: FAIL because the pipeline does not exist.

- [x] **Step 3: Implement the pipeline**

Requirements:
- Iterate documents from the loader.
- Normalize each document.
- Chunk each normalized document.
- Deduplicate by `(doc_id, content_hash)`.
- Write chunks to `chunks.jsonl` using UTF-8 and one JSON object per line.
- Write statistics to `stats.json`.
- Collect per-document errors and continue; do not abort the whole run for one bad document.
- Record counts for loaded, skipped, failed, duplicate, and chunked documents.
- Support `max_docs` and `source_types` filters for development.
- Make reruns deterministic.

- [x] **Step 4: Implement the CLI**

`scripts/ingest_subset.py` arguments:

```text
--raw-dir data/raw
--output-dir data/processed
--chunk-size 400
--chunk-overlap 64
--min-chunk-tokens 50
--tokenizer BAAI/bge-large-en-v1.5
--sources confluence jira github hubspot
--max-docs
```

- [x] **Step 5: Run fixture and real-subset tests**

```bash
uv run pytest backend/tests/integration/test_pipeline.py -v
uv run pytest backend/tests/integration/test_real_subset.py -v -m integration
```

Mark the real-subset test so it skips cleanly when `data/raw/` is absent.

- [x] **Step 6: Run the real ingestion**

```bash
uv run python scripts/ingest_subset.py --raw-dir data/raw --output-dir data/processed
```

Expected: `data/processed/chunks.jsonl` and `data/processed/stats.json` are created.

- [x] **Step 7: Validate the output**

Check:
- Every chunk has a valid `doc_id` and `chunk_id`.
- No chunk exceeds 400 tokens.
- All chunk IDs are unique.
- Every processed document produced at least one chunk or was counted as skipped/failed.
- Rerunning produces identical chunk IDs.

- [x] **Step 8: Commit**

```bash
git add backend/app/ingestion/pipeline.py backend/app/config.py scripts/ingest_subset.py backend/tests/integration/
git commit -m "feat: add ingestion pipeline and subset CLI"
```

---

### Task 7: Produce EDA and Documentation

**Files:**
- Modify: `docs/architecture/ingestion.md`
- Modify: `docs/design-decisions/0001-dataset-subset.md`
- Create: `data/processed/eda.json` (generated; do not commit if large)
- Modify: `README.md`

**Interfaces:**
- Consumes: `chunks.jsonl`, `stats.json`, and the subset manifest.
- Produces: documented ingestion design and real dataset statistics.

- [x] **Step 1: Generate EDA statistics**

Compute:
- Documents per source.
- Characters per document: min, median, p95, max.
- Tokens per document: min, median, p95, max.
- Chunks per document: min, median, p95, max.
- Chunk token distribution.
- Empty documents.
- Duplicate content hashes.
- Documents with malformed IDs.
- Eligible question counts and split counts.
- Exclusion reasons.

- [x] **Step 2: Update `docs/architecture/ingestion.md`**

Document:
- Pipeline flow.
- Loader interfaces.
- Normalization rules.
- Chunking rationale and defaults.
- Metadata fields.
- Manifest rules and label-leakage warning.
- Known limitations.

- [x] **Step 3: Update the decision record**

Add actual counts from the real run. Mark estimates as estimates and measured values as measured.

- [x] **Step 4: Update `README.md`**

Add current status, setup commands, ingestion commands, and known limitations.

- [x] **Step 5: Commit**

```bash
git add docs/ README.md
git commit -m "docs: document ingestion design and subset statistics"
```

---

### Task 8: Verify Block 1 and Prepare Block 2

**Files:**
- Modify: `README.md` if verification reveals issues.
- Modify: `plan.md` only if the plan itself was wrong.

**Interfaces:**
- Consumes: all Block 1 deliverables.
- Produces: verification evidence and a short retrospective.

- [x] **Step 1: Run the full test suite**

```bash
uv run pytest -q
uv run pytest -q -m integration
```

Expected: all non-skipped tests pass.

- [x] **Step 2: Run lint and formatting checks**

```bash
uv run ruff check .
uv run ruff format --check .
```

Expected: no errors.

- [x] **Step 3: Verify the real outputs**

Check:
- `data/processed/chunks.jsonl` exists and is non-empty.
- `data/processed/stats.json` exists.
- `data/sample/subset_manifest.json` exists.
- Manifest has no missing gold documents.
- No raw or processed data is staged in git.

- [x] **Step 4: Record known limitations**

Include:
- Block 1 uses a four-source subset, not the full corpus.
- PDF/OCR is not implemented.
- Metadata-aware retrieval is not implemented.
- Chunking is a fixed baseline, not experimentally tuned.
- No retrieval or generation evaluation has been run.
- No benchmark results are claimed.

- [x] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: complete block 1 verification and limitations"
```

---

## Testing Strategy

### Unit tests

- Model validation.
- Loader parsing and safety.
- Normalization determinism.
- Chunk token bounds, overlap, offsets, and deterministic IDs.
- Manifest selection rules and split stratification.

### Integration tests

- Fixture zip and parquet ingestion end to end.
- Real subset ingestion, skipped when raw data is absent.
- Idempotent reruns.
- Data-quality assertions on the real chunk corpus.

### Commands

```bash
uv run pytest -q
uv run pytest -q -m integration
uv run ruff check .
uv run ruff format --check .
```

### Required data-quality assertions

- No selected question has a missing gold document.
- No chunk exceeds the configured token limit.
- No duplicate chunk IDs.
- No document is silently dropped.
- No raw or processed dataset files are committed.
- No evaluation metadata is passed into retriever-shaped interfaces.

---

## Completion Checklist

- [x] Git repository initialized.
- [x] `uv` environment and lockfile created.
- [x] Ruff and pytest configured and passing.
- [x] Official four-source subset downloaded reproducibly.
- [x] Download manifest records sizes, hashes, and file counts.
- [x] Question manifest selects only questions with complete gold-document coverage.
- [x] Zero-gold questions routed to a separate abstention set.
- [x] Stratified 80/20 split created with a fixed seed.
- [x] Document and chunk models implemented and validated.
- [x] Zip and parquet loaders implemented with safety checks.
- [x] Normalization deterministic and tested.
- [x] Chunking token-bounded, deterministic, and offset-preserving.
- [x] Pipeline writes `chunks.jsonl` and `stats.json`.
- [x] Reruns are idempotent.
- [x] EDA report contains real measured values.
- [x] README and ingestion documentation updated.
- [x] Known limitations documented.
- [x] No embedding, database, retrieval, LLM, or API code added.
- [x] No benchmark results fabricated.

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Mirror drift or provenance confusion | Use the official `onyx-dot-app` repository and record the dataset revision. |
| Missing gold documents in the subset | Build the manifest from actual downloaded document IDs and fail on missing gold docs. |
| Label leakage | Keep evaluation metadata out of retrieval interfaces; add a review check. |
| Tokenizer mismatch | Use the actual BGE tokenizer for chunk sizing. |
| Zip path traversal or zip bombs | Validate member paths and uncompressed sizes; never extract to disk. |
| Silent data loss | Track loaded, skipped, failed, duplicate, and chunked counts. |
| Full-corpus scale later | Keep loaders and pipeline streaming-friendly; do not assume the subset size. |
| Evaluation overfitting | Keep the 20% holdout untouched until Block 6. |

---

## Block 2 Implementation Plan — Embeddings and Dense Retrieval

**Goal:** Add a local BGE embedding provider, persist documents/chunks/vectors in
PostgreSQL with pgvector, implement dense retrieval, and run a development-split
retrieval sanity check over the full 136,383-chunk corpus.

**Architecture:** A provider-agnostic `EmbeddingProvider` produces normalized BGE
vectors. An idempotent batch loader streams `chunks.jsonl`, upserts documents and
chunks, and stores 1024-dimensional vectors. A `DenseRetriever` embeds queries and
uses pgvector cosine distance with tenant/knowledge-base filters. Evaluation metadata
never enters the retriever interface.

**Tech Stack:** Python 3.12, `uv`, sentence-transformers, PyTorch, SQLAlchemy 2.0
async, asyncpg, Alembic, pgvector, PostgreSQL 16, pytest-asyncio, Ruff.

**Estimated effort:** 20-24 hours implementation plus embedding/index runtime.

**Status:** complete. Implementation, full load, development metrics, verification,
and commits are done.

### Global Constraints

- Use `BAAI/bge-large-en-v1.5` with 1024-dimensional normalized embeddings.
- Use the BGE query instruction for queries only; do not prepend it to documents.
- Keep the 42 holdout questions untouched. Development metrics are diagnostics, not
  benchmark claims.
- Never pass `expected_doc_ids`, `question_type`, `source_types`, or `gold_answer` to a
  retriever-shaped interface. `source_types` may be a user-selected product filter.
- Store `(source_type, doc_id, document_content_hash)` as the document identity.
  `(source_type, doc_id)` is not unique.
- Do not add BM25, hybrid retrieval, RRF, reranking, Gemini, FastAPI, or full RAGAS
  evaluation in this block.
- Do not commit raw or processed data, model caches, embeddings, or database volumes.
- Run `uv run pytest`, database integration tests, Ruff, and formatting checks before
  claiming completion.

### Task 0: Lock Block 2 Decisions

- [x] Confirm async SQLAlchemy 2.0 + asyncpg as the database layer.
- [x] Confirm PostgreSQL 16 + pgvector via Docker Compose.
- [x] Confirm HNSW cosine index.
- [x] Confirm recall@1/5/10/20 and MRR on the 169 development questions.
- [x] Record decisions in `docs/design-decisions/0002-vector-store-and-embedding-runtime.md`.

### Task 1: Dependencies, Settings, and Local Infrastructure

**Files:**
- Modify: `pyproject.toml`
- Create: `.env.example`
- Create: `compose.yml`
- Create: `backend/app/settings.py`
- Create: `backend/tests/unit/test_settings.py`
- Create: `scripts/check_embedding_runtime.py`

- [x] **Step 1: Add dependencies**

Runtime: `sentence-transformers`, `torch`, `sqlalchemy[asyncio]`, `asyncpg`,
`alembic`, `pgvector`, `pydantic-settings`.

Development: `pytest-asyncio`.

- [x] **Step 2: Add Docker Compose for PostgreSQL/pgvector**

Use `pgvector/pgvector:pg16` with a named volume, health check, and local port 5433
(port 5432 was occupied on the development host).
The database name, user, and password are local development credentials only.

- [x] **Step 3: Add settings**

`Settings` should provide:

- `database_url`
- `embedding_model`
- `embedding_dimension`
- `embedding_device`
- `embedding_batch_size`
- `embedding_query_instruction`
- `default_tenant_id`
- `default_knowledge_base_id`

Load `.env` with pydantic-settings and expose a cached `get_settings()`.

- [x] **Step 4: Validate the runtime**

`scripts/check_embedding_runtime.py` should print the PyTorch version, CUDA
availability, device name, BGE embedding dimension, and a sample embedding shape.
Run it before the full embedding job.

- [x] **Step 5: Verify**

```bash
uv sync
uv run python scripts/check_embedding_runtime.py
uv run pytest backend/tests/unit/test_settings.py -v
uv run ruff check .
```

- [x] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .env.example compose.yml backend/app/settings.py \
  backend/tests/unit/test_settings.py scripts/check_embedding_runtime.py
git commit -m "chore: add embedding and database infrastructure"
```

### Task 2: Extend the Chunk Contract

**Files:**
- Modify: `backend/app/ingestion/models.py`
- Modify: `backend/app/ingestion/chunking/token_chunker.py`
- Modify: `backend/tests/unit/test_chunking.py`
- Regenerate: `data/processed/chunks.jsonl` (gitignored)

- [x] **Step 1: Add `document_content_hash`**

Add `document_content_hash` to `Chunk` and populate it from
`NormalizedDocument.content_hash` in `TokenChunker`.

- [x] **Step 2: Update tests**

Assert that every chunk carries the document content hash and that chunk IDs still
depend only on chunk content and chunk index.

- [x] **Step 3: Regenerate the corpus**

```bash
uv run python scripts/ingest_subset.py --raw-dir data/raw --output-dir data/processed
```

- [x] **Step 4: Verify**

```bash
uv run pytest backend/tests/unit/test_chunking.py -v
uv run pytest -q
```

Confirm that chunk IDs and the 136,383 chunk count are unchanged.

- [x] **Step 5: Commit**

```bash
git add backend/app/ingestion/models.py backend/app/ingestion/chunking/token_chunker.py \
  backend/tests/unit/test_chunking.py backend/tests/unit/test_models.py
git commit -m "feat: add document identity hash to chunks"
```

### Task 3: Embedding Provider

**Files:**
- Create: `backend/app/embeddings/__init__.py`
- Create: `backend/app/embeddings/base.py`
- Create: `backend/app/embeddings/bge.py`
- Create: `backend/tests/unit/test_embeddings.py`

- [x] **Step 1: Define the protocol**

```python
class EmbeddingProvider(Protocol):
    model_name: str
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
```

- [x] **Step 2: Implement the BGE provider**

- Lazily load `SentenceTransformer`.
- Resolve `auto`, `cuda`, and `cpu` device selection.
- Validate that the model dimension matches the configured dimension.
- Normalize document and query embeddings.
- Prepend the BGE query instruction only in `embed_query`.
- Use configurable batch size and progress reporting.

- [x] **Step 3: Add a fake provider**

Use a deterministic fake provider for downstream tests. It must not download a model.

- [x] **Step 4: Test**

Cover query-instruction behavior, document batching, device resolution, dimension
validation, and normalized output shape. Monkeypatch the model in unit tests.

- [x] **Step 5: Commit**

```bash
git add backend/app/embeddings/__init__.py backend/app/embeddings/base.py \
  backend/app/embeddings/bge.py backend/app/embeddings/fake.py \
  backend/tests/unit/test_embeddings.py
git commit -m "feat: add BGE embedding provider"
```

### Task 4: Database Schema, Migrations, and Repositories

**Files:**
- Create: `backend/app/db/__init__.py`
- Create: `backend/app/db/base.py`
- Create: `backend/app/db/models.py`
- Create: `backend/app/db/session.py`
- Create: `backend/app/db/repositories.py`
- Create: `backend/alembic.ini`
- Create: `backend/alembic/env.py`
- Create: `backend/alembic/script.py.mako`
- Create: `backend/alembic/versions/0001_initial.py`
- Create: `backend/tests/integration/test_database.py`

- [x] **Step 1: Add models**

`documents`:

- UUID primary key.
- `tenant_id`, `knowledge_base_id`.
- `doc_id`, `source_type`, `title`, `content_hash`, `metadata`.
- Unique `(tenant_id, knowledge_base_id, source_type, doc_id, content_hash)`.

`chunks`:

- UUID primary key and `document_id` foreign key.
- `tenant_id`, `knowledge_base_id`, `chunk_id`, `chunk_index`.
- `content`, `token_count`, `start_char`, `end_char`, `content_hash`.
- `source_type`, `title`, `metadata`.
- Nullable `embedding vector(1024)` and `embedding_model`.
- Unique `(tenant_id, knowledge_base_id, chunk_id)`.
- Indexes for document lookup and source filtering.
- HNSW cosine index on `embedding`.

- [x] **Step 2: Add migrations**

Use Alembic. The initial migration must run `CREATE EXTENSION IF NOT EXISTS vector`,
create both tables, and create the HNSW index with `vector_cosine_ops`.

- [x] **Step 3: Add async session and repositories**

Provide an async engine/session factory from settings. Repository methods:

- Upsert a document and return its UUID.
- Upsert a batch of chunks.
- Find chunk IDs that already have embeddings.
- Count documents, chunks, and embedded chunks.
- Search by cosine distance with tenant, knowledge-base, and optional source filters.

- [x] **Step 4: Test**

Unit-test settings and query construction. Add integration tests that skip cleanly when
PostgreSQL is unavailable. Integration tests run migrations and verify:

- Extension, tables, constraints, and HNSW index exist.
- Upserts are idempotent.
- A deterministic fake-vector search returns the expected cosine order.

- [x] **Step 5: Verify**

```bash
uv run alembic -c backend/alembic.ini upgrade head
uv run pytest backend/tests/integration/test_database.py -v -m integration
```

- [x] **Step 6: Commit**

```bash
git add backend/app/db backend/alembic.ini backend/alembic \
  backend/tests/integration/test_database.py
git commit -m "feat: add pgvector schema and repositories"
```

### Task 5: Embedding and Load CLI

**Files:**
- Create: `scripts/embed_and_load.py`
- Create: `backend/tests/integration/test_embedding_load.py`

- [x] **Step 1: Implement the loader**

Stream `chunks.jsonl` in batches. For each batch:

1. Upsert document rows keyed by `(source_type, doc_id, document_content_hash)`.
2. Find chunks that already have embeddings and skip re-embedding them.
3. Embed new chunk contents in batches.
4. Upsert chunks with `embedding`, `embedding_model`, and metadata.

Support `--chunks`, `--batch-size`, `--device`, `--limit`, `--model`, and
`--tenant-id`/`--knowledge-base-id` overrides.

- [x] **Step 2: Make it idempotent**

A second run over the same corpus must not duplicate documents or chunks and must not
re-embed chunks that already have embeddings. Transactions should commit per batch.

- [x] **Step 3: Test**

Use a fake provider and a disposable database. Verify document/chunk counts, embedding
dimensions, rerun idempotency, and skip behavior.

- [x] **Step 4: Commit**

```bash
git add backend/app/embeddings/loader.py scripts/embed_and_load.py \
  backend/tests/integration/test_embedding_load.py
git commit -m "feat: add idempotent embedding load pipeline"
```

### Task 6: Dense Retrieval and Development Metrics

**Files:**
- Create: `backend/app/retrieval/__init__.py`
- Create: `backend/app/retrieval/dense.py`
- Create: `backend/app/retrieval/metrics.py`
- Create: `scripts/dense_search.py`
- Create: `scripts/eval_dense_dev.py`
- Create: `backend/tests/unit/test_retrieval_metrics.py`
- Create: `backend/tests/integration/test_dense_retrieval.py`

- [x] **Step 1: Implement the retriever**

`DenseRetriever.retrieve(query, top_k=10, source_types=None)` embeds the query, runs
pgvector cosine search, and returns chunk ID, document ID, source type, title, chunk
index, content, and score.

The retriever must not accept evaluation metadata.

- [x] **Step 2: Add search CLI**

```bash
uv run python scripts/dense_search.py "How do I rotate an API key?" --top-k 5
```

- [x] **Step 3: Add development metrics**

Read `data/sample/dev_questions.jsonl`. For each question:

1. Retrieve top-k using only the question text.
2. Deduplicate retrieved chunks by `doc_id`.
3. Compare against `expected_doc_ids` after retrieval.
4. Compute recall@1/5/10/20 and MRR.

Write `data/processed/dense_dev_metrics.json` (gitignored). Do not run the holdout set.

- [x] **Step 4: Test**

Cover recall/MRR calculation, doc-level deduplication, empty results, source filters,
and the retriever interface. Database tests use deterministic fake vectors.

- [x] **Step 5: Commit**

```bash
git add backend/app/retrieval scripts/dense_search.py scripts/eval_dense_dev.py \
  backend/tests/unit/test_dense_retriever.py backend/tests/unit/test_retrieval_metrics.py \
  backend/tests/integration/test_dense_retrieval.py
git commit -m "feat: add dense retrieval and development metrics"
```

### Task 7: Full Embedding Run and Verification

- [x] **Step 1: Start PostgreSQL**

```bash
docker compose up -d db
uv run alembic -c backend/alembic.ini upgrade head
```

- [x] **Step 2: Run the full load**

```bash
uv run python scripts/embed_and_load.py \
  --chunks data/processed/chunks.jsonl \
  --device cuda \
  --batch-size 32
```

If HNSW insertion is too slow, drop the index before the bulk load and recreate it
afterward. Record the chosen procedure.

- [x] **Step 3: Verify database state**

- 34,378 documents.
- 136,383 chunks.
- 136,383 non-null embeddings.
- Every vector has 1024 dimensions.
- Every chunk references a document.
- HNSW cosine index exists.
- A rerun adds no rows and skips existing embeddings.

- [x] **Step 4: Run development retrieval metrics**

```bash
uv run python scripts/eval_dense_dev.py --top-k 20
```

Report recall@1/5/10/20 and MRR. These are diagnostics, not benchmark claims. The holdout
set remains untouched.

- [x] **Step 5: Verify**

```bash
uv run pytest -q
uv run pytest -q -m integration
uv run ruff check .
uv run ruff format --check .
```

### Task 8: Documentation and Completion

**Files:**
- Create: `docs/architecture/embeddings-and-retrieval.md`
- Create: `docs/design-decisions/0002-vector-store-and-embedding-runtime.md`
- Modify: `README.md`
- Append: `blog.md` (not committed)

- [x] **Step 1: Document architecture and measured results**

Cover the embedding provider, database schema, HNSW index, load idempotency, retrieval
interface, development metrics, known limitations, and label-leakage rule.

- [x] **Step 2: Update README**

Add Block 2 status, setup commands, database migration/load commands, retrieval
commands, and measured results.

- [x] **Step 3: Append the Block 2 blog entry**

Record what was built, why, what the real run measured, and what was learned. Keep
`blog.md` excluded from git.

- [x] **Step 4: Commit**

```bash
git add docs README.md plan.md
git commit -m "docs: complete block 2 embeddings and dense retrieval"
```

### Block 2 Acceptance Criteria

- `BAAI/bge-large-en-v1.5` provider works on the available GPU or a documented CPU
  fallback.
- All 136,383 chunks are loaded with 1024-dimensional embeddings.
- The database contains 34,378 documents and 136,383 chunks with no duplicates.
- HNSW cosine index exists and dense queries use it.
- Rerunning the loader does not duplicate rows or re-embed existing chunks.
- Development recall@k and MRR are reported without touching holdout data.
- The retriever interface cannot receive evaluation metadata.
- Unit, integration, and real-data checks pass.
- No BM25, hybrid retrieval, reranking, generation, API, or full evaluation code is
  added.

### Block 2 Risks

| Risk | Mitigation |
|---|---|
| RTX 5050/PyTorch CUDA incompatibility | Validate runtime before the full run; document CPU fallback |
| GPU OOM during embedding | Tune batch size; use fp16 only after measuring quality impact |
| Slow HNSW bulk insertion | Drop/recreate the index around the bulk load if needed |
| Duplicate document identity | Use `(source_type, doc_id, document_content_hash)` |
| Embedding-model mismatch | Store `embedding_model` and dimension; refuse mixed dimensions |
| Label leakage | Keep evaluation metadata outside retriever interfaces |
| Holdout overfitting | Use only the development split in Block 2 |

### Next Block Preview

Block 3 should add BM25, hybrid retrieval, and Reciprocal Rank Fusion over the same
document/chunk tables, using the Block 2 dense-retrieval metrics as a comparison point.
Do not start Block 3 until Block 2 verification is complete.

> Historical note: Block 3 instead used PostgreSQL FTS, not exact BM25; see
> `docs/design-decisions/0003-lexical-backend-and-fusion.md`.

---

## Block 3 Implementation Plan — Lexical and Hybrid Retrieval

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add PostgreSQL full-text lexical retrieval, hybrid retrieval with Reciprocal
Rank Fusion, and a development-split comparison against the Block 2 dense baseline.

**Architecture:** A generated `search_vector` column and GIN index make PostgreSQL FTS
available directly on the chunk table. A leakage-safe `LexicalRetriever` returns chunks
ranked by `ts_rank_cd`. A `HybridRetriever` retrieves candidate lists from dense and
lexical retrieval and fuses them with RRF. A unified development evaluator reports
dense, lexical, and hybrid metrics without touching the holdout split.

**Tech Stack:** Python 3.12, `uv`, PostgreSQL 16, pgvector, SQLAlchemy 2 async,
asyncpg, Alembic, sentence-transformers, PyTorch, pytest-asyncio, Ruff.

**Status:** complete. Implementation, migration, measured development metrics,
verification, and documentation are done. The holdout split was not used.

### Global Constraints

- Use PostgreSQL FTS as the lexical baseline. Do **not** call it exact BM25.
- Use `websearch_to_tsquery('english', query)` and `ts_rank_cd(..., 32)`.
- Weight title as `A` and content as `B` in the generated `search_vector`.
- Keep tenant and knowledge-base filters mandatory on every retrieval query.
- `source_types` may be a user-selected product filter; evaluation metadata may not be
  passed to any retriever.
- Never pass `expected_doc_ids`, `question_type`, `gold_answer`, or evaluation
  `source_types` to a retriever-shaped interface.
- Use RRF with `k=60`, dense weight `1.0`, lexical weight `1.0`, and candidate depth
  `50` unless development evaluation justifies a documented change.
- Use only the 169 development questions. Do not run or tune on the 42 holdout
  questions.
- Do not add exact BM25 libraries, rerankers, generation, FastAPI, or full evaluation
  harnesses in this block.
- Do not commit raw or processed data, query embeddings, evaluation reports, or model
  caches.
- Run `uv run pytest`, database integration tests, Ruff, and formatting checks before
  claiming completion.

### File Structure

| File | Responsibility |
|---|---|
| `backend/app/retrieval/types.py` | Shared `RetrievedChunk` result type |
| `backend/app/retrieval/fusion.py` | Reciprocal Rank Fusion |
| `backend/app/retrieval/lexical.py` | PostgreSQL FTS retriever |
| `backend/app/retrieval/hybrid.py` | Dense + lexical candidate fusion |
| `backend/app/db/models.py` | `search_vector` generated column and GIN index |
| `backend/app/db/repositories.py` | SQL for lexical search |
| `backend/alembic/versions/0002_add_lexical_search.py` | Schema migration |
| `backend/app/evaluation/dev.py` | Shared dev-question loading and metrics |
| `scripts/lexical_search.py` | Lexical query CLI |
| `scripts/hybrid_search.py` | Hybrid query CLI |
| `scripts/eval_retrieval_dev.py` | Dense/lexical/hybrid dev evaluation |
| `docs/architecture/hybrid-retrieval.md` | Block 3 design and measured results |
| `docs/design-decisions/0003-lexical-backend-and-fusion.md` | Accepted decisions |

---

### Task 1: Shared Retrieval Types and RRF

**Files:**
- Create: `backend/app/retrieval/types.py`
- Create: `backend/app/retrieval/fusion.py`
- Modify: `backend/app/retrieval/dense.py`
- Modify: `backend/app/retrieval/__init__.py`
- Create: `backend/tests/unit/test_fusion.py`

**Interfaces:**
- Consumes: nothing from earlier Block 3 tasks.
- Produces:
  - `RetrievedChunk(chunk_id: str, doc_id: str, source_type: str, title: str, chunk_index: int, content: str, score: float, metadata: dict[str, Any])`
  - `reciprocal_rank_fusion(rankings: Sequence[Sequence[RetrievedChunk]], *, k: int = 60, weights: Sequence[float] | None = None) -> list[RetrievedChunk]`

- [x] **Step 1: Write the failing fusion tests**

Create `backend/tests/unit/test_fusion.py`:

```python
"""Tests for Reciprocal Rank Fusion."""

from __future__ import annotations

import pytest

from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.types import RetrievedChunk


def _chunk(chunk_id: str, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        source_type="confluence",
        title="Title",
        chunk_index=0,
        content="content",
        score=score,
        metadata={},
    )


def test_rrf_combines_overlapping_rankings() -> None:
    fused = reciprocal_rank_fusion(
        [[_chunk("a"), _chunk("b")], [_chunk("b"), _chunk("c")]],
        k=60,
    )

    assert [chunk.chunk_id for chunk in fused] == ["b", "a", "c"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_honors_weights() -> None:
    fused = reciprocal_rank_fusion(
        [[_chunk("a"), _chunk("b")], [_chunk("b"), _chunk("c")]],
        k=60,
        weights=[2.0, 1.0],
    )

    assert [chunk.chunk_id for chunk in fused] == ["b", "a", "c"]


def test_rrf_counts_a_chunk_once_per_ranking() -> None:
    fused = reciprocal_rank_fusion(
        [[_chunk("a"), _chunk("a"), _chunk("b")]],
        k=60,
    )

    assert [chunk.chunk_id for chunk in fused] == ["a", "b"]
    assert fused[0].score == pytest.approx(1 / 61)


def test_rrf_is_deterministic_on_ties() -> None:
    fused = reciprocal_rank_fusion([[_chunk("b")], [_chunk("a")]], k=60)

    assert [chunk.chunk_id for chunk in fused] == ["a", "b"]


def test_rrf_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="k"):
        reciprocal_rank_fusion([[_chunk("a")]], k=0)
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion([[_chunk("a")]], weights=[1.0, 2.0])
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion([[_chunk("a")]], weights=[-1.0])
```

- [x] **Step 2: Run the tests and verify they fail**

Run:

```bash
uv run pytest backend/tests/unit/test_fusion.py -q
```

Expected: FAIL because `app.retrieval.types` and `app.retrieval.fusion` do not exist.

- [x] **Step 3: Add the shared result type**

Create `backend/app/retrieval/types.py`:

```python
"""Shared retrieval result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by a retrieval strategy."""

    chunk_id: str
    doc_id: str
    source_type: str
    title: str
    chunk_index: int
    content: str
    score: float
    metadata: dict[str, Any]
```

- [x] **Step 4: Implement RRF**

Create `backend/app/retrieval/fusion.py`:

```python
"""Reciprocal Rank Fusion across retrieval strategies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from app.retrieval.types import RetrievedChunk


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[RetrievedChunk]:
    """Fuse ranked result lists using Reciprocal Rank Fusion."""

    if k <= 0:
        raise ValueError("k must be positive")
    resolved_weights = list(weights) if weights is not None else [1.0] * len(rankings)
    if len(resolved_weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")
    if any(weight < 0 for weight in resolved_weights):
        raise ValueError("weights must be non-negative")

    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    chunks: dict[str, RetrievedChunk] = {}

    for ranking, weight in zip(rankings, resolved_weights, strict=True):
        if weight == 0:
            continue
        seen: set[str] = set()
        for rank, chunk in enumerate(ranking, start=1):
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            chunks.setdefault(chunk.chunk_id, chunk)
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + weight / (k + rank)
            best_rank[chunk.chunk_id] = min(best_rank.get(chunk.chunk_id, rank), rank)

    fused = [
        replace(chunks[chunk_id], score=scores[chunk_id])
        for chunk_id in scores
    ]
    fused.sort(key=lambda chunk: (-chunk.score, best_rank[chunk.chunk_id], chunk.chunk_id))
    return fused
```

- [x] **Step 5: Update dense retrieval to use the shared type**

In `backend/app/retrieval/dense.py`, remove the local `RetrievedChunk` dataclass,
remove the now-unused `from dataclasses import dataclass` import, and add:

```python
from app.retrieval.types import RetrievedChunk
```

Keep `DenseRetriever` and its behavior unchanged.

Update `backend/app/retrieval/__init__.py`:

```python
"""Retrieval interfaces and metrics."""

from app.retrieval.dense import DenseRetriever
from app.retrieval.types import RetrievedChunk

__all__ = ["DenseRetriever", "RetrievedChunk"]
```

- [x] **Step 6: Run the tests and verify they pass**

Run:

```bash
uv run pytest backend/tests/unit/test_fusion.py backend/tests/unit/test_dense_retriever.py -q
uv run ruff check backend/app/retrieval backend/tests/unit/test_fusion.py
uv run ruff format --check backend/app/retrieval backend/tests/unit/test_fusion.py
```

Expected: PASS and no lint errors.

- [x] **Step 7: Commit**

```bash
git add backend/app/retrieval/types.py backend/app/retrieval/fusion.py \
  backend/app/retrieval/dense.py backend/app/retrieval/__init__.py \
  backend/tests/unit/test_fusion.py
git commit -m "feat: add shared retrieval types and RRF fusion"
```

---

### Task 2: Lexical Search Schema

**Files:**
- Modify: `backend/app/db/models.py`
- Create: `backend/alembic/versions/0002_add_lexical_search.py`
- Modify: `backend/tests/integration/test_database.py`

**Interfaces:**
- Consumes: existing `ChunkRecord` model and Alembic `0001`.
- Produces:
  - `ChunkRecord.search_vector` generated `tsvector` column.
  - Index `ix_chunks_search_vector` using GIN.

- [x] **Step 1: Write the failing schema test**

Add this test to `backend/tests/integration/test_database.py`:

```python
async def test_lexical_search_schema_exists(session) -> None:
    generated = await session.scalar(
        text(
            "SELECT is_generated FROM information_schema.columns "
            "WHERE table_name = 'chunks' AND column_name = 'search_vector'"
        )
    )
    index = await session.scalar(
        text(
            "SELECT indexname FROM pg_indexes "
            "WHERE tablename = 'chunks' AND indexname = 'ix_chunks_search_vector'"
        )
    )

    assert generated == "ALWAYS"
    assert index == "ix_chunks_search_vector"
```

- [x] **Step 2: Run the test and verify it fails**

Run:

```bash
uv run pytest backend/tests/integration/test_database.py::test_lexical_search_schema_exists -q
```

Expected: FAIL because `search_vector` and the GIN index do not exist.

- [x] **Step 3: Add the generated column and GIN index to the model**

In `backend/app/db/models.py`, import `Computed` and `TSVECTOR`:

```python
from sqlalchemy import (
    Computed,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
```

Add a module constant near `EMBEDDING_DIMENSION`:

```python
SEARCH_VECTOR_EXPRESSION = (
    "setweight(to_tsvector('english'::regconfig, coalesce(title, '')), 'A') || "
    "setweight(to_tsvector('english'::regconfig, coalesce(content, '')), 'B')"
)
```

Add the index to `ChunkRecord.__table_args__`:

```python
Index(
    "ix_chunks_search_vector",
    "search_vector",
    postgresql_using="gin",
),
```

Add the column after `content`:

```python
search_vector: Mapped[str | None] = mapped_column(
    TSVECTOR,
    Computed(SEARCH_VECTOR_EXPRESSION, persisted=True),
    nullable=True,
)
```

- [x] **Step 4: Add the migration**

Create `backend/alembic/versions/0002_add_lexical_search.py`:

```python
"""Add generated lexical search vector and GIN index.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-08
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SEARCH_VECTOR_EXPRESSION = (
    "setweight(to_tsvector('english'::regconfig, coalesce(title, '')), 'A') || "
    "setweight(to_tsvector('english'::regconfig, coalesce(content, '')), 'B')"
)


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column(
            "search_vector",
            postgresql.TSVECTOR(),
            sa.Computed(SEARCH_VECTOR_EXPRESSION, persisted=True),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_chunks_search_vector",
        "chunks",
        ["search_vector"],
        unique=False,
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_search_vector", table_name="chunks")
    op.drop_column("chunks", "search_vector")
```

- [x] **Step 5: Apply the migration**

Run:

```bash
docker compose up -d db
uv run alembic -c backend/alembic.ini upgrade head
```

Expected: migration `0002` applies without error.

- [x] **Step 6: Run the schema test and verify it passes**

Run:

```bash
uv run pytest backend/tests/integration/test_database.py::test_lexical_search_schema_exists -q
```

Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add backend/app/db/models.py backend/alembic/versions/0002_add_lexical_search.py \
  backend/tests/integration/test_database.py
git commit -m "feat: add lexical search schema"
```

---

### Task 3: Lexical Repository Query

**Files:**
- Modify: `backend/app/db/repositories.py`
- Create: `backend/tests/integration/test_lexical_retrieval.py`

**Interfaces:**
- Consumes: `ChunkRecord.search_vector`, `DocumentUpsert`, `ChunkUpsert`, `upsert_documents`, `upsert_chunks`.
- Produces:
  - `LexicalSearchResult(chunk: ChunkRecord, rank: float)`
  - `search_chunks_lexical(session, query, *, tenant_id, knowledge_base_id, top_k=10, source_types=None) -> list[LexicalSearchResult]`

- [x] **Step 1: Write the failing lexical repository test**

Create `backend/tests/integration/test_lexical_retrieval.py`:

```python
"""Integration tests for PostgreSQL lexical retrieval."""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.repositories import (
    ChunkUpsert,
    DocumentUpsert,
    search_chunks_lexical,
    upsert_chunks,
    upsert_documents,
)
from app.settings import Settings

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def session_factory():
    engine = create_async_engine(Settings().database_url)
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1 FROM documents LIMIT 1"))
            await connection.execute(text("SELECT 1 FROM chunks LIMIT 1"))
    except Exception as exc:
        await engine.dispose()
        pytest.skip(f"PostgreSQL schema unavailable: {exc}")

    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


async def _seed_chunk(
    session,
    *,
    tenant_id: uuid.UUID,
    knowledge_base_id: uuid.UUID,
    doc_id: str,
    content: str,
    title: str,
    source_type: str,
) -> None:
    document = DocumentUpsert(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        doc_id=doc_id,
        source_type=source_type,
        title=title,
        content_hash="a" * 64,
    )
    document_id = (await upsert_documents(session, [document]))[
        (document.source_type, document.doc_id, document.content_hash)
    ]
    await upsert_chunks(
        session,
        [
            ChunkUpsert(
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
                document_id=document_id,
                doc_id=doc_id,
                chunk_id=f"{doc_id}:0:abc",
                chunk_index=0,
                content=content,
                token_count=len(content.split()),
                start_char=0,
                end_char=len(content),
                content_hash="b" * 64,
                source_type=source_type,
                title=title,
                embedding=None,
                embedding_model=None,
            )
        ],
    )
    await session.commit()


async def test_lexical_search_orders_by_rank(session_factory) -> None:
    tenant_id = uuid.uuid4()
    knowledge_base_id = uuid.uuid4()
    async with session_factory() as session:
        await _seed_chunk(
            session,
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            doc_id="doc-alpha",
            content="alpha beta gamma",
            title="Alpha guide",
            source_type="confluence",
        )
        await _seed_chunk(
            session,
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            doc_id="doc-delta",
            content="delta epsilon",
            title="Delta guide",
            source_type="confluence",
        )

    async with session_factory() as session:
        results = await search_chunks_lexical(
            session,
            "alpha",
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            top_k=5,
        )

    assert [result.chunk.doc_id for result in results] == ["doc-alpha"]
    assert results[0].rank > 0


async def test_lexical_search_honors_source_filter(session_factory) -> None:
    tenant_id = uuid.uuid4()
    knowledge_base_id = uuid.uuid4()
    async with session_factory() as session:
        await _seed_chunk(
            session,
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            doc_id="doc-confluence",
            content="rotation policy",
            title="Rotation",
            source_type="confluence",
        )
        await _seed_chunk(
            session,
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            doc_id="doc-github",
            content="rotation policy",
            title="Rotation",
            source_type="github",
        )

    async with session_factory() as session:
        results = await search_chunks_lexical(
            session,
            "rotation",
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            top_k=5,
            source_types=["github"],
        )

    assert [result.chunk.doc_id for result in results] == ["doc-github"]
```

- [x] **Step 2: Run the tests and verify they fail**

Run:

```bash
uv run pytest backend/tests/integration/test_lexical_retrieval.py -q
```

Expected: FAIL because `search_chunks_lexical` and `LexicalSearchResult` do not exist.

- [x] **Step 3: Add the result type and query**

In `backend/app/db/repositories.py`, add after `ChunkSearchResult`:

```python
@dataclass(frozen=True)
class LexicalSearchResult:
    chunk: ChunkRecord
    rank: float
```

Add after `search_chunks`:

```python
async def search_chunks_lexical(
    session: AsyncSession,
    query: str,
    *,
    tenant_id: uuid.UUID,
    knowledge_base_id: uuid.UUID,
    top_k: int = 10,
    source_types: Sequence[str] | None = None,
) -> list[LexicalSearchResult]:
    """Return chunks ranked by PostgreSQL full-text search."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not query.strip():
        return []

    tsquery = func.websearch_to_tsquery("english", query)
    rank = func.ts_rank_cd(ChunkRecord.search_vector, tsquery, 32).label("rank")
    statement = (
        select(ChunkRecord, rank)
        .where(
            ChunkRecord.tenant_id == tenant_id,
            ChunkRecord.knowledge_base_id == knowledge_base_id,
            ChunkRecord.search_vector.op("@@")(tsquery),
        )
        .order_by(rank.desc(), ChunkRecord.chunk_id.asc())
        .limit(top_k)
    )
    if source_types:
        statement = statement.where(ChunkRecord.source_type.in_(list(source_types)))

    rows = (await session.execute(statement)).all()
    return [
        LexicalSearchResult(chunk=row[0], rank=float(row[1]))
        for row in rows
    ]
```

- [x] **Step 4: Run the integration tests and verify they pass**

Run:

```bash
uv run pytest backend/tests/integration/test_lexical_retrieval.py -q
uv run ruff check backend/app/db/repositories.py backend/tests/integration/test_lexical_retrieval.py
uv run ruff format --check backend/app/db/repositories.py backend/tests/integration/test_lexical_retrieval.py
```

Expected: PASS and no lint errors.

- [x] **Step 5: Commit**

```bash
git add backend/app/db/repositories.py backend/tests/integration/test_lexical_retrieval.py
git commit -m "feat: add lexical search repository query"
```

---

### Task 4: Lexical Retriever

**Files:**
- Create: `backend/app/retrieval/lexical.py`
- Modify: `backend/app/retrieval/__init__.py`
- Create: `backend/tests/unit/test_lexical_retriever.py`

**Interfaces:**
- Consumes: `search_chunks_lexical` from Task 3 and `RetrievedChunk` from Task 1.
- Produces:
  - `LexicalRetriever(session_factory, *, tenant_id, knowledge_base_id)`
  - `LexicalRetriever.retrieve(query, *, top_k=10, source_types=None) -> list[RetrievedChunk]`

- [x] **Step 1: Write the failing unit tests**

Create `backend/tests/unit/test_lexical_retriever.py`:

```python
"""Tests for the lexical retriever boundary."""

from __future__ import annotations

import inspect
import uuid
from types import SimpleNamespace

import pytest

from app.db.repositories import LexicalSearchResult
from app.retrieval.lexical import LexicalRetriever


class FakeSession:
    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


def _session_factory() -> FakeSession:
    return FakeSession()


@pytest.mark.asyncio
async def test_retrieve_maps_lexical_results(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = SimpleNamespace(
        chunk_id="chunk-1",
        doc_id="doc-1",
        source_type="confluence",
        title="Title",
        chunk_index=2,
        content="content",
        doc_metadata={"origin": "test"},
    )
    calls: list[tuple[object, str, dict[str, object]]] = []

    async def fake_search(session: object, query: str, **kwargs: object):
        calls.append((session, query, kwargs))
        return [LexicalSearchResult(chunk=chunk, rank=0.75)]

    monkeypatch.setattr("app.retrieval.lexical.search_chunks_lexical", fake_search)
    retriever = LexicalRetriever(
        _session_factory,
        tenant_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        knowledge_base_id=uuid.UUID("22222222-2222-2222-2222-222222222222"),
    )

    results = await retriever.retrieve("query", top_k=3, source_types=["confluence"])

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"
    assert results[0].score == 0.75
    assert results[0].metadata == {"origin": "test"}
    _, query, kwargs = calls[0]
    assert query == "query"
    assert kwargs["top_k"] == 3
    assert kwargs["source_types"] == ["confluence"]


@pytest.mark.asyncio
async def test_empty_query_does_not_query_database() -> None:
    retriever = LexicalRetriever(
        _session_factory,
        tenant_id=uuid.uuid4(),
        knowledge_base_id=uuid.uuid4(),
    )

    assert await retriever.retrieve("   ") == []


@pytest.mark.asyncio
async def test_invalid_top_k_is_rejected() -> None:
    retriever = LexicalRetriever(
        _session_factory,
        tenant_id=uuid.uuid4(),
        knowledge_base_id=uuid.uuid4(),
    )

    with pytest.raises(ValueError, match="top_k"):
        await retriever.retrieve("query", top_k=0)


def test_retriever_does_not_expose_evaluation_metadata() -> None:
    parameters = inspect.signature(LexicalRetriever.retrieve).parameters

    for name in ("expected_doc_ids", "question_type", "gold_answer", "source_types_expected"):
        assert name not in parameters
```

- [x] **Step 2: Run the tests and verify they fail**

Run:

```bash
uv run pytest backend/tests/unit/test_lexical_retriever.py -q
```

Expected: FAIL because `app.retrieval.lexical` does not exist.

- [x] **Step 3: Implement the retriever**

Create `backend/app/retrieval/lexical.py`:

```python
"""Lexical retrieval over PostgreSQL full-text search."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.repositories import search_chunks_lexical
from app.retrieval.types import RetrievedChunk


class LexicalRetriever:
    """Run a PostgreSQL full-text query and return ranked chunks."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        tenant_id: uuid.UUID,
        knowledge_base_id: uuid.UUID,
    ) -> None:
        self.session_factory = session_factory
        self.tenant_id = tenant_id
        self.knowledge_base_id = knowledge_base_id

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        source_types: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks using only user-provided inputs."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not query.strip():
            return []

        async with self.session_factory() as session:
            results = await search_chunks_lexical(
                session,
                query,
                tenant_id=self.tenant_id,
                knowledge_base_id=self.knowledge_base_id,
                top_k=top_k,
                source_types=source_types,
            )

        return [
            RetrievedChunk(
                chunk_id=result.chunk.chunk_id,
                doc_id=result.chunk.doc_id,
                source_type=result.chunk.source_type,
                title=result.chunk.title,
                chunk_index=result.chunk.chunk_index,
                content=result.chunk.content,
                score=result.rank,
                metadata=result.chunk.doc_metadata,
            )
            for result in results
        ]
```

Update `backend/app/retrieval/__init__.py`:

```python
"""Retrieval interfaces and metrics."""

from app.retrieval.dense import DenseRetriever
from app.retrieval.lexical import LexicalRetriever
from app.retrieval.types import RetrievedChunk

__all__ = ["DenseRetriever", "LexicalRetriever", "RetrievedChunk"]
```

- [x] **Step 4: Run the tests and verify they pass**

Run:

```bash
uv run pytest backend/tests/unit/test_lexical_retriever.py -q
uv run ruff check backend/app/retrieval/lexical.py backend/app/retrieval/__init__.py \
  backend/tests/unit/test_lexical_retriever.py
uv run ruff format --check backend/app/retrieval/lexical.py \
  backend/tests/unit/test_lexical_retriever.py
```

Expected: PASS and no lint errors.

- [x] **Step 5: Commit**

```bash
git add backend/app/retrieval/lexical.py backend/app/retrieval/__init__.py \
  backend/tests/unit/test_lexical_retriever.py
git commit -m "feat: add lexical retriever"
```

---

### Task 5: Hybrid Retriever

**Files:**
- Create: `backend/app/retrieval/hybrid.py`
- Modify: `backend/app/retrieval/__init__.py`
- Create: `backend/tests/unit/test_hybrid_retriever.py`

**Interfaces:**
- Consumes: `DenseRetriever`, `LexicalRetriever`, `RetrievedChunk`, and `reciprocal_rank_fusion`.
- Produces:
  - `HybridRetriever(dense_retriever, lexical_retriever, *, rrf_k=60, dense_weight=1.0, lexical_weight=1.0, candidate_k=50)`
  - `HybridRetriever.retrieve(query, *, top_k=10, source_types=None) -> list[RetrievedChunk]`

- [x] **Step 1: Write the failing unit tests**

Create `backend/tests/unit/test_hybrid_retriever.py`:

```python
"""Tests for hybrid retrieval and RRF integration."""

from __future__ import annotations

import inspect
import uuid

import pytest

from app.retrieval.hybrid import HybridRetriever
from app.retrieval.types import RetrievedChunk


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[dict[str, object]] = []

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        source_types: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append(
            {"query": query, "top_k": top_k, "source_types": source_types}
        )
        return self.chunks[:top_k]


def _chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        source_type="github",
        title="Title",
        chunk_index=0,
        content="content",
        score=0.0,
        metadata={},
    )


@pytest.mark.asyncio
async def test_hybrid_fuses_and_slices_results() -> None:
    dense = FakeRetriever([_chunk("a"), _chunk("b")])
    lexical = FakeRetriever([_chunk("b"), _chunk("c")])
    retriever = HybridRetriever(
        dense,
        lexical,
        rrf_k=60,
        candidate_k=2,
    )

    results = await retriever.retrieve("query", top_k=2, source_types=["github"])

    assert [result.chunk_id for result in results] == ["b", "a"]
    assert dense.calls[0]["top_k"] == 2
    assert lexical.calls[0]["top_k"] == 2
    assert dense.calls[0]["source_types"] == ["github"]
    assert lexical.calls[0]["source_types"] == ["github"]


@pytest.mark.asyncio
async def test_candidate_k_expands_to_top_k() -> None:
    dense = FakeRetriever([_chunk("a")])
    lexical = FakeRetriever([_chunk("b")])
    retriever = HybridRetriever(dense, lexical, candidate_k=1)

    await retriever.retrieve("query", top_k=5)

    assert dense.calls[0]["top_k"] == 5
    assert lexical.calls[0]["top_k"] == 5


@pytest.mark.asyncio
async def test_empty_query_does_not_call_retrievers() -> None:
    dense = FakeRetriever([])
    lexical = FakeRetriever([])
    retriever = HybridRetriever(dense, lexical)

    assert await retriever.retrieve("   ") == []
    assert dense.calls == []
    assert lexical.calls == []


def test_hybrid_rejects_invalid_parameters() -> None:
    dense = FakeRetriever([])
    lexical = FakeRetriever([])

    with pytest.raises(ValueError, match="rrf_k"):
        HybridRetriever(dense, lexical, rrf_k=0)
    with pytest.raises(ValueError, match="candidate_k"):
        HybridRetriever(dense, lexical, candidate_k=0)
    with pytest.raises(ValueError, match="weights"):
        HybridRetriever(dense, lexical, dense_weight=-1.0)


def test_retriever_does_not_expose_evaluation_metadata() -> None:
    parameters = inspect.signature(HybridRetriever.retrieve).parameters

    for name in ("expected_doc_ids", "question_type", "gold_answer", "source_types_expected"):
        assert name not in parameters
```

- [x] **Step 2: Run the tests and verify they fail**

Run:

```bash
uv run pytest backend/tests/unit/test_hybrid_retriever.py -q
```

Expected: FAIL because `app.retrieval.hybrid` does not exist.

- [x] **Step 3: Implement the hybrid retriever**

Create `backend/app/retrieval/hybrid.py`:

```python
"""Hybrid retrieval using RRF over dense and lexical candidates."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.lexical import LexicalRetriever
from app.retrieval.types import RetrievedChunk


class HybridRetriever:
    """Fuse dense and lexical rankings with Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        lexical_retriever: LexicalRetriever,
        *,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        lexical_weight: float = 1.0,
        candidate_k: int = 50,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        if dense_weight < 0 or lexical_weight < 0:
            raise ValueError("weights must be non-negative")

        self.dense_retriever = dense_retriever
        self.lexical_retriever = lexical_retriever
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight
        self.candidate_k = candidate_k

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        source_types: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve and fuse dense and lexical results."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not query.strip():
            return []

        candidate_k = max(self.candidate_k, top_k)
        dense_results, lexical_results = await asyncio.gather(
            self.dense_retriever.retrieve(
                query,
                top_k=candidate_k,
                source_types=source_types,
            ),
            self.lexical_retriever.retrieve(
                query,
                top_k=candidate_k,
                source_types=source_types,
            ),
        )
        fused = reciprocal_rank_fusion(
            [dense_results, lexical_results],
            k=self.rrf_k,
            weights=[self.dense_weight, self.lexical_weight],
        )
        return fused[:top_k]
```

Update `backend/app/retrieval/__init__.py`:

```python
"""Retrieval interfaces and metrics."""

from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.lexical import LexicalRetriever
from app.retrieval.types import RetrievedChunk

__all__ = ["DenseRetriever", "HybridRetriever", "LexicalRetriever", "RetrievedChunk"]
```

- [x] **Step 4: Run the tests and verify they pass**

Run:

```bash
uv run pytest backend/tests/unit/test_hybrid_retriever.py -q
uv run ruff check backend/app/retrieval/hybrid.py backend/app/retrieval/__init__.py \
  backend/tests/unit/test_hybrid_retriever.py
uv run ruff format --check backend/app/retrieval/hybrid.py \
  backend/tests/unit/test_hybrid_retriever.py
```

Expected: PASS and no lint errors.

- [x] **Step 5: Commit**

```bash
git add backend/app/retrieval/hybrid.py backend/app/retrieval/__init__.py \
  backend/tests/unit/test_hybrid_retriever.py
git commit -m "feat: add hybrid retriever with RRF"
```

---

### Task 6: Lexical Search CLI

**Files:**
- Create: `scripts/lexical_search.py`

**Interfaces:**
- Consumes: `LexicalRetriever`, `Settings`, database session factory.
- Produces: a CLI that prints ranked lexical results or JSON.

- [x] **Step 1: Implement the CLI**

Create `scripts/lexical_search.py`:

```python
#!/usr/bin/env python3
"""Run a PostgreSQL full-text query from the command line."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.db.session import build_engine, build_session_factory  # noqa: E402
from app.ingestion.loaders.base import SOURCE_TYPES  # noqa: E402
from app.retrieval.lexical import LexicalRetriever  # noqa: E402
from app.settings import Settings, get_settings  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--source-types", nargs="+", choices=sorted(SOURCE_TYPES))
    parser.add_argument("--tenant-id", type=uuid.UUID, default=settings.default_tenant_id)
    parser.add_argument(
        "--knowledge-base-id",
        type=uuid.UUID,
        default=settings.default_knowledge_base_id,
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return parser


async def run(args: argparse.Namespace, settings: Settings) -> int:
    engine = build_engine(settings)
    session_factory = build_session_factory(engine)
    try:
        retriever = LexicalRetriever(
            session_factory,
            tenant_id=args.tenant_id,
            knowledge_base_id=args.knowledge_base_id,
        )
        results = await retriever.retrieve(
            args.query,
            top_k=args.top_k,
            source_types=args.source_types,
        )
    finally:
        await engine.dispose()

    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2, ensure_ascii=False))
        return 0

    if not results:
        print("No results.")
        return 0

    for rank, result in enumerate(results, start=1):
        preview = " ".join(result.content.split())[:220]
        print(
            f"{rank}. rank={result.score:.6f} doc={result.doc_id} "
            f"chunk={result.chunk_id} source={result.source_type}"
        )
        print(f"   {result.title}")
        print(f"   {preview}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    try:
        return asyncio.run(run(args, settings))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 2: Smoke-test the CLI**

Run:

```bash
uv run python scripts/lexical_search.py --help
uv run python scripts/lexical_search.py "How do I rotate an API key?" --top-k 3
```

Expected: the first command prints help; the second prints ranked results with
`rank=...` scores.

- [x] **Step 3: Commit**

```bash
git add scripts/lexical_search.py
git commit -m "feat: add lexical search CLI"
```

---

### Task 7: Hybrid Search CLI

**Files:**
- Create: `scripts/hybrid_search.py`

**Interfaces:**
- Consumes: `DenseRetriever`, `LexicalRetriever`, `HybridRetriever`, `BGEEmbeddingProvider`.
- Produces: a CLI that prints fused results or JSON.

- [x] **Step 1: Implement the CLI**

Create `scripts/hybrid_search.py`:

```python
#!/usr/bin/env python3
"""Run a hybrid dense + lexical query from the command line."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.db.session import build_engine, build_session_factory  # noqa: E402
from app.embeddings.bge import BGEEmbeddingProvider  # noqa: E402
from app.ingestion.loaders.base import SOURCE_TYPES  # noqa: E402
from app.retrieval.dense import DenseRetriever  # noqa: E402
from app.retrieval.hybrid import HybridRetriever  # noqa: E402
from app.retrieval.lexical import LexicalRetriever  # noqa: E402
from app.settings import Settings, get_settings  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--lexical-weight", type=float, default=1.0)
    parser.add_argument("--source-types", nargs="+", choices=sorted(SOURCE_TYPES))
    parser.add_argument("--device", default=settings.embedding_device)
    parser.add_argument("--dtype", default=settings.embedding_dtype, choices=["float32", "float16"])
    parser.add_argument("--model", default=settings.embedding_model)
    parser.add_argument("--tenant-id", type=uuid.UUID, default=settings.default_tenant_id)
    parser.add_argument(
        "--knowledge-base-id",
        type=uuid.UUID,
        default=settings.default_knowledge_base_id,
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return parser


async def run(args: argparse.Namespace, settings: Settings) -> int:
    provider = BGEEmbeddingProvider(
        model_name=args.model,
        dimension=settings.embedding_dimension,
        device=args.device,
        batch_size=settings.embedding_batch_size,
        dtype=args.dtype,
        query_instruction=settings.embedding_query_instruction,
    )
    engine = build_engine(settings)
    session_factory = build_session_factory(engine)
    try:
        dense = DenseRetriever(
            provider,
            session_factory,
            tenant_id=args.tenant_id,
            knowledge_base_id=args.knowledge_base_id,
        )
        lexical = LexicalRetriever(
            session_factory,
            tenant_id=args.tenant_id,
            knowledge_base_id=args.knowledge_base_id,
        )
        retriever = HybridRetriever(
            dense,
            lexical,
            rrf_k=args.rrf_k,
            dense_weight=args.dense_weight,
            lexical_weight=args.lexical_weight,
            candidate_k=args.candidate_k,
        )
        results = await retriever.retrieve(
            args.query,
            top_k=args.top_k,
            source_types=args.source_types,
        )
    finally:
        await engine.dispose()

    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2, ensure_ascii=False))
        return 0

    if not results:
        print("No results.")
        return 0

    for rank, result in enumerate(results, start=1):
        preview = " ".join(result.content.split())[:220]
        print(
            f"{rank}. rrf_score={result.score:.6f} doc={result.doc_id} "
            f"chunk={result.chunk_id} source={result.source_type}"
        )
        print(f"   {result.title}")
        print(f"   {preview}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    try:
        return asyncio.run(run(args, settings))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 2: Smoke-test the CLI**

Run:

```bash
uv run python scripts/hybrid_search.py --help
uv run python scripts/hybrid_search.py "How do I rotate an API key?" \
  --top-k 3 --device cuda --dtype float16
```

Expected: ranked fused results with `rrf_score=...`.

- [x] **Step 3: Commit**

```bash
git add scripts/hybrid_search.py
git commit -m "feat: add hybrid search CLI"
```

---

### Task 8: Unified Development Evaluation

**Files:**
- Create: `backend/app/evaluation/__init__.py`
- Create: `backend/app/evaluation/dev.py`
- Create: `backend/tests/unit/test_dev_evaluation.py`
- Create: `scripts/eval_retrieval_dev.py`

**Interfaces:**
- Consumes: dense, lexical, and hybrid retrievers; `evaluate_retrieval`.
- Produces:
  - `DevQuestion(question_id: str, question: str, expected_doc_ids: list[str])`
  - `read_dev_questions(path: Path, limit: int | None = None) -> list[DevQuestion]`
  - `evaluate_retriever(retriever, questions, *, top_k: int, progress_every: int = 25, label: str = "") -> dict[str, object]`
  - CLI `scripts/eval_retrieval_dev.py` with `--mode dense|lexical|hybrid|all`.

- [x] **Step 1: Write the failing evaluation tests**

Create `backend/tests/unit/test_dev_evaluation.py`:

```python
"""Tests for development-split evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation.dev import DevQuestion, evaluate_retriever, read_dev_questions
from app.retrieval.types import RetrievedChunk


class FakeRetriever:
    async def retrieve(self, query: str, *, top_k: int = 10, source_types=None):
        return [
            RetrievedChunk(
                chunk_id=f"{query}-0",
                doc_id="gold",
                source_type="github",
                title="Gold",
                chunk_index=0,
                content="content",
                score=1.0,
                metadata={},
            )
        ]


def test_read_dev_questions_validates_and_limits(tmp_path: Path) -> None:
    path = tmp_path / "dev.jsonl"
    path.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "question": "first",
                "expected_doc_ids": ["doc-1"],
            }
        )
        + "\n"
        + json.dumps(
            {
                "question_id": "q2",
                "question": "second",
                "expected_doc_ids": ["doc-2"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    questions = read_dev_questions(path, limit=1)

    assert questions == [
        DevQuestion(question_id="q1", question="first", expected_doc_ids=["doc-1"])
    ]


def test_read_dev_questions_rejects_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"question_id": "q1"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="question"):
        read_dev_questions(path)


@pytest.mark.asyncio
async def test_evaluate_retriever_reports_metrics() -> None:
    questions = [
        DevQuestion(question_id="q1", question="gold", expected_doc_ids=["gold"]),
    ]

    report = await evaluate_retriever(FakeRetriever(), questions, top_k=5)

    assert report["metrics"]["evaluated_questions"] == 1
    assert report["metrics"]["recall_at_1"] == 1.0
    assert report["metrics"]["mrr"] == 1.0
```

- [x] **Step 2: Run the tests and verify they fail**

Run:

```bash
uv run pytest backend/tests/unit/test_dev_evaluation.py -q
```

Expected: FAIL because `app.evaluation.dev` does not exist.

- [x] **Step 3: Add the shared evaluation module**

Create `backend/app/evaluation/__init__.py`:

```python
"""Evaluation helpers."""
```

Create `backend/app/evaluation/dev.py`:

```python
"""Development-split retrieval evaluation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.retrieval.metrics import DEFAULT_K_VALUES, evaluate_retrieval
from app.retrieval.types import RetrievedChunk


@dataclass(frozen=True)
class DevQuestion:
    question_id: str
    question: str
    expected_doc_ids: list[str]


class RetrieverProtocol(Protocol):
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        source_types: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]: ...


def read_dev_questions(path: Path, limit: int | None = None) -> list[DevQuestion]:
    """Read and validate development questions."""

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    questions: list[DevQuestion] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}") from exc
            if not raw.get("question_id") or not raw.get("question"):
                raise ValueError(
                    f"question is missing question_id or question on line {line_number}"
                )
            questions.append(
                DevQuestion(
                    question_id=str(raw["question_id"]),
                    question=str(raw["question"]),
                    expected_doc_ids=[
                        str(doc_id) for doc_id in raw.get("expected_doc_ids", [])
                    ],
                )
            )
            if limit is not None and len(questions) >= limit:
                break
    return questions


async def evaluate_retriever(
    retriever: RetrieverProtocol,
    questions: Sequence[DevQuestion],
    *,
    top_k: int,
    progress_every: int = 25,
    label: str = "",
) -> dict[str, object]:
    """Evaluate one retriever over development questions."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")

    k_values = tuple(k for k in DEFAULT_K_VALUES if k <= top_k)
    expected_by_question: dict[str, list[str]] = {}
    retrieved_by_question: dict[str, list[str]] = {}
    per_question: list[dict[str, object]] = []

    for index, question in enumerate(questions, start=1):
        results = await retriever.retrieve(question.question, top_k=top_k)
        retrieved = [result.doc_id for result in results]
        expected_by_question[question.question_id] = question.expected_doc_ids
        retrieved_by_question[question.question_id] = retrieved
        per_question.append(
            {
                "question_id": question.question_id,
                "expected_doc_ids": question.expected_doc_ids,
                "retrieved_doc_ids": retrieved,
            }
        )
        if index % progress_every == 0 or index == len(questions):
            prefix = f"{label}: " if label else ""
            print(f"{prefix}evaluated {index}/{len(questions)}", flush=True)

    metrics = evaluate_retrieval(
        expected_by_question,
        retrieved_by_question,
        k_values=k_values,
    )
    return {"metrics": metrics, "per_question": per_question}
```

- [x] **Step 4: Add the evaluation CLI**

Create `scripts/eval_retrieval_dev.py`:

```python
#!/usr/bin/env python3
"""Evaluate dense, lexical, and hybrid retrieval on the development split."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.db.session import build_engine, build_session_factory  # noqa: E402
from app.embeddings.bge import BGEEmbeddingProvider  # noqa: E402
from app.evaluation.dev import evaluate_retriever, read_dev_questions  # noqa: E402
from app.retrieval.dense import DenseRetriever  # noqa: E402
from app.retrieval.hybrid import HybridRetriever  # noqa: E402
from app.retrieval.lexical import LexicalRetriever  # noqa: E402
from app.settings import Settings, get_settings  # noqa: E402

MODES = ("dense", "lexical", "hybrid")


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    parser.add_argument("--questions", type=Path, default=Path("data/sample/dev_questions.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/retrieval_dev_metrics.json"),
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--lexical-weight", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=settings.embedding_device)
    parser.add_argument("--dtype", default=settings.embedding_dtype, choices=["float32", "float16"])
    parser.add_argument("--model", default=settings.embedding_model)
    parser.add_argument("--tenant-id", type=uuid.UUID, default=settings.default_tenant_id)
    parser.add_argument(
        "--knowledge-base-id",
        type=uuid.UUID,
        default=settings.default_knowledge_base_id,
    )
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


async def run(args: argparse.Namespace, settings: Settings) -> dict[str, object]:
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    if args.candidate_k <= 0:
        raise ValueError("candidate-k must be positive")
    if args.rrf_k <= 0:
        raise ValueError("rrf-k must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if not args.questions.is_file():
        raise FileNotFoundError(f"questions file not found: {args.questions}")

    questions = read_dev_questions(args.questions, args.limit)
    if not questions:
        raise ValueError("no development questions found")

    provider = None
    if args.mode in {"dense", "hybrid"}:
        provider = BGEEmbeddingProvider(
            model_name=args.model,
            dimension=settings.embedding_dimension,
            device=args.device,
            batch_size=settings.embedding_batch_size,
            dtype=args.dtype,
            query_instruction=settings.embedding_query_instruction,
        )

    engine = build_engine(settings)
    session_factory = build_session_factory(engine)
    try:
        lexical = LexicalRetriever(
            session_factory,
            tenant_id=args.tenant_id,
            knowledge_base_id=args.knowledge_base_id,
        )
        dense = None
        if provider is not None:
            dense = DenseRetriever(
                provider,
                session_factory,
                tenant_id=args.tenant_id,
                knowledge_base_id=args.knowledge_base_id,
            )

        retriever_by_mode = {"lexical": lexical}
        if dense is not None:
            retriever_by_mode["dense"] = dense
            retriever_by_mode["hybrid"] = HybridRetriever(
                dense,
                lexical,
                rrf_k=args.rrf_k,
                dense_weight=args.dense_weight,
                lexical_weight=args.lexical_weight,
                candidate_k=args.candidate_k,
            )

        modes = MODES if args.mode == "all" else (args.mode,)
        reports: dict[str, object] = {}
        for mode in modes:
            reports[mode] = await evaluate_retriever(
                retriever_by_mode[mode],
                questions,
                top_k=args.top_k,
                progress_every=args.progress_every,
                label=mode,
            )
    finally:
        await engine.dispose()

    output: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": args.model,
        "device": provider.device if provider is not None else None,
        "dtype": provider.dtype if provider is not None else None,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "rrf_k": args.rrf_k,
        "questions_file": str(args.questions),
        "modes": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    try:
        output = asyncio.run(run(args, settings))
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for mode, report in output["modes"].items():
        metrics = report["metrics"]
        print(f"{mode}:")
        for key, value in metrics.items():
            if key == "evaluated_questions":
                print(f"  questions: {value}")
            else:
                print(f"  {key}: {value:.4f}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 5: Run unit tests, lint, and smoke-test the eval CLI**

Run:

```bash
uv run pytest backend/tests/unit/test_dev_evaluation.py -q
uv run ruff check backend/app/evaluation scripts/eval_retrieval_dev.py \
  backend/tests/unit/test_dev_evaluation.py
uv run ruff format --check backend/app/evaluation scripts/eval_retrieval_dev.py \
  backend/tests/unit/test_dev_evaluation.py
uv run python scripts/eval_retrieval_dev.py --help
```

Expected: tests pass, lint passes, and the CLI prints help.

- [x] **Step 6: Commit**

```bash
git add backend/app/evaluation backend/tests/unit/test_dev_evaluation.py \
  scripts/eval_retrieval_dev.py
git commit -m "feat: add dense lexical and hybrid dev evaluation"
```

---

### Task 9: Apply Index and Measure Real Dev Results

**Files:**
- No source files.
- Writes: `data/processed/retrieval_dev_metrics.json` (gitignored).

**Interfaces:**
- Consumes: migration `0002`, lexical and hybrid retrievers, development questions.
- Produces: measured dense/lexical/hybrid metrics and a verified GIN index plan.

- [x] **Step 1: Apply the migration**

Run:

```bash
docker compose up -d db
uv run alembic -c backend/alembic.ini upgrade head
docker exec production-rag-db psql -U rag -d rag -c "VACUUM ANALYZE chunks;"
```

Expected: migration applies and `VACUUM ANALYZE` completes.

- [x] **Step 2: Verify the GIN index is used**

Run:

```bash
docker exec production-rag-db psql -U rag -d rag -c \
  "EXPLAIN SELECT chunk_id FROM chunks WHERE search_vector @@ websearch_to_tsquery('english', 'rotate api key') ORDER BY ts_rank_cd(search_vector, websearch_to_tsquery('english', 'rotate api key'), 32) DESC LIMIT 5;"
```

Expected: the plan contains a `Bitmap Index Scan` on `ix_chunks_search_vector` or
an index-backed plan, not a sequential scan.

- [x] **Step 3: Smoke-test lexical and hybrid search**

Run:

```bash
uv run python scripts/lexical_search.py "How do I rotate an API key?" --top-k 5
uv run python scripts/hybrid_search.py "How do I rotate an API key?" \
  --top-k 5 --device cuda --dtype float16
```

Expected: relevant rotation/revocation chunks appear in both outputs.

- [x] **Step 4: Run the full development evaluation**

Run:

```bash
uv run python scripts/eval_retrieval_dev.py \
  --mode all \
  --top-k 20 \
  --candidate-k 50 \
  --rrf-k 60 \
  --device cuda \
  --dtype float16
```

Expected: the report contains `dense`, `lexical`, and `hybrid` modes, 169 evaluated
questions per mode, and all output writes to
`data/processed/retrieval_dev_metrics.json`.

- [x] **Step 5: Compare against the dense baseline**

Confirm the dense report reproduces the Block 2 baseline approximately:

```text
recall@1  0.2840
recall@5  0.4643
recall@10 0.5362
recall@20 0.5836
MRR       0.4330
```

Then record lexical and hybrid metrics. The hybrid result must not regress dense
recall@10 or MRR. If it does, inspect candidate depth and RRF weights on the
development split only, document any change, and rerun.

- [x] **Step 6: Clean integration-test rows**

If integration tests were run against the development database, remove rows outside
the default tenant:

```bash
docker exec production-rag-db psql -U rag -d rag -c \
  "DELETE FROM documents WHERE tenant_id <> '00000000-0000-0000-0000-000000000001' OR knowledge_base_id <> '00000000-0000-0000-0000-000000000002';"
docker exec production-rag-db psql -U rag -d rag -c "VACUUM ANALYZE documents;"
docker exec production-rag-db psql -U rag -d rag -c "VACUUM ANALYZE chunks;"
```

- [x] **Step 7: Commit**

No source commit for this task. The metrics file is gitignored and will be summarized
in Task 10.

---

### Task 10: Documentation and Final Verification

**Files:**
- Create: `docs/architecture/hybrid-retrieval.md`
- Modify: `docs/design-decisions/0003-lexical-backend-and-fusion.md`
- Modify: `README.md`
- Modify: `plan.md`
- Local/gitignored: `blog3.md`

**Interfaces:**
- Consumes: measured results from Task 9.
- Produces: completed Block 3 documentation and verification evidence.

- [x] **Step 1: Write the architecture document**

Create `docs/architecture/hybrid-retrieval.md` with:

- Scope and non-goals.
- The generated `search_vector` expression and GIN index.
- Lexical query flow and `ts_rank_cd` limitation.
- Dense, lexical, and hybrid component diagram.
- RRF formula and defaults.
- Development metrics table with measured values.
- Error handling and empty-query behavior.
- Limitations: English stemming, no exact BM25, RRF discards score magnitude,
  candidate depth affects recall.
- Holdout status: untouched.

- [x] **Step 2: Add measured results to the ADR**

Append a `## Measured Results` section to
`docs/design-decisions/0003-lexical-backend-and-fusion.md` with the real dense,
lexical, and hybrid recall@1/5/10/20 and MRR values, the migration/index sizes if
available, and a short decision note.

- [x] **Step 3: Update README**

Update `README.md`:

- Change the current status from Blocks 1-2 to Blocks 1-3.
- Add PostgreSQL FTS and RRF to the implemented list.
- Add a Block 3 workflow subsection with lexical, hybrid, and evaluation commands.
- Add Block 3 measured results.
- Add the new architecture and ADR links.
- Update known limitations and the roadmap.

- [x] **Step 4: Mark the plan complete**

Update the Block 3 task checkboxes in `plan.md` and add a Block 3 acceptance criteria
section and risks table. Do not check commit steps until each commit exists.

- [x] **Step 5: Append the Block 3 blog entry**

Append a Block 3 section to the local `blog3.md` (create it if needed). Keep it
gitignored. Include the measured results, the Postgres FTS vs exact BM25 tradeoff,
and the RRF lesson.

- [x] **Step 6: Run final verification**

Run:

```bash
uv lock --check
uv sync --check
uv run pytest -q
uv run pytest -q -m integration
uv run ruff check .
uv run ruff format --check .
```

Expected: all tests pass, integration tests pass, and lint/format are clean.

- [x] **Step 7: Clean integration-test rows again**

Run the cleanup commands from Task 9 Step 6 and confirm:

```bash
docker exec production-rag-db psql -U rag -d rag -tAc \
  "SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM chunks), (SELECT count(*) FROM chunks WHERE embedding IS NOT NULL);"
```

Expected: `34378|136383|136383` for the default tenant/knowledge base.

- [x] **Step 8: Commit documentation**

```bash
git add docs/architecture/hybrid-retrieval.md \
  docs/design-decisions/0003-lexical-backend-and-fusion.md README.md plan.md
git commit -m "docs: complete block 3 hybrid retrieval"
```

---

### Block 3 Acceptance Criteria

- A generated `search_vector` column exists on `chunks` and uses title weight `A`
  and content weight `B`.
- A GIN index exists and lexical queries use it.
- `LexicalRetriever` returns relevant chunks using only query text, `top_k`, and
  optional user source filters.
- `HybridRetriever` fuses dense and lexical candidates with RRF and deterministic
  tie-breaking.
- `scripts/lexical_search.py`, `scripts/hybrid_search.py`, and
  `scripts/eval_retrieval_dev.py` work.
- Development metrics are reported for dense, lexical, and hybrid modes over 169
  questions.
- Hybrid does not regress dense recall@10 or MRR.
- Unit and integration tests pass; Ruff and formatting checks pass.
- The holdout set is untouched.
- No exact BM25 library, reranker, generation, API, or full evaluation harness is
  added.

### Block 3 Risks

| Risk | Mitigation |
|---|---|
| Generated column migration rewrites a large table | Run during a quiet window; verify disk space first |
| GIN index is large or slow to build | Measure index size and build time; keep the index because query speed matters |
| PostgreSQL FTS underperforms exact BM25 | Report honestly; exact BM25 is a later, separate decision |
| RRF hides a weak retriever | Report dense, lexical, and hybrid metrics separately |
| Candidate depth limits hybrid recall | Use `candidate_k=50`; tune only on dev and document changes |
| Evaluation leakage | Keep expected doc IDs outside retriever signatures |
| Holdout overfitting | Never run holdout during Block 3 |
| Integration tests pollute the dev DB | Clean non-default tenant rows after integration runs |

### Next Block Preview

Block 4 should add evidence selection and context construction over the fused
retrieval results. Do not start Block 4 until Block 3 verification is complete and
the hybrid metrics are documented.
