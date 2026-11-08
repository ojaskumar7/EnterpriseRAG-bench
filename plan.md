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
