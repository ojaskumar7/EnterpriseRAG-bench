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

- [ ] **Step 1: Write the failing tests**

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

- [ ] **Step 2: Run the tests to confirm failure**

```bash
uv run pytest backend/tests/unit/test_normalize.py -v
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement normalization**

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

- [ ] **Step 4: Run the tests and lint**

```bash
uv run pytest backend/tests/unit/test_normalize.py -v
uv run ruff check .
```

Expected: PASS.

- [ ] **Step 5: Commit**

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

- [ ] **Step 1: Write the failing tests**

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

- [ ] **Step 2: Run the tests to confirm failure**

```bash
uv run pytest backend/tests/unit/test_chunking.py -v
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement the chunker**

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

- [ ] **Step 4: Run the tests and lint**

```bash
uv run pytest backend/tests/unit/test_chunking.py -v
uv run ruff check .
```

Expected: PASS.

- [ ] **Step 5: Commit**

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

- [ ] **Step 1: Write the failing pipeline tests**

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

- [ ] **Step 2: Run the tests to confirm failure**

```bash
uv run pytest backend/tests/integration/test_pipeline.py -v
```

Expected: FAIL because the pipeline does not exist.

- [ ] **Step 3: Implement the pipeline**

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

- [ ] **Step 4: Implement the CLI**

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

- [ ] **Step 5: Run fixture and real-subset tests**

```bash
uv run pytest backend/tests/integration/test_pipeline.py -v
uv run pytest backend/tests/integration/test_real_subset.py -v -m integration
```

Mark the real-subset test so it skips cleanly when `data/raw/` is absent.

- [ ] **Step 6: Run the real ingestion**

```bash
uv run python scripts/ingest_subset.py --raw-dir data/raw --output-dir data/processed
```

Expected: `data/processed/chunks.jsonl` and `data/processed/stats.json` are created.

- [ ] **Step 7: Validate the output**

Check:
- Every chunk has a valid `doc_id` and `chunk_id`.
- No chunk exceeds 400 tokens.
- All chunk IDs are unique.
- Every processed document produced at least one chunk or was counted as skipped/failed.
- Rerunning produces identical chunk IDs.

- [ ] **Step 8: Commit**

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

- [ ] **Step 1: Generate EDA statistics**

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

- [ ] **Step 2: Update `docs/architecture/ingestion.md`**

Document:
- Pipeline flow.
- Loader interfaces.
- Normalization rules.
- Chunking rationale and defaults.
- Metadata fields.
- Manifest rules and label-leakage warning.
- Known limitations.

- [ ] **Step 3: Update the decision record**

Add actual counts from the real run. Mark estimates as estimates and measured values as measured.

- [ ] **Step 4: Update `README.md`**

Add current status, setup commands, ingestion commands, and known limitations.

- [ ] **Step 5: Commit**

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

- [ ] **Step 1: Run the full test suite**

```bash
uv run pytest -q
uv run pytest -q -m integration
```

Expected: all non-skipped tests pass.

- [ ] **Step 2: Run lint and formatting checks**

```bash
uv run ruff check .
uv run ruff format --check .
```

Expected: no errors.

- [ ] **Step 3: Verify the real outputs**

Check:
- `data/processed/chunks.jsonl` exists and is non-empty.
- `data/processed/stats.json` exists.
- `data/sample/subset_manifest.json` exists.
- Manifest has no missing gold documents.
- No raw or processed data is staged in git.

- [ ] **Step 4: Record known limitations**

Include:
- Block 1 uses a four-source subset, not the full corpus.
- PDF/OCR is not implemented.
- Metadata-aware retrieval is not implemented.
- Chunking is a fixed baseline, not experimentally tuned.
- No retrieval or generation evaluation has been run.
- No benchmark results are claimed.

- [ ] **Step 5: Commit**

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

- [ ] Git repository initialized.
- [ ] `uv` environment and lockfile created.
- [ ] Ruff and pytest configured and passing.
- [ ] Official four-source subset downloaded reproducibly.
- [ ] Download manifest records sizes, hashes, and file counts.
- [ ] Question manifest selects only questions with complete gold-document coverage.
- [ ] Zero-gold questions routed to a separate abstention set.
- [ ] Stratified 80/20 split created with a fixed seed.
- [ ] Document and chunk models implemented and validated.
- [ ] Zip and parquet loaders implemented with safety checks.
- [ ] Normalization deterministic and tested.
- [ ] Chunking token-bounded, deterministic, and offset-preserving.
- [ ] Pipeline writes `chunks.jsonl` and `stats.json`.
- [ ] Reruns are idempotent.
- [ ] EDA report contains real measured values.
- [ ] README and ingestion documentation updated.
- [ ] Known limitations documented.
- [ ] No embedding, database, retrieval, LLM, or API code added.
- [ ] No benchmark results fabricated.

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

## Next Block Preview

Block 2 should add:

- Embedding provider interface.
- BGE-large-en-v1.5 embedding generation.
- PostgreSQL and pgvector via Docker.
- Document and chunk tables with tenant/knowledge-base placeholders.
- Dense retrieval over the Block 1 chunk corpus.
- Retrieval tests on the development question split.

Do not start Block 2 until Block 1 verification is complete.
