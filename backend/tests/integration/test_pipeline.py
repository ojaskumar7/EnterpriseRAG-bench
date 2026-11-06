from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

from app.config import IngestionConfig
from app.ingestion.chunking.token_chunker import TokenChunker
from app.ingestion.loaders.base import LoaderIssue
from app.ingestion.loaders.zip_loader import ZipDocumentLoader
from app.ingestion.models import NormalizedDocument, RawDocument
from app.ingestion.pipeline import IngestionResult, run_ingestion
from app.ingestion.preprocessing.normalize import normalize_document

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
GOOD_ID = "dsid_" + "a" * 32
EMPTY_ID = "dsid_" + "b" * 32
FAIL_ID = "dsid_" + "c" * 32
JIRA_ID = "dsid_" + "2" * 32

CHUNK_FIELDS = {
    "chunk_id",
    "doc_id",
    "chunk_index",
    "content",
    "token_count",
    "start_char",
    "end_char",
    "content_hash",
    "source_type",
    "title",
    "metadata",
}


class WhitespaceTokenizer:
    """Small deterministic tokenizer that treats each word as one token."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, list[object]]:
        offsets = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        return {
            "input_ids": list(range(len(offsets))),
            "offset_mapping": offsets,
        }


class StaticLoader:
    """Deterministic in-memory loader for pipeline tests."""

    def __init__(
        self,
        documents: list[RawDocument],
        issues: list[LoaderIssue] | None = None,
    ) -> None:
        self.documents = documents
        self.issues = issues or []

    def iter_documents(self) -> Iterator[RawDocument]:
        yield from self.documents


def _chunker() -> TokenChunker:
    return TokenChunker(
        chunk_size=10,
        chunk_overlap=3,
        min_chunk_tokens=2,
        tokenizer=WhitespaceTokenizer(),
    )


def _config(output_dir: Path, **overrides: object) -> IngestionConfig:
    values: dict[str, object] = {
        "raw_dir": Path("unused"),
        "output_dir": output_dir,
        "chunk_size": 10,
        "chunk_overlap": 3,
        "min_chunk_tokens": 2,
        "source_types": ("confluence", "jira"),
    }
    values.update(overrides)
    return IngestionConfig(**values)


def _raw(
    doc_id: str,
    content: str,
    source_type: str = "confluence",
) -> RawDocument:
    return RawDocument(
        doc_id=doc_id,
        source_type=source_type,
        title="Title",
        content=content,
        source_path=f"{source_type}/{doc_id}.txt",
    )


def _run_fixture(tmp_path: Path) -> tuple[IngestionConfig, IngestionResult]:
    config = _config(tmp_path / "processed")
    loader = ZipDocumentLoader(FIXTURES / "sample_docs.zip")
    result = run_ingestion(config, loader=loader, chunker=_chunker())
    return config, result


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_pipeline_processes_fixture_zip_end_to_end(tmp_path: Path) -> None:
    config, result = _run_fixture(tmp_path)

    assert result.documents_loaded == 2
    assert result.documents_chunked == 2
    assert result.chunks_written == 2
    assert (config.output_dir / "chunks.jsonl").is_file()
    assert (config.output_dir / "stats.json").is_file()


def test_pipeline_writes_valid_jsonl(tmp_path: Path) -> None:
    config, result = _run_fixture(tmp_path)

    chunks = _jsonl(config.output_dir / "chunks.jsonl")

    assert len(chunks) == result.chunks_written
    for chunk in chunks:
        assert set(chunk) >= CHUNK_FIELDS
        assert str(chunk["doc_id"]).startswith("dsid_")
        assert str(chunk["chunk_id"]).startswith(f"{chunk['doc_id']}:")
        assert len(str(chunk["content_hash"])) == 64


def test_pipeline_counts_documents_and_chunks(tmp_path: Path) -> None:
    config, result = _run_fixture(tmp_path)

    stats = json.loads((config.output_dir / "stats.json").read_text(encoding="utf-8"))

    assert stats["documents_loaded"] == 2
    assert stats["documents_chunked"] == 2
    assert stats["chunks_written"] == 2
    assert sum(stats["chunks_by_source"].values()) == stats["chunks_written"]


def test_pipeline_is_idempotent_on_rerun(tmp_path: Path) -> None:
    config, first = _run_fixture(tmp_path)
    chunks_path = config.output_dir / "chunks.jsonl"
    stats_path = config.output_dir / "stats.json"
    first_chunks = chunks_path.read_bytes()
    first_stats = stats_path.read_bytes()

    second = run_ingestion(
        config,
        loader=ZipDocumentLoader(FIXTURES / "sample_docs.zip"),
        chunker=_chunker(),
    )

    assert first.model_dump() == second.model_dump()
    assert chunks_path.read_bytes() == first_chunks
    assert stats_path.read_bytes() == first_stats


def test_pipeline_records_skipped_and_failed_documents(tmp_path: Path) -> None:
    config = _config(tmp_path / "processed")
    good = _raw(GOOD_ID, "word0 word1 word2")
    empty = _raw(EMPTY_ID, "\x00 \r\n\t\n")
    failing = _raw(FAIL_ID, "boom")
    loader = StaticLoader(
        [good, empty, failing, good],
        issues=[LoaderIssue(source_path="bad.zip", reason="invalid member")],
    )

    def flaky_normalizer(raw: RawDocument) -> NormalizedDocument:
        if raw.doc_id == FAIL_ID:
            raise ValueError("unexpected content")
        return normalize_document(raw)

    result = run_ingestion(
        config,
        loader=loader,
        chunker=_chunker(),
        normalizer=flaky_normalizer,
    )

    assert result.documents_loaded == 4
    assert result.documents_chunked == 1
    assert result.documents_skipped == 1
    assert result.documents_empty == 1
    assert result.documents_duplicate == 1
    assert result.documents_failed == 1
    assert result.loader_issues == 1
    assert result.issues[0].reason == "invalid member"
    assert result.failures[0].stage == "normalize"
    assert result.documents_loaded == (
        result.documents_chunked
        + result.documents_skipped
        + result.documents_duplicate
        + result.documents_failed
    )


def test_pipeline_does_not_drop_documents_silently(tmp_path: Path) -> None:
    config = _config(tmp_path / "processed")
    loader = StaticLoader([_raw(GOOD_ID, "word0 word1 word2"), _raw(EMPTY_ID, "\x00\n")])

    result = run_ingestion(config, loader=loader, chunker=_chunker())
    stats = json.loads((config.output_dir / "stats.json").read_text(encoding="utf-8"))

    assert result.documents_loaded == 2
    assert result.documents_chunked == 1
    assert result.documents_skipped == 1
    assert stats["documents_loaded"] == 2
    assert stats["documents_chunked"] == 1
    assert stats["documents_skipped"] == 1


def test_pipeline_filters_sources_and_honors_max_docs(tmp_path: Path) -> None:
    loader = StaticLoader(
        [
            _raw("dsid_" + "d" * 32, "github words", source_type="github"),
            _raw(GOOD_ID, "confluence words"),
            _raw(JIRA_ID, "jira words", source_type="jira"),
        ]
    )
    config = _config(
        tmp_path / "processed",
        source_types=("confluence", "jira"),
        max_docs=1,
    )

    result = run_ingestion(config, loader=loader, chunker=_chunker())

    assert result.documents_filtered == 1
    assert result.documents_loaded == 1
    assert result.documents_chunked == 1
    assert result.documents_by_source == {"confluence": 1, "jira": 0}
    assert result.chunks_by_source == {"confluence": 1, "jira": 0}
