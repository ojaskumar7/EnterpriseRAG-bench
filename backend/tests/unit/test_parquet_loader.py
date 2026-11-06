from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from app.ingestion.loaders.parquet_loader import ParquetDocumentLoader

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
CONFLUENCE_ID = "dsid_" + "1" * 32
JIRA_ID = "dsid_" + "2" * 32


def _row(
    doc_id: str,
    source_type: str = "jira",
    title: str = "Title",
    content: str = "Body.",
) -> dict[str, str]:
    return {
        "doc_id": doc_id,
        "source_type": source_type,
        "title": title,
        "content": content,
    }


def _write_parquet(
    path: Path, rows: list[dict[str, str]], columns: list[str] | None = None
) -> None:
    table = pa.Table.from_pylist(rows)
    if columns is not None:
        table = table.select(columns)
    pq.write_table(table, path)


def test_parquet_loader_parses_committed_fixture() -> None:
    documents = list(ParquetDocumentLoader(FIXTURES / "sample_docs.parquet").iter_documents())
    by_id = {document.doc_id: document for document in documents}

    assert set(by_id) == {CONFLUENCE_ID, JIRA_ID}
    assert by_id[CONFLUENCE_ID].title == "Sample Confluence"
    assert by_id[CONFLUENCE_ID].content == "Confluence body."
    assert by_id[CONFLUENCE_ID].source_type == "confluence"


def test_parquet_loader_reads_batches(tmp_path: Path) -> None:
    path = tmp_path / "docs.parquet"
    rows = [
        _row("dsid_" + f"{index:032x}", title=f"Title {index}", content=f"Body {index}.")
        for index in range(5)
    ]
    _write_parquet(path, rows)
    loader = ParquetDocumentLoader(path, batch_size=2)

    batches = list(loader.iter_batches())
    documents = [document for batch in batches for document in batch]

    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert [document.title for document in documents] == [f"Title {index}" for index in range(5)]


def test_parquet_loader_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "missing.parquet"
    _write_parquet(
        path,
        [_row("dsid_" + "a" * 32)],
        columns=["doc_id", "source_type", "title"],
    )
    loader = ParquetDocumentLoader(path)

    with pytest.raises(ValueError, match="missing required columns"):
        list(loader.iter_documents())


def test_parquet_loader_skips_invalid_doc_id(tmp_path: Path) -> None:
    path = tmp_path / "invalid.parquet"
    _write_parquet(
        path,
        [
            _row("not-a-doc-id", title="Bad", content="Bad body."),
            _row("dsid_" + "b" * 32, title="Good", content="Good body."),
        ],
    )
    loader = ParquetDocumentLoader(path)

    documents = list(loader.iter_documents())

    assert [document.title for document in documents] == ["Good"]
    assert len(loader.issues) == 1
    assert "invalid doc_id" in loader.issues[0].reason
