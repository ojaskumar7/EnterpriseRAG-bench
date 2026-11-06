from __future__ import annotations

import zipfile
from pathlib import Path

from app.ingestion.loaders.zip_loader import ZipDocumentLoader

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
CONFLUENCE_ID = "dsid_" + "1" * 32
JIRA_ID = "dsid_" + "2" * 32


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def test_zip_loader_parses_committed_fixture() -> None:
    documents = list(ZipDocumentLoader(FIXTURES / "sample_docs.zip").iter_documents())
    by_id = {document.doc_id: document for document in documents}

    assert set(by_id) == {CONFLUENCE_ID, JIRA_ID}
    assert by_id[CONFLUENCE_ID].title == "Sample Confluence"
    assert by_id[CONFLUENCE_ID].content == "Confluence body."
    assert by_id[CONFLUENCE_ID].source_type == "confluence"
    assert by_id[CONFLUENCE_ID].source_path == f"confluence/{CONFLUENCE_ID}__sample.txt"


def test_zip_loader_derives_source_type_from_path(tmp_path: Path) -> None:
    zip_path = tmp_path / "jira.zip"
    _write_zip(
        zip_path,
        {f"jira/{JIRA_ID}__ticket.txt": b"Ticket title\n\nTicket body.\n"},
    )

    documents = list(ZipDocumentLoader(zip_path).iter_documents())

    assert len(documents) == 1
    assert documents[0].source_type == "jira"
    assert documents[0].title == "Ticket title"
    assert documents[0].content == "Ticket body."


def test_zip_loader_skips_malformed_filename(tmp_path: Path) -> None:
    zip_path = tmp_path / "mixed.zip"
    _write_zip(
        zip_path,
        {
            "confluence/not-a-doc.txt": b"Bad\n",
            f"confluence/{CONFLUENCE_ID}__good.txt": b"Good title\n\nGood body.\n",
        },
    )
    loader = ZipDocumentLoader(zip_path)

    documents = list(loader.iter_documents())

    assert [document.doc_id for document in documents] == [CONFLUENCE_ID]
    assert len(loader.issues) == 1
    assert loader.issues[0].source_path == "confluence/not-a-doc.txt"
    assert "malformed filename" in loader.issues[0].reason


def test_zip_loader_rejects_path_traversal(tmp_path: Path) -> None:
    zip_path = tmp_path / "traversal.zip"
    _write_zip(
        zip_path,
        {f"../confluence/{CONFLUENCE_ID}__escape.txt": b"Title\n\nBody.\n"},
    )
    loader = ZipDocumentLoader(zip_path)

    documents = list(loader.iter_documents())

    assert documents == []
    assert len(loader.issues) == 1
    assert "path traversal" in loader.issues[0].reason


def test_zip_loader_rejects_oversized_member(tmp_path: Path) -> None:
    zip_path = tmp_path / "oversized.zip"
    _write_zip(
        zip_path,
        {f"confluence/{CONFLUENCE_ID}__large.txt": b"Title\n\n0123456789\n"},
    )
    loader = ZipDocumentLoader(zip_path, max_uncompressed_bytes=5)

    documents = list(loader.iter_documents())

    assert documents == []
    assert len(loader.issues) == 1
    assert "exceeds" in loader.issues[0].reason
