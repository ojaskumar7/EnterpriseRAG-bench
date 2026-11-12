from pathlib import Path

import Stemmer

from app.retrieval.bm25_index import (
    BM25CatalogEntry,
    BM25Manifest,
    corpus_fingerprint,
    read_catalog,
    tokenize_bm25,
    write_catalog,
)


def test_corpus_fingerprint_is_order_independent_and_title_sensitive() -> None:
    rows = [
        ("chunk-a", "hash-a", "Alpha"),
        ("chunk-b", "hash-b", "Beta"),
    ]

    assert corpus_fingerprint(rows) == corpus_fingerprint(list(reversed(rows)))
    assert corpus_fingerprint(rows) != corpus_fingerprint(
        [("chunk-a", "hash-a", "Alpha changed"), ("chunk-b", "hash-b", "Beta")]
    )


def test_manifest_round_trip() -> None:
    manifest = BM25Manifest(
        schema_version=1,
        version="20261112T083000000000Z-abcdef123456",
        tenant_id="tenant",
        knowledge_base_id="kb",
        created_at="2026-11-12T08:30:00+00:00",
        document_count=2,
        row_count=3,
        corpus_fingerprint="f" * 64,
        catalog_sha256="c" * 64,
        k1=1.5,
        b=0.75,
        method="lucene",
        stopwords="english",
        stemmer="english",
        bm25s_version="0.3.16",
        pystemmer_version="3.1.0",
    )

    assert BM25Manifest.from_dict(manifest.to_dict()) == manifest


def test_catalog_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "catalog.jsonl"
    entries = [
        BM25CatalogEntry(0, "chunk-a", "doc-a", "pdf", "Alpha", 0, "hash-a"),
        BM25CatalogEntry(1, "chunk-b", "doc-b", "html", "Beta", 2, "hash-b"),
    ]

    write_catalog(path, entries)

    assert read_catalog(path) == entries


def test_tokenize_bm25_stems_and_removes_stopwords() -> None:
    stemmer = Stemmer.Stemmer("english")

    tokens = tokenize_bm25(["The runners are running"], stemmer)[0]

    assert "the" not in tokens
    assert "are" not in tokens
    assert "run" in tokens
