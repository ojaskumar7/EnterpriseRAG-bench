import json
from dataclasses import replace
from pathlib import Path

import bm25s
import pytest
import Stemmer

from app.retrieval.bm25_index import (
    BM25_B,
    BM25_CATALOG_FILE,
    BM25_INDEX_DIR,
    BM25_K1,
    BM25_MANIFEST_FILE,
    BM25_METHOD,
    BM25_SCHEMA_VERSION,
    BM25_STEMMER,
    BM25_STOPWORDS,
    BM25CatalogEntry,
    BM25IndexManager,
    BM25IndexStale,
    BM25IndexUnavailable,
    BM25Manifest,
    corpus_fingerprint,
    read_catalog,
    sha256_file,
    tokenize_bm25,
    write_catalog,
    write_manifest,
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


def _tiny_version(root: Path, tenant: str, kb: str, version: str) -> tuple[Path, BM25Manifest]:
    docs = [
        ("chunk-1", "doc-1", "Alpha needle", "Alpha", "pdf"),
        ("chunk-2", "doc-2", "Beta needle", "Beta", "html"),
    ]
    temp = root / tenant / kb / f".building-{version}"
    temp.mkdir(parents=True)
    stemmer = Stemmer.Stemmer("english")
    tokens = tokenize_bm25([f"{title}\n{content}" for _, _, content, title, _ in docs], stemmer)
    index = bm25s.BM25(k1=BM25_K1, b=BM25_B, method=BM25_METHOD)
    index.index(tokens, show_progress=False)
    index.save(temp / BM25_INDEX_DIR, show_progress=False)
    entries = [
        BM25CatalogEntry(i, chunk_id, doc_id, source_type, title, 0, f"hash-{i}")
        for i, (chunk_id, doc_id, _, title, source_type) in enumerate(docs)
    ]
    write_catalog(temp / BM25_CATALOG_FILE, entries)
    manifest = BM25Manifest(
        schema_version=BM25_SCHEMA_VERSION,
        version=version,
        tenant_id=tenant,
        knowledge_base_id=kb,
        created_at="2026-11-12T08:30:00+00:00",
        document_count=2,
        row_count=2,
        corpus_fingerprint=corpus_fingerprint(
            (entry.chunk_id, entry.content_hash, entry.title) for entry in entries
        ),
        catalog_sha256=sha256_file(temp / BM25_CATALOG_FILE),
        k1=BM25_K1,
        b=BM25_B,
        method=BM25_METHOD,
        stopwords=BM25_STOPWORDS,
        stemmer=BM25_STEMMER,
        bm25s_version="0.3.16",
        pystemmer_version="3.1.0",
    )
    write_manifest(temp / BM25_MANIFEST_FILE, manifest)
    return temp, manifest


def test_publish_load_and_reload(tmp_path: Path) -> None:
    manager = BM25IndexManager(tmp_path, retention=2)
    temp, manifest = _tiny_version(tmp_path, "tenant", "kb", "20261112T083000000000Z-111111111111")

    manager.publish(tenant_id="tenant", knowledge_base_id="kb", temp_dir=temp, manifest=manifest)

    snapshot = manager.load(tenant_id="tenant", knowledge_base_id="kb")
    assert snapshot.manifest == manifest
    assert snapshot.row_count == 2
    assert snapshot.search(snapshot.tokenize("needle"), 2)[0] == [0, 1]

    pointer = json.loads((tmp_path / "tenant" / "kb" / "current.json").read_text())
    assert pointer["version"] == manifest.version


def test_retention_keeps_active_and_previous_versions(tmp_path: Path) -> None:
    manager = BM25IndexManager(tmp_path, retention=2)
    for index in range(4):
        version = f"20261112T08300{index}000000Z-{index:012d}"
        temp, manifest = _tiny_version(tmp_path, "tenant", "kb", version)
        manager.publish(
            tenant_id="tenant",
            knowledge_base_id="kb",
            temp_dir=temp,
            manifest=manifest,
        )

    versions = sorted(path.name for path in (tmp_path / "tenant" / "kb" / "versions").iterdir())
    assert len(versions) == 3


def test_check_detects_fingerprint_mismatch(tmp_path: Path) -> None:
    manager = BM25IndexManager(tmp_path, retention=2)
    temp, manifest = _tiny_version(tmp_path, "tenant", "kb", "20261112T083000000000Z-111111111111")
    manager.publish(tenant_id="tenant", knowledge_base_id="kb", temp_dir=temp, manifest=manifest)

    try:
        manager.check(
            tenant_id="tenant",
            knowledge_base_id="kb",
            expected_fingerprint="0" * 64,
        )
    except BM25IndexStale as exc:
        message = str(exc)
        assert "tenant" in message
        assert "kb" in message
        assert "scripts/build_bm25_index.py" in message
        assert "rebuild" in message.lower()
    else:
        raise AssertionError("expected BM25IndexStale")


def test_manager_caches_snapshots_and_publish_invalidates(tmp_path: Path) -> None:
    manager = BM25IndexManager(tmp_path, retention=2)
    first_temp, first_manifest = _tiny_version(
        tmp_path, "tenant", "kb", "20261112T083000000000Z-111111111111"
    )
    manager.publish(
        tenant_id="tenant",
        knowledge_base_id="kb",
        temp_dir=first_temp,
        manifest=first_manifest,
    )

    cached = manager.load(tenant_id="tenant", knowledge_base_id="kb")
    assert manager.load(tenant_id="tenant", knowledge_base_id="kb") is cached

    second_temp, second_manifest = _tiny_version(
        tmp_path, "tenant", "kb", "20261112T083001000000Z-222222222222"
    )
    manager.publish(
        tenant_id="tenant",
        knowledge_base_id="kb",
        temp_dir=second_temp,
        manifest=second_manifest,
    )

    refreshed = manager.load(tenant_id="tenant", knowledge_base_id="kb")
    assert refreshed is not cached
    assert refreshed.manifest.version == second_manifest.version


def test_failed_publish_leaves_current_pointer_and_snapshot_intact(tmp_path: Path) -> None:
    manager = BM25IndexManager(tmp_path, retention=2)
    temp, manifest = _tiny_version(tmp_path, "tenant", "kb", "20261112T083000000000Z-111111111111")
    manager.publish(tenant_id="tenant", knowledge_base_id="kb", temp_dir=temp, manifest=manifest)

    missing_build = tmp_path / "tenant" / "kb" / ".building-missing"
    failed_manifest = replace(manifest, version="20261112T090000000000Z-333333333333")
    with pytest.raises(OSError):
        manager.publish(
            tenant_id="tenant",
            knowledge_base_id="kb",
            temp_dir=missing_build,
            manifest=failed_manifest,
        )

    pointer = json.loads((tmp_path / "tenant" / "kb" / "current.json").read_text())
    assert pointer["version"] == manifest.version
    assert manager.load(tenant_id="tenant", knowledge_base_id="kb").manifest == manifest


def test_load_missing_pointer_reports_tenant_kb_and_rebuild(tmp_path: Path) -> None:
    manager = BM25IndexManager(tmp_path, retention=2)

    with pytest.raises(BM25IndexUnavailable) as excinfo:
        manager.load(tenant_id="tenant-x", knowledge_base_id="kb-y")

    message = str(excinfo.value)
    assert "tenant-x" in message
    assert "kb-y" in message
    assert "--tenant-id tenant-x" in message
    assert "--knowledge-base-id kb-y" in message
    assert "scripts/build_bm25_index.py" in message


def test_load_invalid_pointer_is_stale_with_rebuild(tmp_path: Path) -> None:
    root = tmp_path / "tenant-x" / "kb-y"
    root.mkdir(parents=True)
    (root / "current.json").write_text("{not-json", encoding="utf-8")
    manager = BM25IndexManager(tmp_path, retention=2)

    with pytest.raises(BM25IndexStale) as excinfo:
        manager.load(tenant_id="tenant-x", knowledge_base_id="kb-y")

    message = str(excinfo.value)
    assert "tenant-x" in message
    assert "kb-y" in message
    assert "scripts/build_bm25_index.py" in message


def test_load_detects_pointer_manifest_hash_mismatch(tmp_path: Path) -> None:
    manager = BM25IndexManager(tmp_path, retention=2)
    temp, manifest = _tiny_version(
        tmp_path, "tenant-x", "kb-y", "20261112T083000000000Z-111111111111"
    )
    manager.publish(
        tenant_id="tenant-x",
        knowledge_base_id="kb-y",
        temp_dir=temp,
        manifest=manifest,
    )
    pointer_path = tmp_path / "tenant-x" / "kb-y" / "current.json"
    pointer_path.write_text(
        json.dumps({"version": manifest.version, "manifest_sha256": "0" * 64}),
        encoding="utf-8",
    )

    with pytest.raises(BM25IndexStale) as excinfo:
        manager.load(tenant_id="tenant-x", knowledge_base_id="kb-y")

    message = str(excinfo.value)
    assert "tenant-x" in message
    assert "kb-y" in message
    assert "scripts/build_bm25_index.py" in message


def test_load_missing_catalog_is_unavailable_with_rebuild(tmp_path: Path) -> None:
    manager = BM25IndexManager(tmp_path, retention=2)
    temp, manifest = _tiny_version(
        tmp_path, "tenant-x", "kb-y", "20261112T083000000000Z-111111111111"
    )
    manager.publish(
        tenant_id="tenant-x",
        knowledge_base_id="kb-y",
        temp_dir=temp,
        manifest=manifest,
    )
    version_dir = tmp_path / "tenant-x" / "kb-y" / "versions" / manifest.version
    (version_dir / BM25_CATALOG_FILE).unlink()

    with pytest.raises(BM25IndexUnavailable) as excinfo:
        manager.load(tenant_id="tenant-x", knowledge_base_id="kb-y")

    message = str(excinfo.value)
    assert "tenant-x" in message
    assert "kb-y" in message
    assert "scripts/build_bm25_index.py" in message


def test_load_wraps_catalog_parse_errors_with_context(tmp_path: Path) -> None:
    manager = BM25IndexManager(tmp_path, retention=2)
    temp, manifest = _tiny_version(
        tmp_path, "tenant-x", "kb-y", "20261112T083000000000Z-111111111111"
    )
    manager.publish(
        tenant_id="tenant-x",
        knowledge_base_id="kb-y",
        temp_dir=temp,
        manifest=manifest,
    )
    version_dir = tmp_path / "tenant-x" / "kb-y" / "versions" / manifest.version
    (version_dir / BM25_CATALOG_FILE).write_text("{broken\n", encoding="utf-8")

    with pytest.raises(BM25IndexStale) as excinfo:
        manager.load(tenant_id="tenant-x", knowledge_base_id="kb-y")

    message = str(excinfo.value)
    assert "tenant-x" in message
    assert "kb-y" in message
    assert manifest.version in message
    assert "scripts/build_bm25_index.py" in message


def test_snapshot_search_returns_empty_for_non_positive_k(tmp_path: Path) -> None:
    manager = BM25IndexManager(tmp_path, retention=2)
    temp, manifest = _tiny_version(tmp_path, "tenant", "kb", "20261112T083000000000Z-111111111111")
    manager.publish(tenant_id="tenant", knowledge_base_id="kb", temp_dir=temp, manifest=manifest)

    snapshot = manager.load(tenant_id="tenant", knowledge_base_id="kb")
    tokens = snapshot.tokenize("needle")

    assert snapshot.search(tokens, 0) == ([], [])
    assert snapshot.search(tokens, -1) == ([], [])
