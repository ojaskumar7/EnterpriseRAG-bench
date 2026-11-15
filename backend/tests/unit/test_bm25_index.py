import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import bm25s
import pytest
import Stemmer

from app.retrieval import bm25_index
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
    BM25CorpusIdentity,
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


def _identity(
    chunk_id: str,
    *,
    doc_id: str = "doc-a",
    source_type: str = "pdf",
    title: str = "Alpha",
    chunk_index: int = 0,
    content_hash: str = "hash-a",
) -> BM25CorpusIdentity:
    return BM25CorpusIdentity(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_type=source_type,
        title=title,
        chunk_index=chunk_index,
        content_hash=content_hash,
    )


def test_corpus_fingerprint_is_order_independent_and_metadata_sensitive() -> None:
    rows = [
        _identity("chunk-a"),
        _identity("chunk-b", doc_id="doc-b", title="Beta", content_hash="hash-b"),
    ]

    assert corpus_fingerprint(rows) == corpus_fingerprint(list(reversed(rows)))
    assert corpus_fingerprint(rows) != corpus_fingerprint(
        [_identity("chunk-a", title="Alpha changed"), rows[1]]
    )
    assert corpus_fingerprint(rows) != corpus_fingerprint(
        [_identity("chunk-a", source_type="html"), rows[1]]
    )
    assert corpus_fingerprint(rows) != corpus_fingerprint(
        [_identity("chunk-a", doc_id="doc-changed"), rows[1]]
    )
    assert corpus_fingerprint(rows) != corpus_fingerprint(
        [_identity("chunk-a", chunk_index=1), rows[1]]
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


def _tiny_version(
    root: Path,
    tenant: str,
    kb: str,
    version: str,
    *,
    index_k1: float = BM25_K1,
    index_b: float = BM25_B,
    index_method: str = BM25_METHOD,
    manifest_k1: float | None = None,
    manifest_b: float | None = None,
    manifest_method: str | None = None,
    extra_index_docs: int = 0,
) -> tuple[Path, BM25Manifest]:
    docs = [
        ("chunk-1", "doc-1", "Alpha needle", "Alpha", "pdf"),
        ("chunk-2", "doc-2", "Beta needle", "Beta", "html"),
    ]
    temp = root / tenant / kb / f".building-{version}"
    temp.mkdir(parents=True)
    stemmer = Stemmer.Stemmer("english")
    tokens = tokenize_bm25([f"{title}\n{content}" for _, _, content, title, _ in docs], stemmer)
    tokens.extend([["needle"] for _ in range(extra_index_docs)])
    index = bm25s.BM25(k1=index_k1, b=index_b, method=index_method)
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
            BM25CorpusIdentity(
                chunk_id=entry.chunk_id,
                doc_id=entry.doc_id,
                source_type=entry.source_type,
                title=entry.title,
                chunk_index=entry.chunk_index,
                content_hash=entry.content_hash,
            )
            for entry in entries
        ),
        catalog_sha256=sha256_file(temp / BM25_CATALOG_FILE),
        k1=index_k1 if manifest_k1 is None else manifest_k1,
        b=index_b if manifest_b is None else manifest_b,
        method=index_method if manifest_method is None else manifest_method,
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


def test_snapshot_search_drops_out_of_vocabulary_zero_scores(tmp_path: Path) -> None:
    manager = BM25IndexManager(tmp_path, retention=2)
    temp, manifest = _tiny_version(tmp_path, "tenant", "kb", "20261112T083000000000Z-111111111111")
    manager.publish(tenant_id="tenant", knowledge_base_id="kb", temp_dir=temp, manifest=manifest)

    snapshot = manager.load(tenant_id="tenant", knowledge_base_id="kb")
    tokens = snapshot.tokenize("zzzznotinvocabulary")

    assert snapshot.search(tokens, 2) == ([], [])


def test_snapshot_tokenization_is_safe_across_threads(tmp_path: Path) -> None:
    manager = BM25IndexManager(tmp_path, retention=2)
    temp, manifest = _tiny_version(tmp_path, "tenant", "kb", "20261112T083000000000Z-111111111111")
    manager.publish(tenant_id="tenant", knowledge_base_id="kb", temp_dir=temp, manifest=manifest)

    snapshot = manager.load(tenant_id="tenant", knowledge_base_id="kb")
    queries = [
        "The runners are running",
        "needle alpha",
        "Beta needle",
        "running runner runs",
    ]
    expected = [snapshot.tokenize(query) for query in queries]

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(snapshot.tokenize, queries * 50))

    assert results == expected * 50


def test_load_rejects_stored_parameter_mismatch(tmp_path: Path) -> None:
    manager = BM25IndexManager(tmp_path, retention=2)
    temp, manifest = _tiny_version(
        tmp_path,
        "tenant",
        "kb",
        "20261112T083000000000Z-111111111111",
        index_k1=2.5,
        manifest_k1=1.5,
    )
    manager.publish(tenant_id="tenant", knowledge_base_id="kb", temp_dir=temp, manifest=manifest)

    with pytest.raises(BM25IndexStale, match="k1"):
        manager.load(tenant_id="tenant", knowledge_base_id="kb")


def test_load_rejects_unsupported_stored_method(tmp_path: Path) -> None:
    manager = BM25IndexManager(tmp_path, retention=2)
    temp, manifest = _tiny_version(
        tmp_path,
        "tenant",
        "kb",
        "20261112T083000000000Z-111111111111",
        manifest_method="robertson",
    )
    manager.publish(tenant_id="tenant", knowledge_base_id="kb", temp_dir=temp, manifest=manifest)

    with pytest.raises(BM25IndexStale, match="method"):
        manager.load(tenant_id="tenant", knowledge_base_id="kb")


def test_load_rejects_index_row_count_mismatch(tmp_path: Path) -> None:
    manager = BM25IndexManager(tmp_path, retention=2)
    temp, manifest = _tiny_version(
        tmp_path,
        "tenant",
        "kb",
        "20261112T083000000000Z-111111111111",
        extra_index_docs=1,
    )
    manager.publish(tenant_id="tenant", knowledge_base_id="kb", temp_dir=temp, manifest=manifest)

    with pytest.raises(BM25IndexStale, match="rows"):
        manager.load(tenant_id="tenant", knowledge_base_id="kb")


def test_load_rejects_truncated_index_files(tmp_path: Path) -> None:
    manager = BM25IndexManager(tmp_path, retention=2)
    temp, manifest = _tiny_version(tmp_path, "tenant", "kb", "20261112T083000000000Z-111111111111")
    manager.publish(tenant_id="tenant", knowledge_base_id="kb", temp_dir=temp, manifest=manifest)
    version_dir = tmp_path / "tenant" / "kb" / "versions" / manifest.version
    (version_dir / BM25_INDEX_DIR / "data.csc.index.npy").write_bytes(b"truncated")

    with pytest.raises(BM25IndexUnavailable):
        manager.load(tenant_id="tenant", knowledge_base_id="kb")


def test_check_does_not_load_bm25_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = BM25IndexManager(tmp_path, retention=2)
    temp, manifest = _tiny_version(tmp_path, "tenant", "kb", "20261112T083000000000Z-111111111111")
    manager.publish(tenant_id="tenant", knowledge_base_id="kb", temp_dir=temp, manifest=manifest)

    class ExplodingBM25:
        @classmethod
        def load(cls, *args: object, **kwargs: object) -> None:
            raise AssertionError("check must not load bm25s arrays")

    monkeypatch.setattr(bm25_index.bm25s, "BM25", ExplodingBM25)

    checked = manager.check(
        tenant_id="tenant",
        knowledge_base_id="kb",
        expected_fingerprint=manifest.corpus_fingerprint,
    )

    assert checked == manifest


def test_check_detects_source_type_change(tmp_path: Path) -> None:
    manager = BM25IndexManager(tmp_path, retention=2)
    temp, manifest = _tiny_version(tmp_path, "tenant", "kb", "20261112T083000000000Z-111111111111")
    manager.publish(tenant_id="tenant", knowledge_base_id="kb", temp_dir=temp, manifest=manifest)
    changed = [
        BM25CorpusIdentity(
            chunk_id="chunk-1",
            doc_id="doc-1",
            source_type="html",
            title="Alpha",
            chunk_index=0,
            content_hash="hash-0",
        ),
        BM25CorpusIdentity(
            chunk_id="chunk-2",
            doc_id="doc-2",
            source_type="html",
            title="Beta",
            chunk_index=0,
            content_hash="hash-1",
        ),
    ]

    with pytest.raises(BM25IndexStale, match="stale"):
        manager.check(
            tenant_id="tenant",
            knowledge_base_id="kb",
            expected_fingerprint=corpus_fingerprint(changed),
        )


def test_publish_fsyncs_version_files_before_pointer_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = BM25IndexManager(tmp_path, retention=2)
    temp, manifest = _tiny_version(tmp_path, "tenant", "kb", "20261112T083000000000Z-111111111111")
    events: list[tuple[str, str]] = []
    original_fsync = bm25_index._fsync_file
    original_replace = bm25_index.os.replace

    def record_fsync(path: Path) -> None:
        events.append(("fsync", Path(path).name))
        original_fsync(path)

    def record_replace(source: str | Path, destination: str | Path) -> None:
        events.append(("replace", Path(destination).name))
        original_replace(source, destination)

    monkeypatch.setattr(bm25_index, "_fsync_file", record_fsync)
    monkeypatch.setattr(bm25_index.os, "replace", record_replace)

    manager.publish(tenant_id="tenant", knowledge_base_id="kb", temp_dir=temp, manifest=manifest)

    pointer_swap = events.index(("replace", "current.json"))
    before_swap = events[:pointer_swap]
    fsynced_files = {name for kind, name in before_swap if kind == "fsync"}
    assert {
        BM25_MANIFEST_FILE,
        BM25_CATALOG_FILE,
        "data.csc.index.npy",
        "indices.csc.index.npy",
        "indptr.csc.index.npy",
        "params.index.json",
        "vocab.index.json",
    } <= fsynced_files


def test_manager_publish_clears_cache_when_post_swap_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    manager.load(tenant_id="tenant", knowledge_base_id="kb")
    second_temp, second_manifest = _tiny_version(
        tmp_path, "tenant", "kb", "20261112T083001000000Z-222222222222"
    )

    def fail_cleanup(root: Path, *, active: str) -> None:
        del root, active
        raise RuntimeError("cleanup failed after pointer swap")

    monkeypatch.setattr(manager._store, "_cleanup", fail_cleanup)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        manager.publish(
            tenant_id="tenant",
            knowledge_base_id="kb",
            temp_dir=second_temp,
            manifest=second_manifest,
        )

    assert manager._cache == {}
    assert (
        manager.load(tenant_id="tenant", knowledge_base_id="kb").manifest.version
        == second_manifest.version
    )


def test_publish_survives_cleanup_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    second_temp, second_manifest = _tiny_version(
        tmp_path, "tenant", "kb", "20261112T083001000000Z-222222222222"
    )
    versions_dir = tmp_path / "tenant" / "kb" / "versions"
    original_iterdir = Path.iterdir

    def fail_versions_iterdir(path: Path):
        if path == versions_dir:
            raise OSError("cleanup unavailable")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", fail_versions_iterdir)

    result = manager.publish(
        tenant_id="tenant",
        knowledge_base_id="kb",
        temp_dir=second_temp,
        manifest=second_manifest,
    )

    assert result.version == second_manifest.version
    assert manager._cache == {}
