from pathlib import Path

from app.retrieval.bm25_builder import BM25BuildChunk, BM25Corpus, build_bm25_index
from app.retrieval.bm25_index import BM25IndexManager


def _corpus() -> BM25Corpus:
    return BM25Corpus(
        chunks=(
            BM25BuildChunk(
                chunk_id="chunk-1",
                doc_id="doc-1",
                source_type="pdf",
                title="Alpha",
                chunk_index=0,
                content="needle one",
                content_hash="hash-1",
            ),
            BM25BuildChunk(
                chunk_id="chunk-2",
                doc_id="doc-2",
                source_type="html",
                title="Beta",
                chunk_index=0,
                content="needle two",
                content_hash="hash-2",
            ),
        ),
        document_count=2,
    )


def test_build_publishes_loadable_snapshot(tmp_path: Path) -> None:
    manifest = build_bm25_index(
        corpus=_corpus(),
        index_root=tmp_path,
        tenant_id="tenant",
        knowledge_base_id="kb",
    )

    snapshot = BM25IndexManager(tmp_path).load(tenant_id="tenant", knowledge_base_id="kb")

    assert snapshot.manifest == manifest
    assert snapshot.manifest.document_count == 2
    assert snapshot.manifest.row_count == 2
    assert [entry.chunk_id for entry in snapshot.catalog] == ["chunk-1", "chunk-2"]
    assert snapshot.search(snapshot.tokenize("needle"), 2)[0] == [0, 1]


def test_empty_corpus_is_rejected(tmp_path: Path) -> None:
    try:
        build_bm25_index(
            corpus=BM25Corpus(chunks=(), document_count=0),
            index_root=tmp_path,
            tenant_id="tenant",
            knowledge_base_id="kb",
        )
    except ValueError as exc:
        assert "no chunks" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")
