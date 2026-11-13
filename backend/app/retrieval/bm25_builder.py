"""Build BM25 indexes from PostgreSQL chunks."""

from __future__ import annotations

import fcntl
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import bm25s
import Stemmer
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ChunkRecord, Document
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
    BM25Manifest,
    corpus_fingerprint,
    sha256_file,
    tokenize_bm25,
    write_catalog,
    write_manifest,
)


@dataclass(frozen=True)
class BM25BuildChunk:
    chunk_id: str
    doc_id: str
    source_type: str
    title: str
    chunk_index: int
    content: str
    content_hash: str


@dataclass(frozen=True)
class BM25Corpus:
    chunks: tuple[BM25BuildChunk, ...]
    document_count: int

    @property
    def row_count(self) -> int:
        return len(self.chunks)


async def load_bm25_corpus(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    knowledge_base_id: uuid.UUID,
) -> BM25Corpus:
    document_count = await session.scalar(
        select(func.count())
        .select_from(Document)
        .where(
            Document.tenant_id == tenant_id,
            Document.knowledge_base_id == knowledge_base_id,
        )
    )
    result = await session.execute(
        select(ChunkRecord)
        .where(
            ChunkRecord.tenant_id == tenant_id,
            ChunkRecord.knowledge_base_id == knowledge_base_id,
        )
        .order_by(ChunkRecord.chunk_id)
    )
    rows = result.scalars().all()
    chunks = tuple(
        BM25BuildChunk(
            chunk_id=row.chunk_id,
            doc_id=row.doc_id,
            source_type=row.source_type,
            title=row.title,
            chunk_index=row.chunk_index,
            content=row.content,
            content_hash=row.content_hash,
        )
        for row in rows
    )
    return BM25Corpus(chunks=chunks, document_count=int(document_count or 0))


@contextmanager
def build_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".build.lock"
    with lock_path.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another BM25 build is running for {root}") from exc
        yield


def build_bm25_index(
    *,
    corpus: BM25Corpus,
    index_root: Path,
    tenant_id: str,
    knowledge_base_id: str,
    retention: int = 2,
) -> BM25Manifest:
    chunks = tuple(sorted(corpus.chunks, key=lambda chunk: chunk.chunk_id))
    if not chunks:
        raise ValueError(f"no chunks found for tenant {tenant_id} and KB {knowledge_base_id}")

    fingerprint = corpus_fingerprint(
        (chunk.chunk_id, chunk.content_hash, chunk.title) for chunk in chunks
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    index_version = f"{timestamp}-{fingerprint[:12]}"
    root = Path(index_root) / tenant_id / knowledge_base_id
    manager = BM25IndexManager(index_root, retention=retention)

    with build_lock(root):
        temp_dir = root / f".building-{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True)
        try:
            stemmer = Stemmer.Stemmer(BM25_STEMMER)
            texts = [f"{chunk.title}\n{chunk.content}" for chunk in chunks]
            tokens = tokenize_bm25(texts, stemmer)
            index = bm25s.BM25(k1=BM25_K1, b=BM25_B, method=BM25_METHOD)
            index.index(tokens, show_progress=False)
            index.save(temp_dir / BM25_INDEX_DIR, show_progress=False)

            entries = [
                BM25CatalogEntry(
                    row_index=row_index,
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source_type=chunk.source_type,
                    title=chunk.title,
                    chunk_index=chunk.chunk_index,
                    content_hash=chunk.content_hash,
                )
                for row_index, chunk in enumerate(chunks)
            ]
            catalog_path = temp_dir / BM25_CATALOG_FILE
            write_catalog(catalog_path, entries)
            manifest = BM25Manifest(
                schema_version=BM25_SCHEMA_VERSION,
                version=index_version,
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
                created_at=datetime.now(UTC).isoformat(),
                document_count=corpus.document_count,
                row_count=len(chunks),
                corpus_fingerprint=fingerprint,
                catalog_sha256=sha256_file(catalog_path),
                k1=BM25_K1,
                b=BM25_B,
                method=BM25_METHOD,
                stopwords=BM25_STOPWORDS,
                stemmer=BM25_STEMMER,
                bm25s_version=version("bm25s"),
                pystemmer_version=version("PyStemmer"),
            )
            write_manifest(temp_dir / BM25_MANIFEST_FILE, manifest)

            loaded = bm25s.BM25.load(
                temp_dir / BM25_INDEX_DIR,
                load_corpus=False,
                show_progress=False,
            )
            probe_tokens = next((row for row in tokens if row), [])
            if len(entries) != len(chunks) or not probe_tokens:
                raise RuntimeError("built BM25 index failed validation")
            probe_indices, _ = loaded.retrieve(
                [probe_tokens],
                k=1,
                show_progress=False,
            )
            if len(probe_indices[0]) != 1:
                raise RuntimeError("built BM25 index failed validation")

            return manager.publish(
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
                temp_dir=temp_dir,
                manifest=manifest,
            )
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
