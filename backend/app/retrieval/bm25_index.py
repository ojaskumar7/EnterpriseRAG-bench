"""BM25 index metadata, catalog, tokenization, and version storage."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import bm25s

BM25_SCHEMA_VERSION = 1
BM25_K1 = 1.5
BM25_B = 0.75
BM25_METHOD = "lucene"
BM25_STOPWORDS = "english"
BM25_STEMMER = "english"
BM25_INDEX_DIR = "index"
BM25_MANIFEST_FILE = "manifest.json"
BM25_CATALOG_FILE = "catalog.jsonl"
BM25_POINTER_FILE = "current.json"


class BM25IndexError(RuntimeError):
    """Base error for BM25 index operations."""


class BM25IndexUnavailable(BM25IndexError):
    """The requested index is missing or cannot be loaded."""


class BM25IndexStale(BM25IndexError):
    """The index is inconsistent with its manifest or database."""


def rebuild_command(tenant_id: str, knowledge_base_id: str) -> str:
    return (
        "uv run python scripts/build_bm25_index.py "
        f"--tenant-id {tenant_id} --knowledge-base-id {knowledge_base_id}"
    )


@dataclass(frozen=True)
class BM25CatalogEntry:
    row_index: int
    chunk_id: str
    doc_id: str
    source_type: str
    title: str
    chunk_index: int
    content_hash: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> BM25CatalogEntry:
        return cls(
            row_index=int(data["row_index"]),
            chunk_id=str(data["chunk_id"]),
            doc_id=str(data["doc_id"]),
            source_type=str(data["source_type"]),
            title=str(data["title"]),
            chunk_index=int(data["chunk_index"]),
            content_hash=str(data["content_hash"]),
        )


@dataclass(frozen=True)
class BM25Manifest:
    schema_version: int
    version: str
    tenant_id: str
    knowledge_base_id: str
    created_at: str
    document_count: int
    row_count: int
    corpus_fingerprint: str
    catalog_sha256: str
    k1: float
    b: float
    method: str
    stopwords: str
    stemmer: str
    bm25s_version: str
    pystemmer_version: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> BM25Manifest:
        return cls(
            schema_version=int(data["schema_version"]),
            version=str(data["version"]),
            tenant_id=str(data["tenant_id"]),
            knowledge_base_id=str(data["knowledge_base_id"]),
            created_at=str(data["created_at"]),
            document_count=int(data["document_count"]),
            row_count=int(data["row_count"]),
            corpus_fingerprint=str(data["corpus_fingerprint"]),
            catalog_sha256=str(data["catalog_sha256"]),
            k1=float(data["k1"]),
            b=float(data["b"]),
            method=str(data["method"]),
            stopwords=str(data["stopwords"]),
            stemmer=str(data["stemmer"]),
            bm25s_version=str(data["bm25s_version"]),
            pystemmer_version=str(data["pystemmer_version"]),
        )


def corpus_fingerprint(rows: Iterable[tuple[str, str, str]]) -> str:
    digest = hashlib.sha256()
    for chunk_id, content_hash, title in sorted(rows):
        digest.update(chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_hash.encode("utf-8"))
        digest.update(b"\0")
        digest.update(title.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def tokenize_bm25(texts: Sequence[str], stemmer: Any) -> list[list[str]]:
    tokenized = bm25s.tokenize(
        list(texts),
        stopwords=BM25_STOPWORDS,
        stemmer=stemmer,
        return_ids=False,
        show_progress=False,
    )
    return [[str(token) for token in row if str(token).strip()] for row in tokenized]


def write_catalog(path: Path, entries: Iterable[BM25CatalogEntry]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")


def read_catalog(path: Path) -> list[BM25CatalogEntry]:
    entries: list[BM25CatalogEntry] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                entries.append(BM25CatalogEntry.from_dict(raw))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise BM25IndexStale(f"invalid catalog entry on line {line_number}") from exc
    return entries


def write_manifest(path: Path, manifest: BM25Manifest) -> None:
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class BM25IndexSnapshot:
    manifest: BM25Manifest
    catalog: tuple[BM25CatalogEntry, ...]
    index: bm25s.BM25
    stemmer: Any

    @property
    def row_count(self) -> int:
        return len(self.catalog)

    def tokenize(self, query: str) -> list[str]:
        tokenized = tokenize_bm25([query], self.stemmer)
        return tokenized[0] if tokenized else []

    def search(self, tokens: Sequence[str], k: int) -> tuple[list[int], list[float]]:
        if not tokens or not self.catalog:
            return [], []
        indices, scores = self.index.retrieve(
            [list(tokens)],
            k=min(k, self.row_count),
            show_progress=False,
        )
        row_indices = [int(index) for index in indices[0]]
        if any(index < 0 or index >= self.row_count for index in row_indices):
            raise BM25IndexStale(
                f"BM25 index contains an out-of-range row for tenant "
                f"{self.manifest.tenant_id} and KB {self.manifest.knowledge_base_id}; "
                f"rebuild with: "
                f"{rebuild_command(self.manifest.tenant_id, self.manifest.knowledge_base_id)}"
            )
        return row_indices, [float(score) for score in scores[0]]
