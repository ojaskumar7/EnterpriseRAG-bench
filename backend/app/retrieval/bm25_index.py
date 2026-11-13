"""BM25 index metadata, catalog, tokenization, and version storage."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import bm25s
import Stemmer

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
        if k <= 0:
            return [], []
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
        ranked = sorted(
            zip(row_indices, (float(score) for score in scores[0]), strict=True),
            key=lambda item: (-item[1], item[0]),
        )
        return [index for index, _ in ranked], [score for _, score in ranked]


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class BM25IndexStore:
    def __init__(self, index_root: Path, *, retention: int = 2) -> None:
        if retention < 1:
            raise ValueError("retention must be at least 1")
        self.index_root = index_root
        self.retention = retention

    def tenant_root(self, tenant_id: str, knowledge_base_id: str) -> Path:
        return self.index_root / tenant_id / knowledge_base_id

    def load(self, *, tenant_id: str, knowledge_base_id: str) -> BM25IndexSnapshot:
        root = self.tenant_root(tenant_id, knowledge_base_id)
        command = rebuild_command(tenant_id, knowledge_base_id)
        pointer_path = root / BM25_POINTER_FILE
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise BM25IndexUnavailable(
                f"BM25 index not found for tenant {tenant_id} and KB {knowledge_base_id}; "
                f"rebuild with: {command}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise BM25IndexStale(
                f"invalid BM25 current pointer for tenant {tenant_id} and KB "
                f"{knowledge_base_id}; rebuild with: {command}"
            ) from exc

        try:
            version_name = str(pointer["version"])
            pointer_hash = str(pointer["manifest_sha256"])
        except (KeyError, TypeError) as exc:
            raise BM25IndexStale(
                f"invalid BM25 current pointer for tenant {tenant_id} and KB "
                f"{knowledge_base_id}; rebuild with: {command}"
            ) from exc

        version_dir = root / "versions" / version_name
        manifest_path = version_dir / BM25_MANIFEST_FILE
        try:
            manifest_bytes = manifest_path.read_bytes()
        except OSError as exc:
            raise BM25IndexUnavailable(
                f"BM25 index version {version_name} is missing for tenant {tenant_id} "
                f"and KB {knowledge_base_id}; rebuild with: {command}"
            ) from exc
        if hashlib.sha256(manifest_bytes).hexdigest() != pointer_hash:
            raise BM25IndexStale(
                f"BM25 manifest hash mismatch for version {version_name}; rebuild with: {command}"
            )

        try:
            manifest = BM25Manifest.from_dict(json.loads(manifest_bytes))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise BM25IndexStale(
                f"invalid BM25 manifest for version {version_name}; rebuild with: {command}"
            ) from exc
        if manifest.version != version_name:
            raise BM25IndexStale(
                f"BM25 version mismatch for {version_name}; rebuild with: {command}"
            )
        if manifest.tenant_id != tenant_id or manifest.knowledge_base_id != knowledge_base_id:
            raise BM25IndexStale(
                f"BM25 manifest belongs to a different tenant/KB; rebuild with: {command}"
            )
        if manifest.schema_version != BM25_SCHEMA_VERSION:
            raise BM25IndexStale(
                f"BM25 schema {manifest.schema_version} is not supported; rebuild with: {command}"
            )

        catalog_path = version_dir / BM25_CATALOG_FILE
        try:
            catalog = tuple(read_catalog(catalog_path))
        except OSError as exc:
            raise BM25IndexUnavailable(
                f"BM25 catalog is missing for version {version_name}; rebuild with: {command}"
            ) from exc
        except BM25IndexStale as exc:
            raise BM25IndexStale(
                f"invalid BM25 catalog for version {version_name} for tenant {tenant_id} "
                f"and KB {knowledge_base_id}; rebuild with: {command}"
            ) from exc
        if sha256_file(catalog_path) != manifest.catalog_sha256:
            raise BM25IndexStale(
                f"BM25 catalog hash mismatch for version {version_name}; rebuild with: {command}"
            )
        if len(catalog) != manifest.row_count:
            raise BM25IndexStale(
                f"BM25 catalog has {len(catalog)} rows but manifest expects "
                f"{manifest.row_count}; rebuild with: {command}"
            )
        if [entry.row_index for entry in catalog] != list(range(len(catalog))):
            raise BM25IndexStale(
                f"BM25 catalog row indexes are invalid for version {version_name}; "
                f"rebuild with: {command}"
            )

        try:
            index = bm25s.BM25.load(
                version_dir / BM25_INDEX_DIR,
                load_corpus=False,
                show_progress=False,
            )
        except (OSError, ValueError) as exc:
            raise BM25IndexUnavailable(
                f"BM25 index files are unreadable for version {version_name}; "
                f"rebuild with: {command}"
            ) from exc

        return BM25IndexSnapshot(
            manifest=manifest,
            catalog=catalog,
            index=index,
            stemmer=Stemmer.Stemmer(BM25_STEMMER),
        )

    def check(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        expected_fingerprint: str,
    ) -> BM25Manifest:
        snapshot = self.load(tenant_id=tenant_id, knowledge_base_id=knowledge_base_id)
        if snapshot.manifest.corpus_fingerprint != expected_fingerprint:
            raise BM25IndexStale(
                f"BM25 index is stale for tenant {tenant_id} and KB {knowledge_base_id}; "
                f"rebuild with: {rebuild_command(tenant_id, knowledge_base_id)}"
            )
        return snapshot.manifest

    def publish(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        temp_dir: Path,
        manifest: BM25Manifest,
    ) -> BM25Manifest:
        root = self.tenant_root(tenant_id, knowledge_base_id)
        versions_dir = root / "versions"
        versions_dir.mkdir(parents=True, exist_ok=True)
        target = versions_dir / manifest.version
        if target.exists():
            raise BM25IndexError(
                f"BM25 version {manifest.version} already exists for tenant {tenant_id} "
                f"and KB {knowledge_base_id}"
            )

        temp_dir.replace(target)
        _fsync_dir(versions_dir)
        pointer = {
            "version": manifest.version,
            "manifest_sha256": sha256_file(target / BM25_MANIFEST_FILE),
        }
        pointer_tmp = root / f".current-{uuid.uuid4().hex}.tmp"
        pointer_tmp.write_text(
            json.dumps(pointer, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _fsync_file(pointer_tmp)
        os.replace(pointer_tmp, root / BM25_POINTER_FILE)
        _fsync_dir(root)
        self._cleanup(root, active=manifest.version)
        return manifest

    def _cleanup(self, root: Path, *, active: str) -> None:
        versions_dir = root / "versions"
        versions = sorted(
            path
            for path in versions_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )
        keep = {path.name for path in versions[-(self.retention + 1) :]}
        keep.add(active)
        for path in versions:
            if path.name not in keep:
                shutil.rmtree(path, ignore_errors=True)


class BM25IndexManager:
    def __init__(self, index_root: Path | str, *, retention: int = 2) -> None:
        self._store = BM25IndexStore(Path(index_root), retention=retention)
        self._cache: dict[tuple[str, str], BM25IndexSnapshot] = {}

    def load(self, *, tenant_id: str, knowledge_base_id: str) -> BM25IndexSnapshot:
        key = (tenant_id, knowledge_base_id)
        if key not in self._cache:
            self._cache[key] = self._store.load(
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
            )
        return self._cache[key]

    def reload(self, *, tenant_id: str, knowledge_base_id: str) -> BM25IndexSnapshot:
        self._cache.pop((tenant_id, knowledge_base_id), None)
        return self.load(tenant_id=tenant_id, knowledge_base_id=knowledge_base_id)

    def check(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        expected_fingerprint: str,
    ) -> BM25Manifest:
        return self._store.check(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            expected_fingerprint=expected_fingerprint,
        )

    def publish(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        temp_dir: Path,
        manifest: BM25Manifest,
    ) -> BM25Manifest:
        result = self._store.publish(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            temp_dir=temp_dir,
            manifest=manifest,
        )
        self._cache.clear()
        return result
