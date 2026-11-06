"""Loader for zipped UTF-8 document exports."""

from __future__ import annotations

import re
import zipfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

from app.ingestion.loaders.base import SOURCE_TYPES, LoaderIssue
from app.ingestion.models import RawDocument

MEMBER_PATTERN = re.compile(r"^(dsid_[0-9a-f]{32})(?:__.*)?\.txt$")


class ZipDocumentLoader:
    """Read text documents directly from a zip archive."""

    def __init__(
        self,
        zip_path: Path,
        max_uncompressed_bytes: int = 5 * 1024 * 1024,
    ) -> None:
        self.zip_path = Path(zip_path)
        self.max_uncompressed_bytes = max_uncompressed_bytes
        self.issues: list[LoaderIssue] = []

    def iter_documents(self) -> Iterator[RawDocument]:
        """Yield valid documents, recording rejected members in ``issues``."""

        self.issues.clear()
        with zipfile.ZipFile(self.zip_path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"{self.zip_path}: corrupt zip member {bad_member}")

            for info in archive.infolist():
                if info.is_dir():
                    continue
                document = self._load_member(archive, info)
                if document is not None:
                    yield document

    def _load_member(
        self,
        archive: zipfile.ZipFile,
        info: zipfile.ZipInfo,
    ) -> RawDocument | None:
        member_name = info.filename
        if not member_name.lower().endswith(".txt"):
            return None

        member_path = PurePosixPath(member_name)
        if member_path.is_absolute() or ".." in member_path.parts or "\\" in member_name:
            self._skip(member_name, "path traversal")
            return None

        if len(member_path.parts) < 2:
            self._skip(member_name, "malformed filename")
            return None

        match = MEMBER_PATTERN.match(member_path.name)
        if match is None:
            self._skip(member_name, "malformed filename")
            return None

        source_type = member_path.parts[0]
        if source_type not in SOURCE_TYPES:
            self._skip(member_name, f"unknown source type: {source_type}")
            return None

        if info.file_size > self.max_uncompressed_bytes:
            self._skip(member_name, "exceeds max uncompressed size")
            return None

        try:
            text = archive.read(info).decode("utf-8")
        except UnicodeDecodeError:
            self._skip(member_name, "invalid UTF-8")
            return None

        lines = text.splitlines()
        title_index = next(
            (index for index, line in enumerate(lines) if line.strip()),
            None,
        )
        if title_index is None:
            self._skip(member_name, "empty document")
            return None

        title = lines[title_index].strip()
        content = "\n".join(lines[title_index + 1 :]).strip()

        return RawDocument(
            doc_id=match.group(1),
            source_type=source_type,
            title=title,
            content=content,
            source_path=member_name,
            metadata={"zip_file": self.zip_path.name},
        )

    def _skip(self, source_path: str, reason: str) -> None:
        self.issues.append(LoaderIssue(source_path=source_path, reason=reason))
