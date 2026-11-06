"""Download the approved EnterpriseRAG-Bench development subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

RELEASE_BASE = "https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/download/v1.0.0"
USER_AGENT = "enterprise-knowledge-intelligence/0.1"
CHUNK_SIZE = 1024 * 1024
RETRIES = 3


@dataclass(frozen=True)
class Asset:
    name: str
    kind: str

    @property
    def url(self) -> str:
        return f"{RELEASE_BASE}/{self.name}"


ASSETS = [
    Asset("confluence_slice_0001.zip", "slice"),
    Asset("confluence_slice_0002.zip", "slice"),
    Asset("jira_slice_0001.zip", "slice"),
    Asset("jira_slice_0002.zip", "slice"),
    Asset("github_slice_0001.zip", "slice"),
    Asset("github_slice_0002.zip", "slice"),
    Asset("hubspot_slice_0001.zip", "slice"),
    Asset("hubspot_slice_0002.zip", "slice"),
    Asset("hubspot_slice_0003.zip", "slice"),
    Asset("hubspot_slice_0004.zip", "slice"),
    Asset("questions.jsonl", "questions"),
]


def _destination(raw_dir: Path, asset: Asset) -> Path:
    if asset.kind == "slice":
        return raw_dir / "slices" / asset.name
    return raw_dir / asset.name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_zip(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"{path}: corrupt zip member {bad_member}")
        return sum(1 for member in archive.namelist() if not member.endswith("/"))


def _verify_questions(path: Path) -> int:
    count = 0
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc

            missing = {
                "question_id",
                "question_type",
                "source_types",
                "question",
                "expected_doc_ids",
                "gold_answer",
                "answer_facts",
            } - payload.keys()
            if missing:
                raise ValueError(f"{path}:{line_number}: missing fields {sorted(missing)}")

            question_id = payload["question_id"]
            if question_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate question_id {question_id}")
            seen_ids.add(question_id)
            count += 1

    if count != 500:
        raise ValueError(f"{path}: expected 500 questions, found {count}")

    return count


def _download(url: str, destination: Path, timeout: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_reported = 0

    with urllib.request.urlopen(request, timeout=timeout) as response:
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header else None
        downloaded = 0

        with temporary.open("wb") as handle:
            while chunk := response.read(CHUNK_SIZE):
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded - last_reported >= 10 * CHUNK_SIZE:
                    last_reported = downloaded
                    if total:
                        print(
                            f"  {downloaded / (1024 * 1024):.1f} / {total / (1024 * 1024):.1f} MiB"
                        )
                    else:
                        print(f"  {downloaded / (1024 * 1024):.1f} MiB")

    temporary.replace(destination)


def _download_with_retries(url: str, destination: Path, timeout: int) -> None:
    last_error: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            _download(url, destination, timeout=timeout)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            temporary = destination.with_name(destination.name + ".part")
            temporary.unlink(missing_ok=True)
            if attempt < RETRIES:
                wait = 2**attempt
                print(f"  attempt {attempt} failed: {exc}; retrying in {wait}s")
                time.sleep(wait)

    raise RuntimeError(f"failed to download {url}: {last_error}")


def _record_asset(raw_dir: Path, asset: Asset) -> dict[str, object]:
    destination = _destination(raw_dir, asset)
    record: dict[str, object] = {
        "name": asset.name,
        "kind": asset.kind,
        "url": asset.url,
        "path": str(destination.relative_to(raw_dir)),
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
    }
    if asset.kind == "slice":
        record["zip_member_count"] = _verify_zip(destination)
        record["verified"] = "zip"
    else:
        record["question_count"] = _verify_questions(destination)
        record["verified"] = "jsonl"
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    raw_dir: Path = args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    total_bytes = 0

    for asset in ASSETS:
        destination = _destination(raw_dir, asset)
        print(f"{asset.name} -> {destination}")

        if args.dry_run:
            continue

        if destination.exists() and not args.force:
            print("  already present; verifying")
        else:
            print(f"  downloading {asset.url}")
            _download_with_retries(asset.url, destination, timeout=args.timeout)

        record = _record_asset(raw_dir, asset)
        records.append(record)
        total_bytes += int(record["bytes"])

    if args.dry_run:
        return 0

    manifest = {
        "dataset": "onyx-dot-app/EnterpriseRAG-Bench",
        "release": "v1.0.0",
        "release_base_url": RELEASE_BASE,
        "assets": records,
        "asset_count": len(records),
        "total_bytes": total_bytes,
    }
    manifest_path = raw_dir / "download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"downloaded and verified {len(records)} assets, {total_bytes / (1024 * 1024):.1f} MiB")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
