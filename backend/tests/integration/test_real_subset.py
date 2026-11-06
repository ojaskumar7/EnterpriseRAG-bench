from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import IngestionConfig
from app.ingestion.loaders.zip_loader import ZipDocumentLoader
from app.ingestion.pipeline import run_ingestion

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REAL_SLICES = PROJECT_ROOT / "data" / "raw" / "slices"


def test_real_subset_pipeline_smoke(tmp_path: Path) -> None:
    zip_paths = sorted(REAL_SLICES.glob("*.zip"))
    if not zip_paths:
        pytest.skip("real dataset slices are not available")

    config = IngestionConfig(
        raw_dir=PROJECT_ROOT / "data" / "raw",
        output_dir=tmp_path / "processed",
        source_types=("confluence", "jira", "github", "hubspot"),
        max_docs=20,
    )

    result = run_ingestion(config, loader=ZipDocumentLoader(zip_paths[0]))

    assert result.documents_loaded > 0
    assert result.documents_chunked > 0
    assert result.chunks_written > 0

    chunks_path = config.output_dir / "chunks.jsonl"
    stats_path = config.output_dir / "stats.json"
    assert chunks_path.is_file()
    assert stats_path.is_file()

    lines = chunks_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == result.chunks_written
    for line in lines:
        chunk = json.loads(line)
        assert chunk["token_count"] <= 400
        assert chunk["chunk_id"].startswith(f"{chunk['doc_id']}:")
