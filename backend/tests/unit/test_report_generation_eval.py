from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from report_generation_eval import read_records  # noqa: E402


def _write(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "question_id": "qst_0001",
                "question_type": "factoid",
                "question": "q",
                "expected_doc_ids": ["doc-1"],
                "answer": {"answer": "a", "citation_ids": [1], "claims": [], "conflicts": []},
                "grounding": {"decision": "grounded", "reason": "r", "claim_checks": []},
                "context": {"citation_count": 1, "selected_doc_ids": ["doc-1"], "token_count": 100},
                "latency_ms": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_read_records_rejects_holdout_by_default(tmp_path: Path) -> None:
    path = _write(tmp_path / "generation_holdout.jsonl")
    try:
        read_records(path)
    except ValueError as exc:
        assert "not allowed" in str(exc)
    else:
        raise AssertionError("holdout input must be rejected by default")


def test_read_records_allows_holdout_when_explicit(tmp_path: Path) -> None:
    path = _write(tmp_path / "generation_holdout.jsonl")
    assert len(read_records(path, allow_holdout=True)) == 1


def test_read_records_accepts_dev_named_file(tmp_path: Path) -> None:
    path = _write(tmp_path / "generation_dev.jsonl")
    assert len(read_records(path)) == 1
