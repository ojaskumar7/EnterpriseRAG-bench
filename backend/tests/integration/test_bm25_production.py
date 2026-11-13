import subprocess
import sys
import uuid

import pytest


@pytest.mark.integration
def test_build_cli_check_detects_missing_index(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_bm25_index.py",
            "--tenant-id",
            str(uuid.uuid4()),
            "--knowledge-base-id",
            str(uuid.uuid4()),
            "--index-root",
            str(tmp_path),
            "--check",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "rebuild" in result.stderr.lower()
