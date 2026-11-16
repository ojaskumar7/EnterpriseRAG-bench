from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "rerank_search.py"


def load_main():
    spec = importlib.util.spec_from_file_location("rerank_search", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_cli_rejects_non_positive_top_k(capsys) -> None:
    main = load_main()

    assert main(["q", "--top-k", "0"]) == 1
    captured = capsys.readouterr()
    assert "top-k must be positive" in captured.err
