#!/usr/bin/env python3
"""Summarize saved grounded-generation results without making model calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.evaluation.generation import render_generation_markdown, summarize_generation  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/processed/generation_dev.jsonl",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "data/processed/generation_dev_metrics.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=PROJECT_ROOT / "data/processed/generation_dev_metrics.md",
    )
    parser.add_argument(
        "--allow-holdout",
        action="store_true",
        help="explicitly permit holdout inputs for the official benchmark report",
    )
    return parser


def read_records(path: Path, *, allow_holdout: bool = False) -> list[dict[str, object]]:
    if "holdout" in path.name.lower() and not allow_holdout:
        raise ValueError("holdout inputs are not allowed (pass --allow-holdout)")
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"record on line {line_number} must be an object")
            records.append(record)
    return records


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = summarize_generation(read_records(args.input, allow_holdout=args.allow_holdout))
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(
            render_generation_markdown(summary) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
