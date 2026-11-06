"""Build the development subset manifest from downloaded release slices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ingestion.manifest import (  # noqa: E402
    build_document_inventory,
    load_questions,
    select_questions,
    write_jsonl,
    write_manifest,
)

DEFAULT_SOURCES = ["confluence", "jira", "github", "hubspot"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/sample"))
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.8)
    parser.add_argument(
        "--fail-on-missing-gold-docs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit non-zero if a covered question lacks an expected document.",
    )
    args = parser.parse_args()

    raw_dir: Path = args.raw_dir
    questions_path = args.questions or (raw_dir / "questions.jsonl")

    inventory = build_document_inventory(raw_dir / "slices")
    questions = load_questions(questions_path)
    result = select_questions(
        questions=questions,
        inventory=inventory,
        selected_sources=args.sources,
        seed=args.seed,
        dev_ratio=args.dev_ratio,
    )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_dir / "dev_questions.jsonl", result.dev)
    write_jsonl(output_dir / "holdout_questions.jsonl", result.holdout)
    write_jsonl(output_dir / "abstention_questions.jsonl", result.abstention)
    write_jsonl(output_dir / "excluded_questions.jsonl", result.excluded)
    write_manifest(output_dir / "subset_manifest.json", result)

    inventory_summary = result.inventory
    print(
        f"Inventory: {inventory_summary.document_count} unique doc IDs "
        f"({inventory_summary.document_member_count} document members) "
        f"from {inventory_summary.zip_count} zips"
    )
    print(f"Source counts: {inventory_summary.source_counts}")
    print(
        f"Conflicting-source doc IDs: {inventory_summary.conflicting_document_count}; "
        f"duplicate members: {inventory_summary.duplicate_member_count}"
    )
    print(
        f"Questions: {result.total_questions} total, "
        f"{len(result.dev) + len(result.holdout)} eligible, "
        f"{len(result.dev)} dev, {len(result.holdout)} holdout, "
        f"{len(result.abstention)} abstention, {len(result.excluded)} excluded"
    )

    if inventory_summary.malformed_member_count:
        print(
            f"WARNING: {inventory_summary.malformed_member_count} malformed "
            "document members were skipped"
        )
    if inventory_summary.duplicate_member_count:
        print(
            f"WARNING: {inventory_summary.duplicate_member_count} duplicate "
            "document members share a doc_id; all source variants are retained"
        )

    missing_gold = [
        excluded
        for excluded in result.excluded
        if excluded.reason.startswith("missing expected documents")
    ]
    if missing_gold:
        print(
            f"ERROR: {len(missing_gold)} covered questions have missing gold documents",
            file=sys.stderr,
        )
        for excluded in missing_gold[:10]:
            print(
                f"  {excluded.question_id}: {excluded.reason}",
                file=sys.stderr,
            )
        if args.fail_on_missing_gold_docs:
            return 1

    print(f"Wrote manifest and splits to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
