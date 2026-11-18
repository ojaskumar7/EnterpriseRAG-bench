#!/usr/bin/env python3
"""Audit cached baseline false abstentions without making model calls."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("data/processed/generation_dev.jsonl"))
    p.add_argument("--output", type=Path, default=Path("data/processed/false_abstain_audit.json"))
    args = p.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    false = [r for r in rows if r.get("answer", {}).get("insufficient_evidence") and r.get("expected_doc_ids")]
    audited = []
    for row in false[:10]:
        expected = set(row["expected_doc_ids"])
        selected = set(row.get("context", {}).get("selected_doc_ids", []))
        present = bool(expected & selected)
        audited.append({
            "question_id": row["question_id"],
            "question": row["question"],
            "gold_doc_ids": sorted(expected),
            "selected_doc_ids": sorted(selected),
            "gold_doc_in_selected_context": present,
            "classification": "context-present-but-abstained" if present else "retrieval miss",
        })
    present_count = sum(bool(set(r["expected_doc_ids"]) & set(r.get("context", {}).get("selected_doc_ids", []))) for r in false)
    result = {
        "false_abstain_count": len(false),
        "gold_doc_in_selected_context_count": present_count,
        "retrieval_miss_count": len(false) - present_count,
        "context_present_but_abstained_count": present_count,
        "manual_sample": audited,
        "manual_sample_policy": "first 10 false-abstain records in stable JSONL order",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
