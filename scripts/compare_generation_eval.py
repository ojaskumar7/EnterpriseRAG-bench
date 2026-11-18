#!/usr/bin/env python3
"""Compare two saved generation evaluations without making model calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(records: list[dict[str, object]]) -> dict[str, float | int]:
    if not records:
        raise ValueError("generation input is empty")
    decisions = {"grounded": 0, "abstain": 0, "conflict": 0}
    false_abstain = 0
    for record in records:
        decision = str(record.get("grounding", {}).get("decision", "legacy"))
        if decision in decisions:
            decisions[decision] += 1
        if bool(record.get("answer", {}).get("insufficient_evidence", False)) and record.get("expected_doc_ids"):
            false_abstain += 1
    count = len(records)
    return {
        "question_count": count,
        "grounded_rate": decisions["grounded"] / count,
        "abstain_rate": decisions["abstain"] / count,
        "conflict_rate": decisions["conflict"] / count,
        "false_abstain_rate": false_abstain / count,
        "grounded_count": decisions["grounded"],
        "abstain_count": decisions["abstain"],
        "conflict_count": decisions["conflict"],
        "false_abstain_count": false_abstain,
    }


def paired(bm25: list[dict[str, object]], candidate: list[dict[str, object]]) -> dict[str, int]:
    candidate_by_id = {str(row["question_id"]): row for row in candidate}
    transitions = {"abstain_to_abstain": 0, "abstain_to_non_abstain": 0, "non_abstain_to_abstain": 0, "non_abstain_to_non_abstain": 0}
    for base in bm25:
        other = candidate_by_id[str(base["question_id"])]
        b = bool(base.get("answer", {}).get("insufficient_evidence", False))
        c = bool(other.get("answer", {}).get("insufficient_evidence", False))
        key = ("abstain" if b else "non_abstain") + "_to_" + ("abstain" if c else "non_abstain")
        transitions[key] += 1
    return transitions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bm25", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = read(args.bm25)
    candidate = read(args.candidate)
    candidate_ids = {str(row["question_id"]) for row in candidate}
    matched_baseline = [row for row in baseline if str(row["question_id"]) in candidate_ids]
    result = {"partial": len(candidate) != len(baseline), "bm25": summarize(matched_baseline), "candidate": summarize(candidate)}
    result["paired_question_count"] = len(candidate)
    result["paired_abstention_outcomes"] = paired(matched_baseline, candidate)
    result["delta"] = {
        key: float(result["candidate"][key]) - float(result["bm25"][key])
        for key in ("grounded_rate", "abstain_rate", "conflict_rate", "false_abstain_rate")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
