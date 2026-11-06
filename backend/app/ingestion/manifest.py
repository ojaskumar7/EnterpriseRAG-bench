"""Corpus inventory and question selection for EnterpriseRAG-Bench."""

from __future__ import annotations

import hashlib
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field

KNOWN_SOURCE_TYPES = frozenset(
    {
        "slack",
        "gmail",
        "linear",
        "google_drive",
        "hubspot",
        "fireflies",
        "github",
        "jira",
        "confluence",
    }
)

DOC_ID_RE = re.compile(r"^dsid_[0-9a-f]{32}$")
MEMBER_RE = re.compile(r"^(dsid_[0-9a-f]{32})(?:__.*)?\.txt$", re.IGNORECASE)


class QuestionRecord(BaseModel):
    """A benchmark question and its evaluation metadata."""

    model_config = ConfigDict(extra="ignore")

    question_id: str
    question_type: str
    source_types: list[str] = Field(default_factory=list)
    question: str
    expected_doc_ids: list[str] = Field(default_factory=list)
    gold_answer: str = ""
    answer_facts: list[str] = Field(default_factory=list)


class ExcludedQuestion(BaseModel):
    """A question excluded from the development subset."""

    question_id: str
    question_type: str
    reason: str


class InventorySummary(BaseModel):
    """Summary of the downloaded document inventory."""

    document_count: int
    document_member_count: int
    source_counts: dict[str, int]
    conflicting_document_count: int
    fingerprint: str
    zip_count: int
    member_count: int
    malformed_member_count: int
    duplicate_member_count: int


class SelectionResult(BaseModel):
    """Result of selecting and splitting benchmark questions."""

    dev: list[QuestionRecord]
    holdout: list[QuestionRecord]
    abstention: list[QuestionRecord]
    excluded: list[ExcludedQuestion]
    inventory: InventorySummary
    selected_sources: list[str]
    seed: int
    dev_ratio: float
    total_questions: int


@dataclass(frozen=True)
class Inventory:
    """Document IDs and the source types where each ID appears."""

    documents: dict[str, frozenset[str]]
    source_counts: dict[str, int]
    zip_count: int
    member_count: int
    malformed_members: list[str]
    duplicate_members: list[str]

    def summary(self) -> InventorySummary:
        fingerprint = hashlib.sha256("\n".join(sorted(self.documents)).encode("utf-8")).hexdigest()
        conflicting_count = sum(1 for sources in self.documents.values() if len(sources) > 1)

        return InventorySummary(
            document_count=len(self.documents),
            document_member_count=sum(self.source_counts.values()),
            source_counts=dict(sorted(self.source_counts.items())),
            conflicting_document_count=conflicting_count,
            fingerprint=fingerprint,
            zip_count=self.zip_count,
            member_count=self.member_count,
            malformed_member_count=len(self.malformed_members),
            duplicate_member_count=len(self.duplicate_members),
        )


def load_questions(path: Path) -> list[QuestionRecord]:
    """Load and validate a JSONL question file."""

    path = Path(path)
    questions: list[QuestionRecord] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc

            record = QuestionRecord.model_validate(payload)
            if record.question_id in seen_ids:
                raise ValueError(
                    f"{path}:{line_number}: duplicate question_id {record.question_id}"
                )
            seen_ids.add(record.question_id)
            questions.append(record)

    if not questions:
        raise ValueError(f"{path}: no questions found")

    return questions


def parse_doc_id_from_member(member_name: str) -> tuple[str, str] | None:
    """Return (doc_id, source_type) from a slice member path, or None."""

    path = PurePosixPath(member_name)
    if path.is_absolute() or ".." in path.parts:
        return None
    if len(path.parts) < 2 or not path.name.lower().endswith(".txt"):
        return None

    match = MEMBER_RE.match(path.name)
    if match is None:
        return None

    source_type = path.parts[0]
    if source_type not in KNOWN_SOURCE_TYPES:
        return None

    return match.group(1), source_type


def build_document_inventory(slice_dir: Path) -> Inventory:
    """Build a document inventory from all zip slices in a directory."""

    slice_dir = Path(slice_dir)
    zip_paths = sorted(slice_dir.glob("*.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"no zip slices found in {slice_dir}")

    documents: dict[str, set[str]] = defaultdict(set)
    source_counts: Counter[str] = Counter()
    malformed_members: list[str] = []
    duplicate_members: list[str] = []
    member_count = 0

    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"{zip_path}: corrupt zip member {bad_member}")

            for member_name in archive.namelist():
                member_count += 1
                if member_name.endswith("/"):
                    continue

                parsed = parse_doc_id_from_member(member_name)
                if parsed is None:
                    if member_name.lower().endswith(".txt"):
                        malformed_members.append(f"{zip_path.name}:{member_name}")
                    continue

                doc_id, source_type = parsed
                if doc_id in documents:
                    duplicate_members.append(doc_id)
                documents[doc_id].add(source_type)
                source_counts[source_type] += 1

    return Inventory(
        documents={doc_id: frozenset(sources) for doc_id, sources in documents.items()},
        source_counts=dict(sorted(source_counts.items())),
        zip_count=len(zip_paths),
        member_count=member_count,
        malformed_members=malformed_members,
        duplicate_members=duplicate_members,
    )


def _split_eligible(
    eligible: list[QuestionRecord],
    seed: int,
    dev_ratio: float,
) -> tuple[list[QuestionRecord], list[QuestionRecord]]:
    if not 0 < dev_ratio < 1:
        raise ValueError("dev_ratio must be between 0 and 1")

    grouped: dict[str, list[QuestionRecord]] = defaultdict(list)
    for question in eligible:
        grouped[question.question_type].append(question)

    rng = random.Random(seed)
    dev: list[QuestionRecord] = []
    holdout: list[QuestionRecord] = []

    for question_type in sorted(grouped):
        group = list(grouped[question_type])
        rng.shuffle(group)
        count = len(group)
        dev_count = 1 if count == 1 else min(count - 1, max(1, round(count * dev_ratio)))
        dev.extend(group[:dev_count])
        holdout.extend(group[dev_count:])

    dev.sort(key=lambda question: question.question_id)
    holdout.sort(key=lambda question: question.question_id)
    return dev, holdout


def select_questions(
    questions: Iterable[QuestionRecord],
    inventory: Inventory,
    selected_sources: Iterable[str],
    seed: int = 42,
    dev_ratio: float = 0.8,
) -> SelectionResult:
    """Select questions covered by the subset and split them deterministically."""

    selected = frozenset(selected_sources)
    if not selected:
        raise ValueError("at least one source type must be selected")

    unknown_sources = selected - KNOWN_SOURCE_TYPES
    if unknown_sources:
        raise ValueError(f"unknown source types: {sorted(unknown_sources)}")

    question_list = sorted(questions, key=lambda question: question.question_id)
    abstention: list[QuestionRecord] = []
    excluded: list[ExcludedQuestion] = []
    eligible: list[QuestionRecord] = []

    for question in question_list:
        if not question.expected_doc_ids:
            abstention.append(question)
            continue

        unsupported = sorted(set(question.source_types) - selected)
        if unsupported:
            excluded.append(
                ExcludedQuestion(
                    question_id=question.question_id,
                    question_type=question.question_type,
                    reason=f"source types not in subset: {', '.join(unsupported)}",
                )
            )
            continue

        missing = sorted(set(question.expected_doc_ids) - inventory.documents.keys())
        if missing:
            excluded.append(
                ExcludedQuestion(
                    question_id=question.question_id,
                    question_type=question.question_type,
                    reason=f"missing expected documents: {', '.join(missing)}",
                )
            )
            continue

        listed_sources = set(question.source_types)
        mismatched = sorted(
            doc_id
            for doc_id in question.expected_doc_ids
            if not (inventory.documents[doc_id] & listed_sources)
        )
        if mismatched:
            excluded.append(
                ExcludedQuestion(
                    question_id=question.question_id,
                    question_type=question.question_type,
                    reason=(
                        f"expected documents not in listed source types: {', '.join(mismatched)}"
                    ),
                )
            )
            continue

        eligible.append(question)

    dev, holdout = _split_eligible(eligible, seed=seed, dev_ratio=dev_ratio)

    return SelectionResult(
        dev=dev,
        holdout=holdout,
        abstention=abstention,
        excluded=excluded,
        inventory=inventory.summary(),
        selected_sources=sorted(selected),
        seed=seed,
        dev_ratio=dev_ratio,
        total_questions=len(question_list),
    )


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> None:
    """Write Pydantic records as JSONL."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")


def write_manifest(path: Path, result: SelectionResult) -> None:
    """Write the human-readable subset manifest without question text."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "onyx-dot-app/EnterpriseRAG-Bench",
        "release": "v1.0.0",
        "selected_sources": result.selected_sources,
        "seed": result.seed,
        "dev_ratio": result.dev_ratio,
        "selection_rules": [
            "Questions with no expected documents are routed to the abstention set.",
            "Questions requiring a source outside the subset are excluded.",
            "Questions with any missing expected document are excluded.",
            "Eligible questions are split by question_type with a fixed seed.",
            "Evaluation metadata is never passed to retrieval.",
        ],
        "counts": {
            "total_questions": result.total_questions,
            "eligible": len(result.dev) + len(result.holdout),
            "dev": len(result.dev),
            "holdout": len(result.holdout),
            "abstention": len(result.abstention),
            "excluded": len(result.excluded),
        },
        "inventory": result.inventory.model_dump(),
        "dev_question_ids": [question.question_id for question in result.dev],
        "holdout_question_ids": [question.question_id for question in result.holdout],
        "abstention_question_ids": [question.question_id for question in result.abstention],
        "excluded": [question.model_dump() for question in result.excluded],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
