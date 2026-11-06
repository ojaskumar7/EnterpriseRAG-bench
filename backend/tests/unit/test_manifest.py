from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from app.ingestion.manifest import (
    ExcludedQuestion,
    Inventory,
    QuestionRecord,
    build_document_inventory,
    load_questions,
    parse_doc_id_from_member,
    select_questions,
    write_manifest,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SELECTED_SOURCES = {"confluence", "jira", "github", "hubspot"}

CONFLUENCE_1 = "dsid_11111111111111111111111111111111"
CONFLUENCE_2 = "dsid_77777777777777777777777777777777"
JIRA_1 = "dsid_22222222222222222222222222222222"
SLACK_1 = "dsid_33333333333333333333333333333333"
GITHUB_1 = "dsid_44444444444444444444444444444444"
HUBSPOT_1 = "dsid_66666666666666666666666666666666"


def _fixture_questions() -> list[QuestionRecord]:
    return load_questions(FIXTURES / "questions.jsonl")


def _fixture_inventory() -> Inventory:
    return Inventory(
        documents={
            CONFLUENCE_1: frozenset({"confluence"}),
            CONFLUENCE_2: frozenset({"confluence"}),
            JIRA_1: frozenset({"jira"}),
            SLACK_1: frozenset({"slack"}),
            GITHUB_1: frozenset({"github"}),
            HUBSPOT_1: frozenset({"hubspot"}),
        },
        source_counts={
            "confluence": 2,
            "jira": 1,
            "slack": 1,
            "github": 1,
            "hubspot": 1,
        },
        zip_count=1,
        member_count=6,
        malformed_members=[],
        duplicate_members=[],
    )


def _ids(questions: list[QuestionRecord]) -> set[str]:
    return {question.question_id for question in questions}


def test_load_questions_reads_fixture() -> None:
    questions = _fixture_questions()

    assert len(questions) == 10
    assert questions[0].question_id == "qst_0001"
    assert questions[0].source_types == ["confluence"]


def test_question_kept_when_all_sources_selected_and_gold_docs_present() -> None:
    result = select_questions(
        _fixture_questions(),
        _fixture_inventory(),
        SELECTED_SOURCES,
    )

    eligible = _ids(result.dev) | _ids(result.holdout)

    assert "qst_0001" in eligible
    assert "qst_0007" in eligible


def test_question_excluded_when_source_type_not_selected() -> None:
    result = select_questions(
        _fixture_questions(),
        _fixture_inventory(),
        SELECTED_SOURCES,
    )

    assert "qst_0003" not in (_ids(result.dev) | _ids(result.holdout))
    excluded = {item.question_id: item for item in result.excluded}
    assert excluded["qst_0003"].reason == "source types not in subset: slack"


def test_question_excluded_when_expected_doc_missing() -> None:
    result = select_questions(
        _fixture_questions(),
        _fixture_inventory(),
        SELECTED_SOURCES,
    )

    assert "qst_0004" not in (_ids(result.dev) | _ids(result.holdout))
    excluded = {item.question_id: item for item in result.excluded}
    assert excluded["qst_0004"].reason.startswith("missing expected documents:")


def test_zero_gold_question_routed_to_abstention_set() -> None:
    result = select_questions(
        _fixture_questions(),
        _fixture_inventory(),
        SELECTED_SOURCES,
    )

    assert _ids(result.abstention) == {"qst_0005", "qst_0006"}
    assert "qst_0005" not in (_ids(result.dev) | _ids(result.holdout))
    assert "qst_0006" not in (_ids(result.dev) | _ids(result.holdout))


def test_split_is_stratified_and_uses_fixed_seed() -> None:
    first = select_questions(
        _fixture_questions(),
        _fixture_inventory(),
        SELECTED_SOURCES,
        seed=7,
    )
    second = select_questions(
        _fixture_questions(),
        _fixture_inventory(),
        SELECTED_SOURCES,
        seed=7,
    )

    assert _ids(first.dev) == _ids(second.dev)
    assert _ids(first.holdout) == _ids(second.holdout)
    assert _ids(first.dev).isdisjoint(_ids(first.holdout))
    assert (_ids(first.dev) | _ids(first.holdout)) == {
        "qst_0001",
        "qst_0002",
        "qst_0007",
        "qst_0008",
        "qst_0009",
        "qst_0010",
    }

    for question_type in {"basic", "semantic"}:
        type_dev = [q for q in first.dev if q.question_type == question_type]
        type_holdout = [q for q in first.holdout if q.question_type == question_type]
        assert type_dev
        assert type_holdout


def test_manifest_never_contains_question_text_as_document(tmp_path: Path) -> None:
    result = select_questions(
        _fixture_questions(),
        _fixture_inventory(),
        SELECTED_SOURCES,
    )
    manifest_path = tmp_path / "subset_manifest.json"
    write_manifest(manifest_path, result)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_text = manifest_path.read_text(encoding="utf-8")

    assert "Question one?" not in manifest_text
    assert "documents" not in payload["inventory"]
    assert payload["inventory"]["document_count"] == 6
    assert payload["inventory"]["document_member_count"] == 6
    assert payload["inventory"]["conflicting_document_count"] == 0
    assert payload["counts"]["abstention"] == 2


def test_parse_doc_id_from_member() -> None:
    assert parse_doc_id_from_member(f"confluence/{CONFLUENCE_1}__title.txt") == (
        CONFLUENCE_1,
        "confluence",
    )
    assert parse_doc_id_from_member("confluence/bad-name.txt") is None
    assert parse_doc_id_from_member(f"../{CONFLUENCE_1}__title.txt") is None
    assert parse_doc_id_from_member(f"unknown/{CONFLUENCE_1}__title.txt") is None


def test_build_document_inventory_from_zip(tmp_path: Path) -> None:
    slice_path = tmp_path / "confluence_slice_0001.zip"
    with zipfile.ZipFile(slice_path, "w") as archive:
        archive.writestr(f"confluence/{CONFLUENCE_1}__one.txt", "Title\nBody")
        archive.writestr(f"jira/{JIRA_1}__two.txt", "Title\nBody")
        archive.writestr("confluence/not-a-doc.txt", "ignored")

    inventory = build_document_inventory(tmp_path)

    assert inventory.documents == {
        CONFLUENCE_1: frozenset({"confluence"}),
        JIRA_1: frozenset({"jira"}),
    }
    assert inventory.source_counts == {"confluence": 1, "jira": 1}
    assert inventory.zip_count == 1
    assert inventory.member_count == 3
    assert inventory.malformed_members == ["confluence_slice_0001.zip:confluence/not-a-doc.txt"]


def test_inventory_records_conflicting_duplicate_doc_id(tmp_path: Path) -> None:
    slice_path = tmp_path / "mixed_slice.zip"
    with zipfile.ZipFile(slice_path, "w") as archive:
        archive.writestr(f"confluence/{CONFLUENCE_1}__one.txt", "Title\nBody")
        archive.writestr(f"jira/{CONFLUENCE_1}__two.txt", "Title\nBody")

    inventory = build_document_inventory(tmp_path)
    summary = inventory.summary()

    assert inventory.documents[CONFLUENCE_1] == frozenset({"confluence", "jira"})
    assert inventory.duplicate_members == [CONFLUENCE_1]
    assert summary.document_count == 1
    assert summary.document_member_count == 2
    assert summary.conflicting_document_count == 1
    assert summary.source_counts == {"confluence": 1, "jira": 1}


def test_inventory_records_same_source_duplicate_doc_id(tmp_path: Path) -> None:
    slice_path = tmp_path / "jira_slice_0001.zip"
    with zipfile.ZipFile(slice_path, "w") as archive:
        archive.writestr(f"jira/{JIRA_1}__one.txt", "Title\nBody")
        archive.writestr(f"jira/{JIRA_1}__two.txt", "Title\nBody")

    inventory = build_document_inventory(tmp_path)
    summary = inventory.summary()

    assert inventory.documents[JIRA_1] == frozenset({"jira"})
    assert inventory.duplicate_members == [JIRA_1]
    assert summary.document_count == 1
    assert summary.document_member_count == 2
    assert summary.source_counts == {"jira": 2}
    assert summary.conflicting_document_count == 0


def test_question_excluded_when_gold_doc_source_does_not_match() -> None:
    question = QuestionRecord(
        question_id="qst_mismatch",
        question_type="basic",
        source_types=["jira"],
        question="Question?",
        expected_doc_ids=[CONFLUENCE_1],
        gold_answer="Answer.",
        answer_facts=[],
    )

    result = select_questions([question], _fixture_inventory(), SELECTED_SOURCES)
    excluded = {item.question_id: item for item in result.excluded}

    assert "qst_mismatch" not in (_ids(result.dev) | _ids(result.holdout))
    assert excluded["qst_mismatch"].reason.startswith(
        "expected documents not in listed source types:"
    )


def test_load_questions_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    payload = {
        "question_id": "qst_0001",
        "question_type": "basic",
        "source_types": ["confluence"],
        "question": "Question?",
        "expected_doc_ids": [CONFLUENCE_1],
        "gold_answer": "Answer.",
        "answer_facts": [],
    }
    line = json.dumps(payload)
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate question_id"):
        load_questions(path)


def test_excluded_question_model_serializes() -> None:
    excluded = ExcludedQuestion(
        question_id="qst_0099",
        question_type="basic",
        reason="source types not in subset: slack",
    )

    assert excluded.model_dump() == {
        "question_id": "qst_0099",
        "question_type": "basic",
        "reason": "source types not in subset: slack",
    }
