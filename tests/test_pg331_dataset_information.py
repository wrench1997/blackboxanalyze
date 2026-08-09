from __future__ import annotations

import json
from pathlib import Path

from app.pg331_source_row import sha256_json
from scripts.audit_pg331_dataset_information import (
    _required_inventory,
    audit_dataset_information,
    audit_document,
)
from tests.test_pg331_source_row_audit import _row


ROOT = Path(__file__).parents[1]
ONTOLOGY = json.loads((ROOT / "research" / "pg331_web_token_ontology_v1.json").read_text(encoding="utf-8"))


def _rows() -> list[dict[str, object]]:
    return [
        _row("info-a", split="train", source="source-a", implementation="implementation-a", family="family-a"),
        _row("info-b", split="dev", source="source-b", implementation="implementation-b", family="family-b"),
    ]


def _artifacts(rows: list[dict[str, object]]) -> tuple[dict[str, object], dict[str, object]]:
    context = {str(token) for row in rows for token in row["context_tokens"]}
    context.update(_required_inventory(ONTOLOGY))
    target = {str(token) for row in rows for token in row["target_tokens"]}
    vocabulary = {
        "schema_version": "pg331-web-token-vocabulary-v1",
        "status": "diagnostic_only_audit_blocked",
        "context_tokens": sorted(context),
        "target_tokens": sorted(target),
    }
    capacity = {
        "schema_version": "pg331-model-capacity-audit-v1",
        "status": "blocked",
        "required_context_window": 400,
        "variants": [{"config": {"id": "fixture"}, "context_window_pass": True, "capacity_pass": True, "truncation_risk": False}],
    }
    return vocabulary, capacity


def test_fresh_typed_rows_measure_every_axis_and_field_but_stay_diagnostic() -> None:
    rows = _rows()
    vocabulary, capacity = _artifacts(rows)
    report = audit_document(rows, ontology=ONTOLOGY, vocabulary=vocabulary, capacity=capacity, dataset_path="fixture.json")

    assert report["status"] == "diagnostic"
    assert len(report["axes"]) == 7
    assert sum(item["field_count"] for item in report["axes"].values()) == 107
    assert all(item["fields"] for item in report["axes"].values())
    assert all(field["entropy"]["status"] == "measured" for axis in report["axes"].values() for field in axis["fields"].values())
    assert all(field["ablation"]["status"] == "measured" for axis in report["axes"].values() for field in axis["fields"].values())
    assert report["vocabulary_coverage"]["status"] == "measured"
    assert report["capacity_coverage"]["status"] == "measured"
    assert report["validation"]["implementation_count"] == 2
    assert report["validation"]["accepted_training_eligible_count"] == 0
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_missing_typed_field_and_raw_context_are_incomplete_without_persisting_literal() -> None:
    rows = [_rows()[0]]
    row = rows[0]
    row["context_tokens"] = [token for token in row["context_tokens"] if not str(token).startswith("document_structure_field_doctype=")]
    row["context_tokens"].append("response_body=do-not-persist-secret")
    row["record_sha256"] = sha256_json({key: value for key, value in row.items() if key != "record_sha256"})
    vocabulary, capacity = _artifacts(rows)
    report = audit_document(rows, ontology=ONTOLOGY, vocabulary=vocabulary, capacity=capacity, dataset_path="fixture.json")
    serialized = json.dumps(report, ensure_ascii=False)

    assert report["status"] == "incomplete"
    assert "field_missing:document_structure.doctype" in report["failures"]
    assert "context_firewall" in report["failures"]
    assert report["context_firewall"]["forbidden_token_count"] == 1
    assert "do-not-persist-secret" not in serialized
    assert report["promotion"]["training_allowed"] is False


def test_missing_vocab_or_capacity_is_incomplete_and_operator_review_never_promotes() -> None:
    rows = _rows()
    report = audit_document(rows, ontology=ONTOLOGY, vocabulary=None, capacity=None, dataset_path="fixture.json")
    assert report["status"] == "incomplete"
    assert "missing:vocabulary" in report["failures"]
    assert "missing:capacity" in report["failures"]
    assert report["validation"]["operator_reviewed_count"] == 2
    assert report["training_eligibility"]["accepted_count"] == 0
    assert report["promotion"]["training_allowed"] is False


def test_missing_dataset_is_blocked_and_path_api_is_read_only() -> None:
    report = audit_dataset_information(Path("D:/definitely-missing-pg331-dataset.json"), ontology=ONTOLOGY, vocabulary=None, capacity=None)
    assert report["status"] == "blocked"
    assert "missing:dataset" in report["failures"]
    assert report["promotion"]["training_allowed"] is False


def test_single_implementation_is_diagnostic_failure_not_training_success() -> None:
    row = _rows()[0]
    vocabulary, capacity = _artifacts([row])
    report = audit_document([row], ontology=ONTOLOGY, vocabulary=vocabulary, capacity=capacity, dataset_path="fixture.json")
    assert report["status"] == "diagnostic"
    assert "single_implementation_diagnostic" in report["failures"]
    assert report["promotion"]["training_allowed"] is False


def test_capacity_report_is_checked_against_observed_row_window() -> None:
    rows = _rows()
    vocabulary, capacity = _artifacts(rows)
    capacity["required_context_window"] = 16
    report = audit_document(rows, ontology=ONTOLOGY, vocabulary=vocabulary, capacity=capacity, dataset_path="fixture.json")
    assert report["status"] == "incomplete"
    assert "capacity_dataset_window" in report["failures"]
    assert report["capacity_coverage"]["dataset_required_context_window"] > 16
    assert report["promotion"]["training_allowed"] is False
