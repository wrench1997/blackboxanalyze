from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from app.pg331_source_row import collect_pg331_source_row
from tests.test_pg331_source_row import _evaluator, _field_capture_manifest, _observation, _reset, _source_meta, _target


_SPEC = importlib.util.spec_from_file_location("pg331_source_row_audit", Path(__file__).parents[1] / "scripts" / "audit_pg331_source_rows.py")
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
audit_source_rows = _MODULE.audit_source_rows
audit = _MODULE.audit


def _row(record_id: str, *, split: str, source: str = "fixture-page-01", implementation: str = "fixture-impl-a", family: str = "family-a") -> dict[str, object]:
    meta = _source_meta()
    meta.update({"source_id": source, "implementation": implementation, "family_id": family})
    return collect_pg331_source_row(record_id=record_id, observation=_observation(), source_meta=meta, reset=_reset(), evaluator=_evaluator(), field_capture_manifest=_field_capture_manifest(), target_projection=_target(), split=split, operator_reviewed=True)


def test_audit_passes_one_complete_training_row_without_promoting_it() -> None:
    report = audit_source_rows({"records": [_row("r1", split="train")]}, dataset_path="fixture")
    assert report["status"] == "passed"
    assert report["promotion"]["training_allowed"] is True
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["context_target_alignment"]["rate"] == 1.0


def test_audit_blocks_missing_axis_and_turns_it_into_ask_only() -> None:
    row = _row("r2", split="train")
    row["context_tokens"] = [token for token in row["context_tokens"] if not str(token).startswith("javascript_presence=")]
    row["training_eligible"] = False
    report = audit_source_rows({"records": [row]}, dataset_path="fixture")
    assert report["status"] == "blocked"
    assert "invalid_row:r2" in report["failures"]
    assert report["promotion"]["training_allowed"] is False


def test_audit_blocks_source_cross_split_instead_of_leaking_implementation() -> None:
    rows = [_row("r3", split="train"), _row("r4", split="dev")]
    report = audit_source_rows({"records": rows}, dataset_path="fixture")
    assert report["status"] == "blocked"
    assert "source_cross_split" in report["failures"]


def test_audit_maps_rule_ir_slots_to_prefixed_ontology_fields_without_leaking_variant() -> None:
    row = _row("r-slot", split="train")
    # The fixture tokenizer uses the same prefixed field names as live rows.
    row["context_tokens"] = [
        token
        for token in row["context_tokens"]
        if not str(token).startswith("request_transport_field_parameter_role=")
    ] + [
        "request_transport_field_parameter_role=two",
        "request_transport_field_encoding_chain=url_percent",
    ]
    report = audit_source_rows({"records": [row]}, dataset_path="fixture")
    assert report["context_target_alignment"]["rate"] == 1.0
    assert "context_target_alignment" not in report["failures"]


def test_audit_blocks_rule_ir_slot_when_its_context_field_is_missing() -> None:
    row = _row("r-slot-missing", split="train")
    row["context_tokens"] = [
        token
        for token in row["context_tokens"]
        if not str(token).startswith("request_transport_field_parameter_role=")
        and not str(token).startswith("parameter_role=")
    ]
    report = audit_source_rows({"records": [row]}, dataset_path="fixture")
    assert report["context_target_alignment"]["rate"] == 0.0
    assert "context_target_alignment" in report["failures"]


def test_cli_audit_accepts_relative_missing_dataset_without_path_exception() -> None:
    report = audit(Path("research/__pg331_relative_missing_fixture__.json"))
    assert report["status"] == "blocked"
    assert "missing:dataset" in report["failures"]


def test_cli_audit_accepts_external_staging_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "rows.json"
    dataset.write_text(json.dumps({"records": []}), encoding="utf-8")
    report = audit(dataset)
    assert report["status"] == "blocked"
    assert report["dataset"] == str(dataset.resolve())
