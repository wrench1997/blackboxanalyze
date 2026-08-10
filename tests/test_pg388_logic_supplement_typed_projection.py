from __future__ import annotations

import json

from scripts import audit_pg388_logic_supplement_typed_projection as audit
from scripts import build_pg388_logic_supplement_typed_projection as builder
from scripts import run_pg388_logic_supplement_canary_local as local
from scripts.audit_pg388_logic_supplement_canary_local import audit as audit_source


def _write_source(tmp_path):
    report_path = tmp_path / "source.json"
    audit_path = tmp_path / "source_audit.json"
    local.write_report(report_path, local.run())
    audit_path.write_text(json.dumps(audit_source(report_path), ensure_ascii=False), encoding="utf-8")
    return report_path, audit_path


def test_typed_projection_is_abstract_and_has_no_train_split(tmp_path) -> None:
    report_path, audit_path = _write_source(tmp_path)
    artifact = builder.build(report_path, audit_path)
    assert artifact["status"] == "typed_rule_ir_diagnostic_candidate_only"
    assert artifact["counts"] == {
        "records": 120,
        "evaluator_diagnostic": 120,
        "train": 0,
        "implementation_holdout": 0,
        "cases": 10,
        "seeds": 3,
        "roles": 4,
        "typed_evaluator_observed": 120,
        "fresh_reset": 120,
        "role_bound_evidence": 120,
    }
    serialized = json.dumps(artifact, ensure_ascii=False).casefold()
    for marker in ("http://", "https://", "payload=", "wire=", "response_body=", "<script", "effect_shape"):
        assert marker not in serialized
    assert all(row["split"] == "evaluator_diagnostic" for row in artifact["rows"])
    assert all(row["training_eligible"] is False for row in artifact["rows"])


def test_typed_projection_audit_passes_without_returning_rows(tmp_path) -> None:
    report_path, audit_path = _write_source(tmp_path)
    artifact_path = tmp_path / "projection.json"
    artifact_path.write_text(json.dumps(builder.build(report_path, audit_path), ensure_ascii=False), encoding="utf-8")
    result = audit.audit(artifact_path)
    assert result["status"] == "passed_diagnostic_only"
    assert result["counts"]["records"] == 120
    assert result["counts"]["train"] == 0
    assert result["context_firewall_passed"] is True
    assert "rows" not in result


def test_typed_projection_audit_blocks_tampered_row(tmp_path) -> None:
    report_path, audit_path = _write_source(tmp_path)
    artifact = builder.build(report_path, audit_path)
    artifact["rows"][0]["target_tokens"].append("response_body=leak")
    artifact_path = tmp_path / "tampered.json"
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    result = audit.audit(artifact_path)
    assert result["status"] == "blocked_typed_projection_contract"
    assert "raw_or_evaluator_material_in_row" in result["failures"]
    assert "row_hash_mismatch" in result["failures"]

