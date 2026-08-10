from __future__ import annotations

import json

from scripts import audit_pg388_logic_supplement_canary_local as audit
from scripts import run_pg388_logic_supplement_canary_local as local


def test_supplemental_canary_audit_passes_without_returning_rows(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    local.write_report(report_path, local.run())
    result = audit.audit(report_path)
    assert result["status"] == "passed_candidate_only"
    assert result["counts"]["role_rows"] == 120
    assert result["counts"]["unique_evidence_hashes"] == 120
    assert result["counts"]["negative_violation"] == 0
    assert "rows" not in result
    assert result["training_eligible"] == 0


def test_supplemental_canary_audit_blocks_tampered_evidence(tmp_path) -> None:
    report = local.run()
    report["rows"][0]["effect_shape"] = "tampered_shape"
    report_path = tmp_path / "tampered.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    result = audit.audit(report_path)
    assert result["status"] == "blocked_supplement_canary_contract"
    assert any(item["code"] == "report_hash_mismatch" for item in result["failures"])
    assert any(item["code"] == "evidence_hash_mismatch" for item in result["failures"])


def test_supplemental_canary_audit_blocks_raw_marker(tmp_path) -> None:
    report = local.run()
    report["rows"][0]["effect_shape"] = "response_body=not-retained"
    report_path = tmp_path / "raw.json"
    report["report_sha256"] = local._sha({key: value for key, value in report.items() if key != "report_sha256"})
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    result = audit.audit(report_path)
    assert result["status"] == "blocked_supplement_canary_contract"
    assert any(item["code"] == "raw_marker_in_row" for item in result["failures"])

