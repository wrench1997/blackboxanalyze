from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_pg379_dynamic_source_rows_live import audit


BASE = Path("research")


def test_pg379_matrix_v5_audit_is_candidate_only_and_strict() -> None:
    result = audit(
        report_path=BASE / "pg379_dynamic_source_rows_live_matrix_v5_20260810_report.json",
        rows_path=BASE / "pg379_dynamic_source_rows_live_matrix_v5_20260810_rows.json",
        sidecars_path=BASE / "pg379_dynamic_source_rows_live_matrix_v5_20260810_sidecars.json",
    )
    assert result["status"] == "passed_candidate_source_row_audit"
    assert result["counts"]["records"] == 72
    assert result["counts"]["strict_valid_records"] == 72
    assert result["counts"]["authorization_matches"] == 72
    assert result["counts"]["typed_roles"] == 72
    assert result["counts"]["failure_repair"] == 24
    assert result["counts"]["negative_violations"] == 0
    assert result["training_eligible_count"] == 0
    assert result["promotion"]["vulnerability_claim_allowed"] is False


def test_pg379_rows_have_no_raw_literal_markers() -> None:
    path = BASE / "pg379_dynamic_source_rows_live_matrix_v5_20260810_rows.json"
    text = path.read_text(encoding="utf-8").casefold()
    for marker in ("http://", "https://", "payload=", "wire=", "response_body_text=", "oracle_answer="):
        assert marker not in text
