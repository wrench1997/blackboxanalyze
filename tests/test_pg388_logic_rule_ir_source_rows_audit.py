import json
from pathlib import Path

from scripts.audit_pg388_logic_rule_ir_source_rows import audit


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "research" / "pg388_logic_rule_ir_source_rows_live_v1.json"
ROWS = ROOT / "research" / "pg388_logic_rule_ir_source_rows_live_rows_v1.json"
SIDECARS = ROOT / "research" / "pg388_logic_rule_ir_source_rows_live_sidecars_v1.json"


def test_pg388_logic_rule_ir_live_artifacts_pass_read_only_audit():
    result = audit(REPORT, ROWS, SIDECARS)
    assert result["status"] == "passed_candidate_logic_rule_ir_source_row_audit"
    assert result["counts"] == {"records": 140, "strict_valid": 140, "sidecars": 140, "unique_record_refs": 140, "raw_literal_hits": 0}
    assert result["training_eligible"] == 0
    assert result["promotion"]["training_allowed"] is False


def test_pg388_logic_rule_ir_rows_do_not_contain_raw_payload_or_wire():
    text = ROWS.read_text(encoding="utf-8").casefold() + SIDECARS.read_text(encoding="utf-8").casefold()
    for marker in ("payload=", "wire=", "response_body=", "oracle_answer=", "evaluator_answer="):
        assert marker not in text
