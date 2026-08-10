import json
from pathlib import Path

from scripts.project_pg388_logic_frontend_summary import build_summary


def test_pg388_frontend_projection_is_bounded_and_fail_closed():
    summary = build_summary()
    assert summary["status"] == "diagnostic_rule_ir_candidate"
    assert summary["dataset"]["records"] == 840
    assert summary["dataset"]["split_counts"] == {"implementation_holdout": 420, "train": 420}
    assert summary["dataset"]["slot_order"][-1] == "safe_to_send"
    assert summary["audit"]["status"] == "passed_candidate_rule_ir_audit"
    assert summary["audit"]["context_firewall_passed"] is True
    assert summary["gates"]["capability_training_allowed"] is False
    assert summary["gates"]["training_eligible"] is False
    assert summary["live_evidence"]["row_bound"] is True
    assert summary["live_evidence"]["audit_status"] == "passed_candidate_logic_rule_ir_source_row_audit"
    assert summary["gates"]["fresh_role_reset_attested"] is True
    assert summary["gates"]["operator_reviewed"] is False
    assert summary["plan"]["optimizer_started"] is False


def test_pg388_frontend_projection_contains_no_rows_or_raw_markers():
    serialized = json.dumps(build_summary(), ensure_ascii=False).casefold()
    for marker in (
        '"rows"',
        "context_tokens",
        "target_tokens",
        "record_ref_sha256",
        "row_sha256",
        "payload=",
        "wire=",
        "response_body",
        "oracle_answer",
        "evaluator_answer",
        "http://",
        "https://",
    ):
        assert marker not in serialized
