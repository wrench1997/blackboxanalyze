import json

from scripts.audit_pg388_logic_cross_implementation import audit


def test_cross_implementation_audit_is_candidate_only_and_fail_closed():
    result = audit()
    assert result["status"] == "passed_candidate_cross_implementation_logic_audit"
    assert result["sources"]["implementation_count"] == 2
    assert result["sources"]["split_counts"] == {"implementation_holdout": 280}
    assert result["coverage"]["source_row_count"] == 280
    assert result["coverage"]["strict_valid"] == 280
    assert result["coverage"]["typed_evidence"] == 280
    assert result["coverage"]["fresh_resets"] == 280
    assert result["coverage"]["negative_violations"] == 0
    assert result["hard_gate"]["train_split_present"] is False
    assert result["hard_gate"]["training_eligible"] == 0
    assert result["promotion"]["training_allowed"] is False


def test_cross_implementation_projection_has_no_rows_or_raw_payload_surface():
    serialized = json.dumps(audit(), ensure_ascii=False).casefold()
    for marker in (
        '"rows"',
        "context_tokens",
        "target_tokens",
        "raw_payload",
        "payload=",
        "wire",
        "response_body",
        "oracle_answer",
        "evaluator_answer",
        "http://",
        "https://",
    ):
        assert marker not in serialized
