from app.pg301_payload_assembly import (
    TARGET_KEYS,
    assembly_target_for_context,
    audit_assembly_records,
    canonical_assembly_context,
    evaluate_assembly_rows,
    project_assembly_record,
    render_abstract_plan,
)


def _context(**overrides):
    values = {
        "typed_available": "1",
        "feedback_state": "observable_progress",
        "replay_ready": "1",
        "evidence_present": "1",
        "negative_control": "1",
        "fresh_reset": "1",
        "surface_method": "GET",
        "surface_field_role": "query_param",
        "surface_encoding": "url_percent",
        "history_action": "observe",
        "failure_class": "none",
        "step_budget": "present",
    }
    values.update(overrides)
    return [f"{key}={value}" for key, value in values.items()]


def test_missing_typed_observation_forces_question_and_no_send():
    target = assembly_target_for_context(_context(typed_available="unknown"))
    values = {token.split("=", 1)[0]: token.split("=", 1)[1] for token in target if "=" in token}
    assert values["question"] == "ask_typed_availability"
    assert values["next_action"] == "request_observation"
    assert values["safe_to_send"] == "0"
    assert values["canary"] == "none"


def test_ready_context_assembles_abstract_get_plan_without_literal_payload():
    target = assembly_target_for_context(_context())
    assert len(target) == len(TARGET_KEYS) + 2
    rendered = render_abstract_plan(target)
    assert "ABSTRACT PLAN" in rendered
    assert "<RUNTIME_CANARY>" in rendered
    assert "http" not in rendered.lower()
    assert "payload" not in rendered.lower()


def test_failed_history_repairs_but_does_not_send():
    target = assembly_target_for_context(_context(history_action="candidate_failed", failure_class="candidate_mismatch"))
    values = {token.split("=", 1)[0]: token.split("=", 1)[1] for token in target if "=" in token}
    assert values["next_action"] == "repair_abstract_plan"
    assert values["repair_action"] == "retry_bounded_variant"
    assert values["safe_to_send"] == "0"


def test_audit_and_metrics_keep_assembly_abstract_and_safe():
    ready = project_assembly_record({"record_id": "r1", "split": "train", "training_eligible": True, "context_tokens": _context()})
    missing = project_assembly_record({"record_id": "r2", "split": "implementation_holdout", "training_eligible": False, "context_tokens": _context(typed_available="unknown")})
    hard = project_assembly_record({"record_id": "r3", "split": "hard_negative_eval", "training_eligible": False, "hard_negative": True, "context_tokens": _context(history_action="candidate_failed", failure_class="candidate_mismatch")})
    audit = audit_assembly_records([ready, missing, hard])
    assert audit["status"] == "passed"
    metrics = evaluate_assembly_rows([ready, missing, hard], [ready["target_tokens"], missing["target_tokens"], hard["target_tokens"]])
    assert metrics["assembly_slot_exact"] == 1.0
    assert metrics["hard_negative_false_allow"] == 0
