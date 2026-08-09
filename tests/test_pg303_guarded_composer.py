from app.pg301_payload_assembly import target_map
from app.pg303_guarded_composer import compose_guarded_plan


def _context(**overrides):
    values = {
        "surface_method": "GET", "surface_field_role": "query_param", "surface_encoding": "url_percent",
        "typed_available": "1", "feedback_state": "observable_progress", "replay_ready": "1", "evidence_present": "1", "negative_control": "1", "fresh_reset": "1",
        "history_action": "observe", "failure_class": "none", "step_budget": "present",
    }
    values.update(overrides)
    return [f"{key}={value}" for key, value in values.items()]


def _proposal(**overrides):
    values = {
        "question": "none", "next_action": "assemble_abstract_plan", "repair_action": "none", "transport_ref": "surface_method", "field_role_ref": "surface_field_role", "encoding_ref": "surface_encoding", "canary": "runtime", "oracle": "typed", "stop_condition": "typed_effect_or_abstain", "safe_to_send": "1",
    }
    values.update(overrides)
    return ["[TARGET_BOS]", *[f"{key}={value}" for key, value in values.items()], "[TARGET_EOS]"]


def test_missing_observation_overrides_model_proposal_with_question():
    compiled = compose_guarded_plan(_proposal(), _context(typed_available="unknown"))
    values = target_map(compiled)
    assert values["question"] == "ask_typed_availability"
    assert values["next_action"] == "request_observation"
    assert values["safe_to_send"] == "0"


def test_guard_binds_surface_slots_and_keeps_safe_bit_fail_closed():
    compiled = compose_guarded_plan(_proposal(), _context(surface_method="POST", surface_field_role="form_field", surface_encoding="form_urlencoded"))
    values = target_map(compiled)
    assert values["transport"] == "POST"
    assert values["field_role"] == "form_field"
    assert values["encoding"] == "form_urlencoded"
    assert values["safe_to_send"] == "1"
    assert all("http" not in token.lower() for token in compiled)


def test_failed_history_cannot_be_promoted_by_model_safe_bit():
    compiled = compose_guarded_plan(_proposal(), _context(history_action="candidate_failed", failure_class="candidate_mismatch"))
    values = target_map(compiled)
    assert values["next_action"] == "assemble_abstract_plan" or values["next_action"] == "repair_abstract_plan"
    assert values["safe_to_send"] == "0"
