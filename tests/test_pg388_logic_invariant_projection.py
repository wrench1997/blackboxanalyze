from __future__ import annotations

import json

from app.pg388_logic_invariant_projection import LOGIC_CASES, project_logic_case, project_logic_observation


def test_logic_inventory_covers_business_state_categories() -> None:
    assert len(LOGIC_CASES) >= 40
    surfaces = {case["surface"] for case in LOGIC_CASES}
    assert {"transaction_price", "password_reset", "captcha_state", "vertical_authorization", "session_rotation", "execution_order"} <= surfaces


def test_missing_logic_observation_asks_before_any_action() -> None:
    result = project_logic_observation({"case_ref": "quota_boundary", "role": "candidate"})
    assert result["status"] == "ask_missing_logic_observation"
    assert result["next_action"] == "ask"
    assert result["safe_to_send"] is False


def test_logic_failure_changes_one_abstract_variable_and_negative_abstains() -> None:
    candidate = project_logic_case("coupon_reuse_boundary", role="candidate", feedback_state="invariant_mismatch")
    negative = project_logic_case("coupon_reuse_boundary", role="negative", feedback_state="invariant_mismatch")
    assert candidate["target_projection"]["next_action"] == "repair"
    assert candidate["target_projection"]["repair_action"] == "replay"
    assert negative["target_projection"]["next_action"] == "abstain"
    assert negative["target_projection"]["safe_to_send"] is False


def test_logic_projection_has_no_raw_values_or_evaluator_answer() -> None:
    projection = project_logic_case("password_reset_subject_binding", role="candidate", feedback_state="state_mismatch")
    text = json.dumps({"context_tokens": projection["context_tokens"], "target_tokens": projection["target_tokens"], "logic_context": projection["logic_context"]}, ensure_ascii=False)
    for marker in ("http://", "https://", "payload", "wire", "response_body", "credential", "<script"):
        assert marker not in text.casefold()
    assert projection["raw_source_stored"] is False
    assert projection["oracle_answer_in_context"] is False
