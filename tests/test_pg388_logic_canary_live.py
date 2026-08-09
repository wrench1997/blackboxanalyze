from __future__ import annotations

from scripts import run_pg388_logic_canary_live as live


def _fake_request(_base: str, path: str, *, method: str = "GET", payload=None, timeout: float = 5.0):
    if path == "/health":
        return {"status": "ok"}
    if path == "/api/manifest":
        return {"status": "dynamic_fixture_only_unbound", "implementation_id": "fake", "case_count": 56}
    if path == "/api/reset":
        return {"status": "fresh_reset", "state_clean": True}
    case_ref = payload["case_ref"]
    role = payload["role"]
    phase = payload["phase"]
    risk_cases = {
        "install_reentry_gate",
        "purchase_price_binding",
        "purchase_status_transition",
        "purchase_quantity_floor",
        "identity_canonicalization",
        "password_reset_subject_binding",
        "two_factor_reset_binding",
        "captcha_reuse",
        "session_fixation_boundary",
        "query_object_scope",
        "vertical_role_scope",
        "query_identifier_entropy",
        "execution_order",
        "sensitive_projection",
    }
    violation = (case_ref == "subject_resource_scope" and phase != "baseline" and role in {"candidate", "replay"}) or (case_ref in risk_cases and phase in {"candidate", "replay"})
    if case_ref in {"nonce_replay", "coupon_reuse_boundary"} and role == "replay":
        violation = True
    return {
        "status": "typed_local_canary_result",
        "case_ref": case_ref,
        "role": role,
        "phase": phase,
        "state_before": "zero_effect",
        "state_after": "one_or_more_effects",
        "state_delta": "abstract_delta",
        "effect_shape": "abstract_shape",
        "action_shape": "abstract_action",
        "invariant_holds": not violation,
        "vulnerable_effect": violation,
        "typed_observation": True,
        "negative_control_clean": role == "negative",
        "safe_to_send": False,
        "target_contacted": False,
        "external_network": False,
        "persistent_storage": False,
        "fresh_reset_required": True,
    }


def test_live_lane_is_blocked_without_explicit_local_flag() -> None:
    report = live.run(environ={})
    assert report["status"] == "planning_only_live_blocked"
    assert report["counts"]["typed_observations"] == 0
    assert report["execution"]["local_frontend_contacted"] is False


def test_live_lane_rejects_non_local_origin() -> None:
    report = live.run("https://example.invalid/pg388-api", environ={"PG388_LOCAL_EVAL": "1"})
    assert report["status"] == "blocked_preflight"
    assert report["reason"] == "local_origin_required"


def test_live_lane_keeps_only_abstract_typed_projection(monkeypatch) -> None:
    monkeypatch.setenv("PG388_LOCAL_EVAL", "1")
    report = live.run(environ={"PG388_LOCAL_EVAL": "1"}, request=_fake_request)
    assert report["status"] == "passed_live_local_canary_only"
    assert report["counts"] == {"fresh_resets": 17, "typed_observations": 85, "candidate_effects": 32, "negative_control_clean": 17, "unsafe_allow": 0}
    assert all("evaluator_sidecar" not in row for row in report["rows"])
    assert all(row["safe_to_send"] is False for row in report["rows"])
    assert report["model_boundary"]["raw_response_stored"] is False


def test_live_projection_rejects_untyped_response(monkeypatch) -> None:
    monkeypatch.setenv("PG388_LOCAL_EVAL", "1")

    def bad_request(*args, **kwargs):
        if args[1] == "/health":
            return {"status": "ok"}
        if args[1] == "/api/manifest":
            return {"status": "dynamic_fixture_only_unbound"}
        if args[1] == "/api/reset":
            return {"status": "fresh_reset", "state_clean": True}
        return {"status": "ask", "safe_to_send": False}

    report = live.run(environ={"PG388_LOCAL_EVAL": "1"}, request=bad_request)
    assert report["status"] == "completed_incomplete_live_canary"
    assert report["counts"]["typed_observations"] == 0
    assert report["execution"]["external_network"] is False
