from __future__ import annotations

import io
import json

from fixtures.pg388.logic_lab import application, manifest


def _call(path: str, method: str = "GET", document: dict | None = None) -> tuple[int, dict]:
    body = json.dumps(document or {}, ensure_ascii=False).encode("utf-8") if method == "POST" else b""
    status: list[str] = []

    def start_response(value, headers):
        status.append(value)

    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "CONTENT_LENGTH": str(len(body)),
        "wsgi.input": io.BytesIO(body),
    }
    payload = b"".join(application(environ, start_response))
    return int(status[0].split()[0]), json.loads(payload.decode("utf-8"))


def test_dynamic_fixture_exposes_abstract_cases_and_reset() -> None:
    status, document = _call("/api/manifest")
    assert status == 200
    assert document["case_count"] >= 40
    assert document["network"]["external_network"] is False
    assert document["state"]["persistent_storage"] is False
    status, reset = _call("/api/reset", "POST")
    assert status == 200
    assert reset["state_clean"] is True


def test_episode_has_failure_repair_and_negative_boundary() -> None:
    status, result = _call("/api/episode", "POST", {"case_ref": "coupon_reuse_boundary", "role": "candidate", "feedback_state": "invariant_mismatch"})
    assert status == 200
    assert result["next_action"] == "repair"
    assert result["safe_to_send"] is False
    assert result["state_delta"] == "zero"
    status, negative = _call("/api/episode", "POST", {"case_ref": "coupon_reuse_boundary", "role": "negative", "feedback_state": "invariant_mismatch"})
    assert status == 200
    assert negative["negative_control_clean"] is True
    assert negative["typed_effect"] is False


def test_supplemental_taxonomy_endpoint_is_separate_and_candidate_only() -> None:
    status, document = _call("/api/supplemental-cases")
    assert status == 200
    assert document["status"] == "supplemental_candidate_only"
    assert document["training_eligible"] is False
    assert len(document["cases"]) == 10
    assert {item["case_ref"] for item in document["cases"]} >= {"oauth_second_factor", "captcha_predictability", "session_leakage"}


def test_fixture_rejects_non_enum_input_and_never_returns_raw_values() -> None:
    status, result = _call("/api/episode", "POST", {"case_ref": "coupon_reuse_boundary", "role": "candidate", "feedback_state": "typed_effect", "value": "secret"})
    assert status == 400
    assert result["safe_to_send"] is False
    assert "secret" not in json.dumps(result, ensure_ascii=False)
    assert manifest()["model_boundary"]["raw_values"] is False


def test_local_canary_replays_a_bounded_logic_state_machine() -> None:
    status, reset = _call("/api/reset", "POST")
    assert status == 200 and reset["state_clean"] is True
    status, baseline = _call("/api/canary", "POST", {"case_ref": "nonce_replay", "role": "candidate", "phase": "baseline"})
    assert status == 200 and baseline["invariant_holds"] is True
    status, first = _call("/api/canary", "POST", {"case_ref": "nonce_replay", "role": "candidate", "phase": "candidate"})
    assert status == 200 and first["state_delta"] == "one_effect"
    status, replay = _call("/api/canary", "POST", {"case_ref": "nonce_replay", "role": "replay", "phase": "replay"})
    assert status == 200
    assert replay["vulnerable_effect"] is True
    assert replay["state_delta"] == "duplicate_effect"
    assert replay["safe_to_send"] is False
    assert replay["external_network"] is False


def test_local_canary_negative_and_reference_stay_clean() -> None:
    _call("/api/reset", "POST")
    for case_ref in ("coupon_reuse_boundary", "subject_resource_scope"):
        status, reference = _call("/api/canary", "POST", {"case_ref": case_ref, "role": "reference", "phase": "reference"})
        assert status == 200
        assert reference["vulnerable_effect"] is False
        status, negative = _call("/api/canary", "POST", {"case_ref": case_ref, "role": "negative", "phase": "negative"})
        assert status == 200
        assert negative["negative_control_clean"] is True
        assert negative["state_delta"] == "zero"


def test_local_canary_accepts_only_scenario_role_phase_enums() -> None:
    status, result = _call("/api/canary", "POST", {"case_ref": "nonce_replay", "role": "candidate", "phase": "candidate", "value": "secret"})
    assert status == 400
    assert result["safe_to_send"] is False
    assert "secret" not in json.dumps(result, ensure_ascii=False)


def test_local_canary_exposes_extended_logic_matrix_without_raw_values() -> None:
    document = manifest()
    canary = document["canary"]
    assert len(canary["cases"]) == 17
    assert set(canary["concrete_effect_cases"]) >= {"install_reentry_gate", "two_factor_reset_binding", "query_object_scope"}
    _call("/api/reset", "POST")
    status, candidate = _call("/api/canary", "POST", {"case_ref": "install_reentry_gate", "role": "candidate", "phase": "candidate"})
    assert status == 200
    assert candidate["vulnerable_effect"] is True
    assert candidate["effect_shape"] == "setup_reentered"
    assert candidate["safe_to_send"] is False
    assert "identifier" not in json.dumps(candidate, ensure_ascii=False)
    status, negative = _call("/api/canary", "POST", {"case_ref": "two_factor_reset_binding", "role": "negative", "phase": "negative"})
    assert status == 200
    assert negative["vulnerable_effect"] is False
    assert negative["negative_control_clean"] is True
