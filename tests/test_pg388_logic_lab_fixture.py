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
