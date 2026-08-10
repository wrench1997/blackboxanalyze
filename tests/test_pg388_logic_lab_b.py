from __future__ import annotations

import io
import json

from fixtures.pg388.logic_lab_b import IMPLEMENTATION_ID, application, manifest


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


def test_holdout_manifest_is_distinct_and_candidate_only() -> None:
    status, document = _call("/api/manifest")
    assert status == 200
    assert document["implementation_id"] == IMPLEMENTATION_ID
    assert document["implementation_independence"] == "separate_transition_table_and_module"
    assert document["network"]["external_network"] is False
    assert document["state"]["persistent_storage"] is False
    assert document["training_eligible"] is False


def test_holdout_transition_table_differs_from_a_and_keeps_negative_clean() -> None:
    _call("/api/reset", "POST")
    status, candidate = _call("/api/canary", "POST", {"case_ref": "purchase_concurrency_lock", "role": "candidate", "phase": "candidate"})
    assert status == 200
    assert candidate["implementation_id"] == IMPLEMENTATION_ID
    assert candidate["effect_shape"] == "version_guard_gap"
    assert candidate["transition_shape"] == "commit_order_shape"
    assert candidate["vulnerable_effect"] is True
    status, negative = _call("/api/canary", "POST", {"case_ref": "purchase_concurrency_lock", "role": "negative", "phase": "negative"})
    assert status == 200
    assert negative["vulnerable_effect"] is False
    assert negative["negative_control_clean"] is True
    assert negative["safe_to_send"] is False


def test_holdout_rejects_arbitrary_values_and_requires_phase_role_contract() -> None:
    status, rejected = _call("/api/canary", "POST", {"case_ref": "nonce_replay", "role": "candidate", "phase": "candidate", "value": "secret"})
    assert status == 400
    assert rejected["safe_to_send"] is False
    assert "secret" not in json.dumps(rejected, ensure_ascii=False)
    status, mismatch = _call("/api/canary", "POST", {"case_ref": "nonce_replay", "role": "reference", "phase": "candidate"})
    assert status == 400
    assert mismatch["ask_reason"] == "phase_role_mismatch"
