from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.pg282_evaluator_binding import bind_abstract_plan, sha256_json, validate_plan


client = TestClient(app)


def _plan(*, safe: bool = True, action: str | None = None) -> dict:
    return {
        "probe_class": "sql",
        "channel": "query",
        "encoding": "url_percent",
        "final_action": action or ("replay_confirmed" if safe else "abstain"),
        "safe_to_send": safe,
        "oracle_required": True,
    }


def _surface(*, contracts: bool = True) -> dict:
    return {
        "surface_id": "surface-001",
        "path": "/vul/sqli/read.php",
        "method": "GET",
        "channel": "query",
        "field_count": 2,
        "authorization": "operator_allowlisted_remote_docker",
        "typed_evaluator": "sql_ast_v1",
        "fresh_reset_contract": contracts,
        "reference_contract": contracts,
        "negative_contract": contracts,
    }


def _evidence() -> dict:
    evidence = {
        "typed_effect_confirmed": True,
        "negative_control_clean": True,
        "fresh_reset_attested": True,
        "reference_agreement": True,
        "replay_consistent": True,
        "non_destructive": True,
    }
    evidence["evidence_sha256"] = sha256_json(evidence)
    return evidence


def test_missing_literals_are_rejected():
    with pytest.raises(ValueError, match="literal payload"):
        validate_plan({**_plan(), "payload": "raw"})


def test_unavailable_remote_keeps_safe_plan_awaiting_evaluator():
    result = bind_abstract_plan(_plan(), _surface(), remote_probe={"status": "unavailable"})
    assert result["status"] == "await_evaluator"
    assert result["training_eligible"] is False
    assert "authorized_remote_docker_unavailable" in result["reasons"]
    assert result["wire_shape"]["literal_values_present"] is False


def test_abstain_plan_never_becomes_candidate():
    result = bind_abstract_plan(_plan(safe=False), _surface(), remote_probe={"status": "available"}, evaluator_evidence=_evidence())
    assert result["status"] == "abstain"
    assert result["decision"] == "do_not_send"


def test_all_typed_evidence_can_confirm_effect_but_not_vulnerability_claim():
    result = bind_abstract_plan(_plan(), _surface(), remote_probe={"status": "available"}, evaluator_evidence=_evidence())
    assert result["status"] == "confirmed_positive"
    assert result["checks"]["evidence_hash"] is True
    assert result["vulnerability_claim_allowed"] is False
    assert result["literal_payload_stored"] is False


def test_hard_negative_forces_abstain_even_with_complete_evidence():
    result = bind_abstract_plan(_plan(), _surface(), remote_probe={"status": "available"}, evaluator_evidence=_evidence(), hard_negative=True)
    assert result["status"] == "abstain"
    assert result["hard_negative"] is True
    assert result["training_eligible"] is False


def test_api_binding_requires_operator_confirmation(monkeypatch):
    called = False

    def fail_probe():
        nonlocal called
        called = True
        raise AssertionError("probe must not run without confirmation")

    monkeypatch.setattr("app.main.probe_authorized_remote_docker", fail_probe)
    response = client.post("/api/maze/remote-docker/bind", json={"operator_confirmed": False})
    assert response.status_code == 400
    assert called is False


def test_api_binding_exposes_blocked_abstract_result(monkeypatch):
    monkeypatch.setattr("app.main.probe_authorized_remote_docker", lambda: {"status": "unavailable"})
    response = client.post(
        "/api/maze/remote-docker/bind",
        json={"operator_confirmed": True, "plan": _plan(), "surface": _surface()},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "await_evaluator"
    assert body["training_eligible"] is False
    assert body["wire_shape"]["literal_values_present"] is False
