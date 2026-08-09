from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.pg284_evaluator_contract import evaluate_typed_replay, sha256_json


client = TestClient(app)


def _surface() -> dict:
    return {
        "surface_id": "pg284:surface:001",
        "method": "GET",
        "path": "/vul/test/read.php",
        "channel": "query",
        "field_count": 2,
        "authorization": "operator_allowlisted_remote_docker",
        "source_attestation_sha256": "a" * 64,
        "evaluator_kind": "dom_effect",
    }


def _projection(shape: str, *, status: str = "2xx", backend: bool = True, marker: str = "none") -> dict:
    return {"status_class": status, "shape_sha256": shape, "redirect_hops": 0, "backend_observed": backend, "effect_marker": marker}


def _reset() -> dict:
    return {"reset_id": "reset-001", "fresh_target": True, "container_recreated": True, "container_restart_used": False, "volume_mount_count": 0, "database_health_gate": "healthy", "state_change_allowed": False}


def _evidence() -> dict:
    value = {"effect_type": "dom_effect", "typed_effect_confirmed": True, "negative_control_clean": True, "reference_agreement": True, "replay_consistent": True, "non_destructive": True, "evaluator_id": "evaluator-001"}
    value["evidence_sha256"] = sha256_json(value)
    return value


def _kwargs() -> dict:
    return {
        "surface": _surface(),
        "reset": _reset(),
        "reference": _projection("b" * 64, marker="reference"),
        "negative": _projection("c" * 64, backend=True, marker="none"),
        "candidate": _projection("d" * 64, marker="candidate"),
        "replay": _projection("d" * 64, marker="candidate"),
        "typed_evidence": _evidence(),
    }


def test_remote_unavailable_blocks_typed_effect():
    result = evaluate_typed_replay(**_kwargs(), remote_probe={"status": "unavailable"})
    assert result["status"] == "blocked"
    assert result["typed_effect_confirmed"] is False
    assert result["vulnerability_claim_allowed"] is False
    assert "remote_docker_available" in result["reasons"]


def test_complete_contract_can_confirm_effect_but_not_vulnerability_claim():
    result = evaluate_typed_replay(**_kwargs(), remote_probe={"status": "available"})
    assert result["status"] == "confirmed_effect"
    assert result["typed_effect_confirmed"] is True
    assert result["confirmed_positive"] is False
    assert result["training_eligible"] is False


def test_hard_negative_forces_blocked():
    result = evaluate_typed_replay(**_kwargs(), remote_probe={"status": "available"}, hard_negative=True)
    assert result["status"] == "blocked"
    assert result["typed_effect_confirmed"] is False
    assert result["hard_negative"] is True


def test_raw_response_field_is_rejected():
    kwargs = _kwargs()
    kwargs["candidate"]["body_text"] = "raw"
    with pytest.raises(ValueError, match="raw"):
        evaluate_typed_replay(**kwargs, remote_probe={"status": "available"})


def test_api_evaluate_requires_confirmation(monkeypatch):
    called = False

    def fail_probe():
        nonlocal called
        called = True
        raise AssertionError("probe must not run without confirmation")

    monkeypatch.setattr("app.main.probe_authorized_remote_docker", fail_probe)
    response = client.post("/api/maze/remote-docker/evaluate", json={"operator_confirmed": False})
    assert response.status_code == 400
    assert called is False


def test_api_evaluate_surfaces_blocked_remote(monkeypatch):
    monkeypatch.setattr("app.main.probe_authorized_remote_docker", lambda: {"status": "unavailable"})
    payload = {"operator_confirmed": True, **_kwargs()}
    response = client.post("/api/maze/remote-docker/evaluate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "blocked"
    assert body["typed_effect_confirmed"] is False
