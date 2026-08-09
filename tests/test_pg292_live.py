from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.pg284_evaluator_contract import sha256_json
from app.pg292_live import evaluate_pg292_live


client = TestClient(app)


def _surface() -> dict:
    return {
        "surface_id": "pg292-live:surface:001",
        "method": "GET",
        "path": "/vul/test/read.php",
        "channel": "query",
        "field_count": 2,
        "authorization": "operator_allowlisted_remote_docker",
        "source_attestation_sha256": "a" * 64,
        "evaluator_kind": "dom_effect",
    }


def _projection(shape: str, *, marker: str = "none") -> dict:
    return {
        "status_class": "2xx",
        "shape_sha256": shape,
        "redirect_hops": 0,
        "backend_observed": True,
        "effect_marker": marker,
    }


def _reset() -> dict:
    return {
        "reset_id": "reset-292-001",
        "fresh_target": True,
        "container_recreated": True,
        "container_restart_used": False,
        "volume_mount_count": 0,
        "database_health_gate": "healthy",
        "state_change_allowed": False,
    }


def _evidence() -> dict:
    value = {
        "effect_type": "dom_effect",
        "typed_effect_confirmed": True,
        "negative_control_clean": True,
        "reference_agreement": True,
        "replay_consistent": True,
        "non_destructive": True,
        "evaluator_id": "typed-evaluator-292",
    }
    value["evidence_sha256"] = sha256_json(value)
    return value


def _kwargs() -> dict:
    return {
        "context_tokens": [
            "[BOS]",
            "phase=post_observation",
            "method=GET",
            "placement=query",
            "typed_available=1",
            "feedback=resolved",
            "candidate_sent=1",
            "replay_consistent=1",
            "ir_layer=shared_slot_ontology",
            "ir_family_agnostic=1",
            "[CTX_END]",
        ],
        "plan_tokens": [
            "[TARGET_BOS]",
            "plan=replay_confirmed",
            "method=GET",
            "probe_class=sql",
            "channel=query",
            "encoding=url_percent",
            "wire=query_param",
            "field_slot=observed_or_runtime_canary",
            "repair_delta=none",
            "family_agnostic=1",
            "final_action=replay_confirmed",
            "safe_to_send=1",
            "[TARGET_EOS]",
        ],
        "gate_probability": 0.91,
        "gate_threshold": 0.8,
        "surface": _surface(),
        "reset": _reset(),
        "reference": _projection("b" * 64, marker="reference"),
        "negative": _projection("c" * 64),
        "candidate": _projection("d" * 64, marker="candidate"),
        "replay": _projection("d" * 64, marker="candidate"),
        "typed_evidence": _evidence(),
    }


def test_unavailable_remote_blocks_even_with_a_valid_model_plan():
    result = evaluate_pg292_live(**_kwargs(), remote_probe={"status": "unavailable"})
    assert result["status"] == "blocked"
    assert result["decision"] == "abstain"
    assert result["checks"]["feature_gate"] is True
    assert result["checks"]["rule_ir_renderable"] is True
    assert result["wire_emission_allowed"] is False
    assert result["training_eligible"] is False
    assert "authorized_remote_docker_unavailable" in result["reasons"]


def test_complete_replay_is_only_a_candidate_for_explicit_training_review():
    result = evaluate_pg292_live(
        **_kwargs(),
        remote_probe={"status": "available"},
        operator_reviewed=True,
        independent_audit_pass=True,
        cross_seed_reviewed=True,
    )
    assert result["status"] == "typed_replay_candidate_for_training_review"
    assert result["decision"] == "hold_for_explicit_promotion"
    assert result["typed_replay"]["status"] == "confirmed_effect"
    assert result["training_eligible"] is True
    assert result["confirmed_positive"] is False
    assert result["vulnerability_claim_allowed"] is False
    assert result["wire_emission_allowed"] is False
    assert len(result["evidence_sha256"]) == 64


def test_hard_negative_never_becomes_training_candidate():
    result = evaluate_pg292_live(
        **_kwargs(),
        remote_probe={"status": "available"},
        hard_negative=True,
        operator_reviewed=True,
        independent_audit_pass=True,
        cross_seed_reviewed=True,
    )
    assert result["status"] == "blocked"
    assert result["training_eligible"] is False
    assert "family_or_route_hard_negative" in result["reasons"]


def test_context_and_plan_literals_are_rejected():
    kwargs = _kwargs()
    kwargs["context_tokens"] = ["ir_layer=shared_slot_ontology", "ir_family_agnostic=1", "payload=<script>"]
    with pytest.raises(ValueError, match="literal probe"):
        evaluate_pg292_live(**kwargs, remote_probe={"status": "unavailable"})

    kwargs = _kwargs()
    kwargs["plan_tokens"] = list(kwargs["plan_tokens"])
    kwargs["plan_tokens"][3] = "probe_class=javascript:alert"
    result = evaluate_pg292_live(**kwargs, remote_probe={"status": "unavailable"})
    assert result["status"] == "blocked"
    assert result["rule_ir"]["literal_probe_token"] is True
    assert result["wire_emission_allowed"] is False


def test_context_requires_shared_family_agnostic_slot():
    kwargs = _kwargs()
    kwargs["context_tokens"] = ["phase=post_observation"]
    with pytest.raises(ValueError, match="family-agnostic"):
        evaluate_pg292_live(**kwargs, remote_probe={"status": "unavailable"})


def test_api_pg292_live_is_fail_closed_when_remote_docker_is_unavailable(monkeypatch):
    monkeypatch.setattr("app.main.probe_authorized_remote_docker", lambda: {"status": "unavailable", "evidence_sha256": "e" * 64})
    payload = _kwargs()
    payload.update({"operator_confirmed": True, "authorization": "remote_docker"})
    response = client.post("/api/maze/remote-docker/pg292-live", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "blocked"
    assert body["remote_probe"]["status"] == "unavailable"
    assert body["wire_emission_allowed"] is False


def test_api_pg292_live_does_not_probe_without_operator_confirmation(monkeypatch):
    called = False

    def fail_probe():
        nonlocal called
        called = True
        raise AssertionError("probe must not run without confirmation")

    monkeypatch.setattr("app.main.probe_authorized_remote_docker", fail_probe)
    payload = _kwargs()
    payload.update({"operator_confirmed": False, "authorization": "remote_docker"})
    response = client.post("/api/maze/remote-docker/pg292-live", json=payload)
    assert response.status_code == 400
    assert called is False
