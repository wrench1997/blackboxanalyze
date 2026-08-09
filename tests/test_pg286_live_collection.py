from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.pg284_evaluator_contract import sha256_json
from app.pg286_live_collection import collect_pg286_live_record


def _surface() -> dict:
    return {
        "surface_id": "pg286:surface:001",
        "method": "GET",
        "path": "/vul/test/read.php",
        "channel": "query",
        "field_count": 1,
        "authorization": "operator_allowlisted_remote_docker",
        "source_attestation_sha256": "a" * 64,
        "evaluator_kind": "dom_effect",
    }


def _projection(shape: str, *, marker: str = "none") -> dict:
    return {"status_class": "2xx", "shape_sha256": shape, "redirect_hops": 0, "backend_observed": True, "effect_marker": marker}


def _reset() -> dict:
    return {"reset_id": "reset-286-001", "fresh_target": True, "container_recreated": True, "container_restart_used": False, "volume_mount_count": 0, "database_health_gate": "healthy", "state_change_allowed": False}


def _evidence(effect_type: str = "dom_effect") -> dict:
    value = {"effect_type": effect_type, "typed_effect_confirmed": True, "negative_control_clean": True, "reference_agreement": True, "replay_consistent": True, "non_destructive": True, "evaluator_id": "pg286-evaluator-001"}
    value["evidence_sha256"] = sha256_json(value)
    return value


def _kwargs() -> dict:
    return {
        "record_id": "pg286:live:001",
        "surface": _surface(),
        "reset": _reset(),
        "baseline": _projection("a" * 64),
        "reference": _projection("b" * 64, marker="reference"),
        "negative": _projection("c" * 64),
        "candidate": _projection("d" * 64, marker="candidate"),
        "replay": _projection("d" * 64, marker="candidate"),
        "typed_evidence": _evidence(),
        "remote_probe": {"status": "available"},
        "fields": ["query", "submit"],
        "modality_projection": {"browser_dom_observed": True, "marker_hits": 0, "body_text_hits": 0, "element_count": 4, "script_tag_count": 1},
        "operator_reviewed": True,
    }


def test_live_record_requires_cross_seed_review_even_after_typed_effect():
    record = collect_pg286_live_record(**_kwargs())
    assert record["decision"] == "eligible_for_cross_seed_review"
    assert record["promotion"]["collection_complete"] is True
    assert record["evaluator_status"] == "confirmed_effect"
    assert record["token_evidence_status"] == "complete"
    assert record["field_roles"] == ["control", "text"]
    assert record["training_eligible"] is False
    assert record["memory_promotion_allowed"] is False
    assert record["raw_payload_stored"] is False
    assert all("family=" not in token and "oracle=" not in token for token in record["context_tokens"])


def test_remote_unavailable_quarantines_record_without_training_gold():
    kwargs = _kwargs()
    kwargs["remote_probe"] = {"status": "unavailable"}
    record = collect_pg286_live_record(**kwargs)
    assert record["decision"] == "quarantine"
    assert record["promotion"]["collection_complete"] is False
    assert record["evaluator_status"] == "blocked"
    assert record["training_eligible"] is False
    assert "remote_docker_available" in record["reasons"]


def test_missing_typed_modality_is_incomplete():
    kwargs = _kwargs()
    kwargs["modality_projection"] = None
    record = collect_pg286_live_record(**kwargs)
    assert record["decision"] == "quarantine"
    assert record["token_evidence_status"] == "incomplete"
    assert "dom_or_sql_or_logic_or_redirect" in record["missing_modalities"]
    assert "observation_modality_missing" in record["reasons"]


def test_hard_negative_cannot_be_promoted():
    kwargs = _kwargs()
    kwargs["hard_negative"] = True
    record = collect_pg286_live_record(**kwargs)
    assert record["decision"] == "quarantine"
    assert record["hard_negative"] is True
    assert record["promotion"]["collection_complete"] is False
    assert "hard_negative" in record["reasons"]


def test_raw_observation_field_is_rejected():
    kwargs = _kwargs()
    kwargs["modality_projection"] = {"browser_dom_observed": True, "body_text": "raw"}
    with pytest.raises(ValueError, match="raw|unsupported"):
        collect_pg286_live_record(**kwargs)


def test_exact_field_names_are_not_persisted():
    record = collect_pg286_live_record(**_kwargs())
    assert "query" not in record["context_tokens"]
    assert "submit" not in record["context_tokens"]
    assert "field_roles" in record


def test_observation_api_applies_remote_probe_and_returns_bounded_record(monkeypatch):
    monkeypatch.setattr("app.main.probe_authorized_remote_docker", lambda: {"status": "available", "evidence_sha256": "e" * 64})
    payload = {"authorization": "remote_docker", "operator_confirmed": True, **_kwargs()}
    response = TestClient(app).post("/api/maze/remote-docker/observation", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "eligible_for_cross_seed_review"
    assert body["training_eligible"] is False
    assert body["raw_payload_stored"] is False
    assert body["remote_probe"]["status"] == "available"


def test_observation_api_requires_operator_confirmation(monkeypatch):
    called = False

    def fail_probe():
        nonlocal called
        called = True
        raise AssertionError("probe must not run without confirmation")

    monkeypatch.setattr("app.main.probe_authorized_remote_docker", fail_probe)
    response = TestClient(app).post("/api/maze/remote-docker/observation", json={"record_id": "pg286:live:blocked", "operator_confirmed": False})
    assert response.status_code == 400
    assert called is False


def test_pg286_live_protocol_declares_get_post_and_cross_seed_gate():
    import json
    from pathlib import Path

    protocol = json.loads((Path(__file__).resolve().parents[1] / "research" / "pg286_live_protocol_v1.json").read_text(encoding="utf-8"))
    assert protocol["protocol_id"] == "pg286-live-observation-collection-v1"
    assert protocol["runner_contract"]["request_methods"] == ["GET", "POST"]
    assert protocol["promotion_contract"]["cross_seed_minimum"] == 3
    assert protocol["promotion_contract"]["hard_negative_false_allow"] == 0
    assert protocol["current_status"]["remote_docker"] == "unavailable"
    assert protocol["current_status"]["training_gold"] == 0
