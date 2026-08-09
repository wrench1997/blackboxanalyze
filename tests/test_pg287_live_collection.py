from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.pg286_live_collection import collect_pg286_live_record
from app.pg287_live_batch import audit_pg287_live_batch
from app.pg287_live_collection import collect_pg287_live_record as collect_pg287_live
from app.pg284_evaluator_contract import sha256_json
from app.main import app


def _pg286_kwargs(index: int = 1, *, method: str = "GET", effect_type: str = "dom_effect") -> dict:
    surface = {
        "surface_id": f"pg286:surface:{index:03d}",
        "method": method,
        "path": "/vul/test/read.php",
        "channel": "query" if method == "GET" else "form",
        "field_count": 1,
        "authorization": "operator_allowlisted_remote_docker",
        "source_attestation_sha256": "a" * 64,
        "evaluator_kind": effect_type,
    }
    projection = lambda shape, marker="none": {"status_class": "2xx", "shape_sha256": shape, "redirect_hops": 0, "backend_observed": True, "effect_marker": marker}
    reset = {"reset_id": f"reset-287-{index:03d}", "fresh_target": True, "container_recreated": True, "container_restart_used": False, "volume_mount_count": 0, "database_health_gate": "healthy", "state_change_allowed": False}
    evidence = {"effect_type": effect_type, "typed_effect_confirmed": True, "negative_control_clean": True, "reference_agreement": True, "replay_consistent": True, "non_destructive": True, "evaluator_id": f"pg287-evaluator-{index:03d}"}
    evidence["evidence_sha256"] = sha256_json(evidence)
    modality = {"browser_dom_observed": True, "marker_hits": 0, "body_text_hits": 0, "element_count": 4, "script_tag_count": 1} if effect_type == "dom_effect" else {"transition_delta": "visibility", "visibility_changed": True, "state_changed": False}
    return {
        "record_id": f"pg286:live:{index:03d}",
        "surface": surface,
        "reset": reset,
        "baseline": projection("a" * 64),
        "reference": projection("b" * 64, marker="reference"),
        "negative": projection("c" * 64),
        "candidate": projection("d" * 64, marker="candidate"),
        "replay": projection("d" * 64, marker="candidate"),
        "typed_evidence": evidence,
        "remote_probe": {"status": "available"},
        "fields": ["query", "submit"],
        "modality_projection": modality,
        "operator_reviewed": True,
    }


def _attestation(index: int = 1) -> dict:
    return {"authorized": True, "authorization_id": "auth-pg287", "collector_id": "collector-pg287", "target_instance_digest": f"{index:064x}", "image_digest": "b" * 64, "source_digest": "c" * 64}


def _plan(method: str = "GET", action: str = "candidate_probe", encoding: str = "url_percent") -> dict:
    return {"final_action": action, "method": method, "probe_class": "xss", "channel": "query" if method == "GET" else "form", "encoding": encoding, "wire_kind": "query_param" if method == "GET" else "form_field", "field_slot": "observed_or_runtime_canary", "repair_delta": "none", "safe_to_send": False, "oracle_required": True}


def _live(index: int = 1, *, method: str = "GET", split: str = "family_holdout", encoding: str = "url_percent", field_role: str = "text", action: str = "candidate_probe", effect_type: str = "dom_effect") -> dict:
    observation = collect_pg286_live_record(**_pg286_kwargs(index, method=method, effect_type=effect_type))
    return collect_pg287_live(
        observation_record=observation,
        observed_binding={"encoding": encoding, "field_role": field_role, "evidence_hash": observation["evidence_hash"], "observation_id": f"obs-{index:03d}"},
        reference_plan=_plan(method, action, encoding),
        source_attestation=_attestation(index),
        remote_probe={"status": "available"},
        split=split,
        operator_reviewed=True,
    )


def test_live_resolved_record_is_bounded_and_training_candidate_only():
    record = _live()
    assert record["variant"] == "resolved"
    assert record["training_eligible"] is True
    assert record["memory_promotion_allowed"] is False
    assert record["quality"]["fresh_reset"] is True
    assert "encoding_observed=url_percent" in record["context_tokens"]
    assert "observation_sufficiency=resolved" in record["context_tokens"]
    assert all("family=" not in token and "oracle=" not in token for token in record["context_tokens"])
    assert record["raw_payload_strings_stored"] is False
    assert record["raw_response_bodies_stored"] is False


def test_live_missing_encoding_requires_ask_target():
    record = _live(2, encoding="unknown", field_role="unknown", action="ask_typed")
    assert record["variant"] == "ambiguous"
    assert record["target"]["next_action"] == "ask_typed"
    assert record["target"]["safe_to_send"] is False
    assert "observation_sufficiency=ambiguous" in record["context_tokens"]


def test_live_collection_rejects_unavailable_remote_and_raw_plan():
    observation = collect_pg286_live_record(**_pg286_kwargs())
    with pytest.raises(ValueError, match="available"):
        collect_pg287_live(observation_record=observation, observed_binding={"encoding": "plain", "field_role": "text", "evidence_hash": observation["evidence_hash"], "observation_id": "obs"}, reference_plan=_plan(), source_attestation=_attestation(), remote_probe={"status": "unavailable"})
    with pytest.raises(ValueError, match="raw"):
        collect_pg287_live(observation_record=observation, observed_binding={"encoding": "plain", "field_role": "text", "evidence_hash": observation["evidence_hash"], "observation_id": "obs"}, reference_plan={**_plan(), "payload": "raw"}, source_attestation=_attestation(), remote_probe={"status": "available"})


def test_identifiability_api_is_ingest_only_and_fail_closed(monkeypatch):
    observation = collect_pg286_live_record(**_pg286_kwargs())
    payload = {
        "authorization": "remote_docker",
        "operator_confirmed": True,
        "observation_record": observation,
        "observed_binding": {"encoding": "url_percent", "field_role": "text", "evidence_hash": observation["evidence_hash"], "observation_id": "api-obs-001"},
        "reference_plan": _plan(),
        "source_attestation": _attestation(),
        "split": "family_holdout",
        "operator_reviewed": True,
    }
    monkeypatch.setattr("app.main.probe_authorized_remote_docker", lambda: {"status": "available", "evidence_sha256": "e" * 64})
    response = TestClient(app).post("/api/maze/remote-docker/identifiability", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["variant"] == "resolved"
    assert body["training_eligible"] is True
    assert body["memory_promotion_allowed"] is False
    monkeypatch.setattr("app.main.probe_authorized_remote_docker", lambda: {"status": "unavailable", "evidence_sha256": "f" * 64})
    blocked = TestClient(app).post("/api/maze/remote-docker/identifiability", json=payload)
    assert blocked.status_code == 409


def test_live_batch_requires_family_resolved_coverage_and_get_post_resets():
    rows = [_live(index, method="GET" if index <= 3 else "POST", effect_type="dom_effect" if index % 2 else "logic_transition") for index in range(1, 7)]
    hard = dict(rows[0])
    hard["hard_negative"] = True
    hard["training_eligible"] = False
    hard["promotion"] = {**hard["promotion"], "training_eligible": False}
    hard["record_sha256"] = sha256_json({key: value for key, value in hard.items() if key != "record_sha256"})
    result = audit_pg287_live_batch(rows, hard_negative_records=[hard], independent_audit_pass=True, remote_docker_status="available", min_typed_modalities=2)
    assert result["status"] == "ready_for_remote_a800_training"
    assert result["family_resolved_count"] == 6
    assert result["get_count"] == 3
    assert result["post_count"] == 3
    no_family = audit_pg287_live_batch([dict(row, split="route_dev") for row in rows], hard_negative_records=[hard], independent_audit_pass=True, remote_docker_status="available", min_typed_modalities=2)
    assert no_family["status"] == "blocked"
    assert "family_resolved_coverage" in no_family["blocking_reasons"]


def test_pg287_live_protocol_is_explicitly_fail_closed():
    import hashlib
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "research" / "pg287_live_protocol_v1.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    digest = protocol.pop("protocol_sha256")
    expected = hashlib.sha256(json.dumps(protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert digest == expected
    assert protocol["endpoint"] == "POST /api/maze/remote-docker/identifiability"
    assert protocol["current_status"]["remote_docker"] == "unavailable"
    assert protocol["current_status"]["family_resolved_coverage"] == "missing"
    assert protocol["promotion_contract"]["memory_promotion_allowed"] is False
