from __future__ import annotations

import json

import pytest

from app.pg331_evaluator_sidecar import sha256_json
from scripts.run_pg332_dvwa_source_rows_live import IMAGE, RELAY_HOST, ROLES, attest_container_inspection, build_container_command, build_live_contract, build_evaluator_sidecar, container_name, run


def test_contract_is_network_none_per_role_and_never_reclassifies_bridge() -> None:
    contract = build_live_contract(seeds=(33204,))
    assert contract["status"] == "planning_only_reviewed_contract"
    assert contract["execution"]["docker_started"] is False and contract["execution"]["network_contacted"] is False
    assert len(contract["episodes"]) == 3
    assert "/vulnerabilities/" not in json.dumps(contract)
    for episode in contract["episodes"]:
        assert set(episode["roles"]) == set(ROLES)
        assert len({item["target_identity_sha256"] for item in episode["roles"].values()}) == 4
        assert episode["transport"] == {"network_mode": "none", "relay_host": RELAY_HOST, "loopback_only": True, "published_ports_allowed": False, "bind_or_volume_mounts_allowed": False, "legacy_bridge_reclassification_allowed": False}
        assert episode["model_projection"]["safe_to_send"] is False


def test_bounded_evaluator_sidecar_keeps_raw_values_out_of_model_context() -> None:
    reset = {"reset_id": "r", "fresh_reset": True, "target_instance_digest": "a" * 64, "network_mode": "none", "external_network": False, "loopback_only": True, "state_clean": True, "volume_mount_count": 0}
    role = lambda typed: {"sent": True, "available": True, "executed": True, "typed_effect_confirmed": typed, "effect_class": "dom_effect" if typed else "none", "projection": {"status_class": "2xx", "body_shape": "html", "non_destructive": True}, "evidence_sha256": sha256_json({"typed": typed})}
    sidecar = build_evaluator_sidecar(seed=33204, route_ref_sha256="b" * 64, reset=reset, candidate=role(True), reference=role(True), negative=role(False), replay_consistent=True)
    assert sidecar["training_eligible"] is False
    # Model context contains only availability facts, not evaluator projection.
    assert set(sidecar["model_context"]) == {
        "typed_available", "evidence_present", "negative_control", "fresh_reset",
        "reference_present", "candidate_present", "replay_ready", "step_budget",
    }
    assert "fixture-secret-literal" not in json.dumps(sidecar).casefold()

    unsafe = role(True)
    unsafe["projection"] = {"response_body": "fixture-secret-literal"}
    with pytest.raises(ValueError, match="raw"):
        build_evaluator_sidecar(seed=33204, route_ref_sha256="b" * 64, reset=reset, candidate=unsafe, reference=role(True), negative=role(False), replay_consistent=True)


def test_runtime_gate_never_starts_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PG332_LOCAL_DOCKER_EVAL", raising=False)
    with pytest.raises(RuntimeError, match="PG332_LOCAL_DOCKER_EVAL"): run()
    monkeypatch.setenv("PG332_LOCAL_DOCKER_EVAL", "1")
    report = run()
    assert report["status"] == "incomplete_environment_failure"
    assert report["target_contacted"] is False and report["docker_started"] is False
    assert report["model_projection"]["next_action"] == "ask_typed"


def test_fake_runtime_attestation_is_fixed_digest_network_none_and_fresh_role_name() -> None:
    route_ref = "c" * 64
    name = container_name(seed=33204, route_ref_sha256=route_ref, role="candidate")
    command = build_container_command(seed=33204, route_ref_sha256=route_ref, role="candidate")
    assert command[:3] == ("docker", "run", "--detach")
    assert command[command.index("--network") + 1] == "none"
    assert "-p" not in command and "--publish" not in command and "--volume" not in command and "--mount" not in command
    assert command.count("--cap-add") == 5
    assert {command[index + 1] for index, value in enumerate(command[:-1]) if value == "--cap-add"} == {"DAC_OVERRIDE", "CHOWN", "FOWNER", "SETUID", "SETGID"}
    assert command[-1] == IMAGE and name in command
    attestation = attest_container_inspection({"name": "/" + name, "image": IMAGE, "network_mode": "none", "mounts": [], "published_ports": [], "relay_host": RELAY_HOST, "legacy_bridge_reclassified": False}, expected_name=name)
    assert attestation["valid"] is True
    bad = attest_container_inspection({"name": "/" + name, "image": IMAGE, "network_mode": "bridge", "mounts": ["x"], "published_ports": ["x"], "relay_host": "0.0.0.0", "legacy_bridge_reclassified": True}, expected_name=name)
    assert set(bad["failures"]) >= {"network_mode", "mounts", "published_ports", "relay_host", "legacy_bridge_reclassified"}
