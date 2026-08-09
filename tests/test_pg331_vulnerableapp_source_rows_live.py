from __future__ import annotations

import json

import pytest

from scripts.run_pg331_vulnerableapp_source_rows_live import ROLES, SEEDS, build_live_contract, prepare_role_capture, run


def test_live_contract_is_planning_only_and_blocks_legacy_bridge_lifecycle() -> None:
    contract = build_live_contract()
    assert contract["status"] == "blocked_network_contract"
    assert contract["execution"]["planning_only"] is True
    assert contract["execution"]["docker_started"] is False
    assert contract["execution"]["network_contacted"] is False
    assert contract["execution"]["required_network_mode"] == "none"
    assert contract["execution"]["legacy_reset_compatible"] is False
    assert len(contract["episodes"]) == len(SEEDS) * 6
    assert all(set(episode["roles"]) == set(ROLES) for episode in contract["episodes"])
    assert all(len({role["container_name"] for role in episode["roles"].values()}) == 4 for episode in contract["episodes"])
    assert all(episode["relay_contract"]["legacy_bridge_reclassification_allowed"] is False for episode in contract["episodes"])
    assert all(episode["relay_contract"]["role_container_names"] == {role: details["container_name"] for role, details in episode["roles"].items()} for episode in contract["episodes"])
    assert all(role["fresh_reset_evidence_sha256_required"] is True for episode in contract["episodes"] for role in episode["roles"].values())
    assert all(episode["training_eligible"] is False for episode in contract["episodes"])
    assert "/VulnerableApp/" not in json.dumps(contract)


def test_role_capture_is_abstract_and_post_is_ask_only() -> None:
    captured = prepare_role_capture(method="POST", html="<html><body><form></form></body></html>", headers={"Location": "/done"}, request_projection={"method": "POST", "parameters": []}, response_projection={"status": 302, "body_length": 0}, post_supported=False)
    assert captured["observation"]["response_transport"]["status_class"] == "3xx"
    assert captured["typed_projection"] == {"typed_available": False, "next_action": "ask_typed", "safe_to_send": False}
    assert captured["training_eligible"] is False
    assert "<html" not in str(captured)


def test_run_never_starts_legacy_lifecycle() -> None:
    with pytest.raises(RuntimeError, match="blocked"):
        run()
