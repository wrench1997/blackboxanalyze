from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.run_pg379_source_collection import (
    build_pg379_source_collection_contract,
    validate_pg379_projection,
    validate_pg379_source_collection_contract,
)


HTML = "<!doctype html><html><head><script>history.pushState({}, '', '#x')</script></head><body><form method='get'><input name='q'></form></body></html>"


def _reset() -> dict[str, object]:
    return {
        "reset_id": "pg379-diagnostic-reset",
        "fresh_reset": True,
        "target_instance_digest": "a" * 64,
        "network_mode": "none",
        "external_network": False,
        "loopback_only": True,
        "state_clean": True,
        "volume_mount_count": 0,
        "container_restart_used": False,
    }


def test_default_contract_is_planning_only_and_closed() -> None:
    contract = build_pg379_source_collection_contract()
    assert contract["status"] == "planning_only_live_blocked"
    assert contract["rows_emitted"] is False
    assert contract["rows_emitted_count"] == 0
    assert contract["diagnostic_projections"] == []
    assert all(value is False for value in contract["execution"].values())
    assert all(value is False for value in contract["promotion"].values())
    assert validate_pg379_source_collection_contract(contract)["status"] == "passed"


def test_route_role_slot_contract_has_get6_post6_and_thirteen_slots() -> None:
    contract = build_pg379_source_collection_contract()
    routes = contract["route_contract"]["routes"]
    assert len(routes) == 12
    assert contract["route_contract"]["method_counts"] == {"GET": 6, "POST": 6}
    assert set(contract["role_contract"]) == {"candidate", "reference", "negative", "replay"}
    assert contract["rule_ir_target"]["slot_count"] == 13
    assert len(contract["rule_ir_target"]["slots"]) == 13
    assert all(route["target_slots_required"] == contract["rule_ir_target"]["slots"] for route in routes)


def test_live_request_stays_blocked_even_with_operator_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PG379_LOCAL_DOCKER_EVAL", "1")
    contract = build_pg379_source_collection_contract(live_requested=True)
    assert contract["live_gate"]["requested"] is True
    assert contract["live_gate"]["operator_flag_present"] is True
    assert contract["live_gate"]["ready"] is False
    assert contract["live_gate"]["status"] == "blocked_unbound_implementation_attestations"
    assert all(not entry["attestation"]["bound"] for entry in contract["independent_implementations"].values())
    assert validate_pg379_source_collection_contract(contract)["status"] == "passed"


def test_projection_reuses_pg377_adapter_but_emits_no_training_row() -> None:
    result = validate_pg379_projection(
        implementation="train",
        route_class="get_query_html_text",
        seed=37901,
        role="candidate",
        html=HTML,
        headers={"Content-Type": "text/html"},
        request_projection={"method": "GET", "parameters": [{"role": "query_term"}]},
        response_projection={"status": 200, "body_length": 64, "body_shape": "html"},
        reset=_reset(),
    )
    assert result["status"] == "projection_validated_diagnostic"
    assert result["rows_emitted"] is False
    assert result["training_eligible"] is False
    assert result["attestation_bound"] is False
    assert result["adapter_validation"]["valid"] is True
    assert result["diagnostic_row"]["field_capture_manifest_count"] == 107
    assert result["diagnostic_row"]["context_firewall"] == {"forbidden_token_count": 0, "sidecars_off_context": True}
    assert HTML not in str(result)


def test_projection_rejects_raw_response_projection_without_side_effects() -> None:
    result = validate_pg379_projection(
        implementation="holdout",
        route_class="post_json_state_transition",
        seed=37902,
        role="negative",
        html="<html><body>page</body></html>",
        request_projection={"method": "POST", "parameters": [{"role": "json_value"}]},
        response_projection={"response_body": "raw literal"},
        reset=_reset(),
    )
    assert result["status"] == "projection_blocked"
    assert result["rows_emitted"] is False
    assert result["training_eligible"] is False
    assert any(reason.startswith("adapter_rejected:") for reason in result["blocked_reasons"])
    assert result["diagnostic_row"] is None


def test_sidecar_and_failure_repair_belief_contract_is_explicit() -> None:
    contract = build_pg379_source_collection_contract()
    sidecar = contract["sidecar_context_firewall"]
    assert sidecar["typed_sidecar_evaluator_only"] is True
    assert sidecar["evidence_sha256_evaluator_only"] is True
    assert sidecar["oracle_answer_in_context"] is False
    repair = contract["failure_repair_belief_contract"]
    assert repair["failure_action_change_required"] is True
    assert repair["repair_action_required"] is True
    assert repair["belief_prior_posterior_delta_required"] is True
    assert repair["role_bound_evidence_required"] is True


def test_contract_tampering_is_detected() -> None:
    contract = build_pg379_source_collection_contract()
    tampered = deepcopy(contract)
    tampered["independent_implementations"]["train"]["attestation"]["bound"] = True
    result = validate_pg379_source_collection_contract(tampered)
    assert result["status"] == "blocked"
    assert "attestation:train:bound" in result["failures"]
    assert "contract_hash_mismatch" in result["failures"]

