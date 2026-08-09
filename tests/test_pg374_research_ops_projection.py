from __future__ import annotations

import json

from app.research_ops import _pg374_replay_plan_projection, build_research_ops_snapshot


def _plan() -> dict:
    return {
        "schema_version": "pg374-model-selected-replay-plan-v1",
        "status": "planning_only_blocked",
        "implementation": {"implementation_id": "pg374_webgoat_second_implementation", "independent_implementation_required": True},
        "staged_candidate": {"candidate_seed_count": 3, "output_materialized": False, "full_13_slot_output_materialized": False, "typed_live_replay_with_model_selected_wire": False},
        "rule_ir_schema": {"model_slots_pg370": ["safe_to_send"]},
        "execution": {"docker_started": False, "gpu_started": False, "network_contacted": False, "network_mode": "none_required_for_future_live_run", "loopback_only_required": True},
        "fresh_typed_replay_contract": {"candidate_reference_negative_replay_required": True, "fresh_reset_per_seed_route_role": True, "typed_evidence_sha256_required": True, "negative_violation_zero_required": True, "model_selected_separate_from_typed_effect": True, "wire_creation_separate_from_model_selected": True, "observed_in_this_plan": False},
        "counts": {"seeds": 3, "routes": 2, "episodes": 6, "roles": 24, "get_rows": 12, "post_rows": 12, "model_selected": 0, "typed_effect_confirmed": 0, "wire_created": 0, "target_contacted": 0},
        "promotion": {"training_allowed": False},
        "report_sha256": "a" * 64,
        "rows": [{"route_ref_sha256": "b" * 64, "method": "GET", "role": "candidate", "safe_to_send": False}],
    }


def test_pg374_projection_is_bounded_and_closed() -> None:
    projected = _pg374_replay_plan_projection(_plan(), plan_present=True)
    assert projected["artifact_status"] == "planning_only_blocked"
    assert projected["counts"]["roles"] == 24
    assert projected["counts"]["model_selected"] == 0
    assert projected["promotion"]["training_allowed"] is False
    assert projected["raw_material_available"] is False
    assert "rows" not in projected
    assert "route_ref_sha256" not in json.dumps(projected)


def test_pg374_missing_plan_is_pending() -> None:
    projected = _pg374_replay_plan_projection({}, plan_present=False)
    assert projected["artifact_status"] == "pending"
    assert projected["promotion_blocked"] is True


def test_pg374_snapshot_projection_is_present() -> None:
    snapshot = build_research_ops_snapshot()
    model = snapshot["capability"]["model"]["pg374_model_selected_replay_plan"]
    assert model["artifact_status"] == "planning_only_blocked"
    assert model["training_eligible"] is False
    assert model["vulnerability_claim_allowed"] is False
    encoded = json.dumps(model, ensure_ascii=False)
    assert "route_ref_sha256" not in encoded
    assert "payload=" not in encoded
    assert "wire=" not in encoded
