from __future__ import annotations

import json

from app.pg331_evaluator_sidecar import sha256_json
from app.pg331_source_row import validate_pg331_source_row
from scripts.plan_pg331_vulnerableapp_source_rows import ROLES, SEEDS, bind_source_row, build_pg331_vulnerableapp_source_plan, validate_pg331_vulnerableapp_source_plan


def test_plan_is_static_full_ontology_and_non_promoting() -> None:
    plan = build_pg331_vulnerableapp_source_plan()
    assert plan["status"] == "planning_only"
    assert plan["execution"]["docker_started"] is False
    assert plan["execution"]["network_contacted"] is False
    assert plan["ontology_contract"] == {"axis_count": 7, "field_count": 107, "missing_status_forces_ask": True}
    assert plan["episode_count"] == len(SEEDS) * 6
    assert validate_pg331_vulnerableapp_source_plan(plan)["valid"] is True
    assert "/VulnerableApp/" not in json.dumps(plan)
    assert all(episode["training_eligible"] is False for episode in plan["episodes"])


def test_plan_requires_distinct_fresh_role_resets_and_ask() -> None:
    episode = build_pg331_vulnerableapp_source_plan(seeds=(24604,))["episodes"][0]
    assert set(episode["roles"]) == set(ROLES)
    assert len({entry["target_identity_sha256"] for entry in episode["roles"].values()}) == 4
    assert all(entry["fresh_reset_required"] is True and entry["fresh_reset_observed"] is False for entry in episode["roles"].values())
    assert episode["model_context_projection"] == {"next_action": "ask_typed", "safe_to_send": False}
    assert all(status == "not_observed" for axis in episode["observation_contract"]["field_capture_manifest"].values() for status in axis.values())


def test_future_binding_cannot_promote_and_post_send_fails_closed() -> None:
    from tests.test_pg331_source_row import _field_capture_manifest, _observation

    reset = {"fresh_reset": True, "reset_id": "vapp-fixture", "target_instance_digest": sha256_json("vapp"), "network_mode": "loopback", "external_network": False, "loopback_only": True, "state_clean": True}
    evaluator = {"typed_available": True, "negative_control": True, "reference_present": True, "candidate_present": True, "fresh_reset": True, "evidence_hash": "a" * 64, "confirmed_positive": False, "effect_class": "dom_effect", "evaluator_version": "vapp-fixture-v1"}
    row = bind_source_row(seed=24604, case_id="vapp_html_level1_get", role="candidate", observation=_observation(), reset=reset, evaluator=evaluator, field_capture_manifest=_field_capture_manifest())
    assert row["training_eligible"] is False
    assert row["operator_reviewed"] is False
    assert validate_pg331_source_row(row)["valid"] is True
    try:
        bind_source_row(seed=24604, case_id="vapp_html_level1_post_405", role="candidate", observation={}, reset={}, evaluator={}, field_capture_manifest={}, target_projection={"safe_to_send": True})
    except ValueError as error:
        assert "ASK-only" in str(error)
    else:
        raise AssertionError("POST send was accepted")
