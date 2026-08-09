from __future__ import annotations

from scripts.plan_pg368_second_implementation import (
    IMAGE,
    PG367_IMPLEMENTATION_ID,
    ROUTES,
    SEEDS,
    build_pg368_second_implementation_audit,
    build_pg368_second_implementation_plan,
    validate_pg368_second_implementation_plan,
)


def test_pg368_selects_source_attested_webgoat_as_different_implementation():
    plan = build_pg368_second_implementation_plan()
    implementation = plan["implementation"]
    assert implementation["image"] == IMAGE
    assert implementation["independent_from"] == PG367_IMPLEMENTATION_ID
    contract = implementation["independence_contract"]
    assert all(contract[key] is True for key in contract)
    assert plan["status"] == "planning_only"
    assert plan["source_attestation"]["source_audit_status"] == "passed"


def test_pg368_has_get_post_and_all_four_role_lanes_per_fresh_episode():
    plan = build_pg368_second_implementation_plan()
    assert {route["method"] for route in plan["routes"]} == {"GET", "POST"}
    assert len(plan["episodes"]) == len(SEEDS) * len(ROUTES)
    for episode in plan["episodes"]:
        assert set(episode["roles"]) == {"candidate", "reference", "negative", "replay"}
        assert episode["typed_contract"] == {
            "candidate_reference_negative_replay": True,
            "fresh_reset_per_role": True,
            "evidence_sha256": True,
            "method_shape_only": True,
            "vulnerability_oracle": False,
        }
        for role in episode["roles"].values():
            assert role["fresh_container_required"] is True
            assert role["fresh_reset_before_required"] is True
            assert role["fresh_reset_after_required"] is True
            assert role["role_bound_evidence_sha256_required"] is True
            assert role["training_eligible"] is False
            assert role["model_projection"]["question"] == "ask_typed"
            assert role["model_projection"]["safe_to_send"] is False


def test_pg368_is_fail_closed_and_does_not_start_or_promote_anything():
    plan = build_pg368_second_implementation_plan()
    assert validate_pg368_second_implementation_plan(plan)["status"] == "passed"
    execution = plan["execution"]
    assert execution["docker_started"] is False
    assert execution["network_contacted"] is False
    assert execution["network_mode"] == "none"
    assert execution["published_ports_allowed"] is False
    assert execution["bind_or_volume_mounts_allowed"] is False
    assert all(value is False for key, value in plan["promotion"].items() if key.endswith("_allowed"))
    assert plan["model_context_policy"]["raw_payload_or_probe"] is False
    assert plan["model_context_policy"]["raw_request_or_response"] is False
    assert plan["model_context_policy"]["url_or_route_literal"] is False


def test_pg368_rejects_tampering_and_cannot_be_relabelled_as_live():
    plan = build_pg368_second_implementation_plan()
    tampered = {**plan, "status": "completed"}
    assert validate_pg368_second_implementation_plan(tampered)["status"] == "blocked"
    tampered = {**plan, "execution": {**plan["execution"], "network_contacted": True}}
    assert validate_pg368_second_implementation_plan(tampered)["status"] == "blocked"


def test_pg368_audit_is_read_only_and_promotion_closed():
    plan = build_pg368_second_implementation_plan()
    audit = build_pg368_second_implementation_audit(plan)
    assert audit["status"] == "passed"
    assert audit["target_contacted"] is False
    assert audit["docker_started"] is False
    assert audit["network_contacted"] is False
    assert audit["plan_sha256"] == plan["plan_sha256"]
    assert all(value is False for key, value in audit["promotion"].items() if key.endswith("_allowed"))
