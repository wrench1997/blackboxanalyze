from __future__ import annotations

import copy

import pytest

from scripts.plan_pg379_source_collection import (
    METHODS,
    ROLES,
    ROUTE_SHAPES,
    SLOTS,
    build_pg379_source_collection_plan,
    validate_pg379_source_collection_plan,
)


def test_pg379_default_plan_is_read_only_and_uses_current_artifacts() -> None:
    plan = build_pg379_source_collection_plan()
    assert plan["status"] == "planning_only_blocked"
    assert validate_pg379_source_collection_plan(plan)["status"] == "passed"
    assert plan["immutable_existing_artifacts"]["pg333_three_impl_rows"]["record_count"] == 45
    assert plan["immutable_existing_artifacts"]["pg377_webgoat_rows"]["record_count"] == 24
    assert plan["dynamic_page_and_lab_inventory"]["record_count"] == 520
    assert plan["dynamic_page_and_lab_inventory"]["inventory_only"] is True
    assert plan["execution"] == {
        "docker_started": False,
        "network_contacted": False,
        "gpu_touched": False,
        "training_started": False,
        "rows_written": False,
        "split_relabelled": False,
        "network_mode_required_for_future_live_run": "none",
        "explicit_operator_flag_required": "PG379_LOCAL_DOCKER_EVAL=1",
    }
    assert all(value is False for value in plan["promotion"].values())


def test_pg379_route_matrix_prioritizes_balanced_dynamic_get_post_and_13_slots() -> None:
    plan = build_pg379_source_collection_plan()
    routes = plan["route_shape_matrix"]
    assert len(routes) == len(ROUTE_SHAPES) == 12
    assert {route["method"] for route in routes} == set(METHODS)
    assert sum(route["method"] == "GET" for route in routes) == 6
    assert sum(route["method"] == "POST" for route in routes) == 6
    assert all(route["target_slots_required"] == list(SLOTS) for route in routes)
    assert all(route["raw_route_literal_stored"] is False for route in routes)
    scale = plan["expected_source_row_scale"]
    assert scale["planned_train_source_rows"] == 108
    assert scale["planned_holdout_source_rows"] == 108
    assert scale["planned_source_rows_total"] == 216
    assert scale["planned_role_episode_rows_total"] == 288
    assert scale["training_eligible_before_independent_audit"] == 0


def test_pg379_planned_train_and_holdout_are_implementation_disjoint() -> None:
    plan = build_pg379_source_collection_plan()
    train = plan["planned_collections"]["train"]
    holdout = plan["planned_collections"]["implementation_holdout"]
    train_impls = {row["implementation"] for row in train}
    holdout_impls = {row["implementation"] for row in holdout}
    assert train_impls.isdisjoint(holdout_impls)
    assert {row["planned_collection_split"] for row in train} == {"new_train_collection"}
    assert {row["planned_collection_split"] for row in holdout} == {"new_implementation_holdout_collection"}
    assert all(row["roles"] == list(ROLES) for row in [*train, *holdout])
    assert all(row["source_roles"] == ["candidate", "reference", "negative"] for row in [*train, *holdout])
    assert all(row["fresh_reset_per_role"] is True for row in [*train, *holdout])
    assert all(row["typed_candidate_reference_negative_replay"] is True for row in [*train, *holdout])
    assert all(row["failure_repair_episode_required"] is True for row in [*train, *holdout])
    assert all(row["training_eligible_before_audit"] is False for row in [*train, *holdout])


def test_pg379_strict_gates_are_explicitly_planned_not_claimed() -> None:
    plan = build_pg379_source_collection_plan()
    gates = plan["strict_gates"]
    assert set(gates) == {
        "source_implementation_disjoint",
        "split_immutability",
        "full_page_ontology",
        "rule_ir_target",
        "get_post_balance",
        "role_typed_evidence",
        "failure_repair",
        "fresh_local_safety",
        "capacity_and_replay",
    }
    assert all(gate["required"] is True and gate["status"] == "planned_unobserved" for gate in gates.values())
    assert gates["full_page_ontology"]["field_count"] == 107
    assert gates["rule_ir_target"]["slot_count"] == 13
    assert gates["role_typed_evidence"]["negative_violation_max"] == 0
    assert gates["failure_repair"]["failure_rows_without_action_change_max"] == 0
    assert gates["fresh_local_safety"]["external_network"] is False


def test_pg379_validator_blocks_tampered_promotion_or_split() -> None:
    plan = build_pg379_source_collection_plan()
    tampered = copy.deepcopy(plan)
    tampered["promotion"]["training_allowed"] = True
    assert validate_pg379_source_collection_plan(tampered)["status"] == "blocked"
    tampered = copy.deepcopy(plan)
    tampered["execution"]["rows_written"] = True
    assert validate_pg379_source_collection_plan(tampered)["status"] == "blocked"
    tampered = copy.deepcopy(plan)
    tampered["planned_collections"]["implementation_holdout"][0]["implementation"] = tampered["planned_collections"]["train"][0]["implementation"]
    assert validate_pg379_source_collection_plan(tampered)["status"] == "blocked"


def test_pg379_validator_rejects_raw_payload_key_in_plan() -> None:
    plan = build_pg379_source_collection_plan()
    tampered = copy.deepcopy(plan)
    tampered["route_shape_matrix"][0]["payload"] = "literal"
    assert validate_pg379_source_collection_plan(tampered)["status"] == "blocked"


def test_pg379_rejects_duplicate_seeds() -> None:
    with pytest.raises(ValueError, match="unique"):
        build_pg379_source_collection_plan(seeds=(37901, 37901))

