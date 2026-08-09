from __future__ import annotations

import json
from pathlib import Path

from app.pg331_evaluator_sidecar import sha256_json
from scripts.run_pg331_juice_shop_source_rows_plan import (
    IMAGE,
    ROUTES,
    ROLES,
    SEEDS,
    bind_source_row,
    build_evaluator_binding,
    build_pg331_juice_shop_source_plan,
    validate_evaluator_binding,
    validate_pg331_juice_shop_plan,
)


def _reset() -> dict[str, object]:
    return {
        "reset_id": "juice-reset-fixture",
        "fresh_reset": True,
        "target_instance_digest": sha256_json("juice-container"),
        "network_mode": "none",
        "external_network": False,
        "loopback_only": True,
        "state_clean": True,
        "volume_mount_count": 0,
        "container_restart_used": False,
    }


def _role(role: str, typed: bool) -> dict[str, object]:
    return {
        "sent": True,
        "available": True,
        "executed": True,
        "typed_effect_confirmed": typed,
        "effect_class": "dom_effect" if typed else "none",
        "projection": {
            "status_class": "2xx",
            "content_type_class": "json",
            "body_shape": "json",
            "response_shape_changed": typed,
            "dom_script_execution": typed,
            "non_destructive": True,
            "database_touched": False,
        },
        "evidence_sha256": sha256_json({"role": role, "typed": typed}),
    }


def _complete_observation() -> dict[str, object]:
    # Use the repository's full abstract fixture to exercise the adapter.  It
    # contains no URL, wire value or response body.
    from tests.test_pg331_source_row import _observation

    return _observation()


def _manifest() -> dict[str, dict[str, str]]:
    from tests.test_pg331_source_row import _field_capture_manifest

    return _field_capture_manifest()


def test_plan_is_planning_only_and_covers_seeds_routes_roles_and_ontology() -> None:
    plan = build_pg331_juice_shop_source_plan()
    assert plan["status"] == "planning_only"
    assert plan["execution"]["real_execution"] is False
    assert plan["execution"]["docker_started"] is False
    assert plan["execution"]["network_mode"] == "none"
    assert plan["execution"]["image"] == IMAGE
    assert plan["seeds"] == list(SEEDS)
    assert plan["episode_count"] == len(SEEDS) * len(ROUTES)
    assert plan["ontology_contract"] ["axis_count"] == 7
    assert plan["ontology_contract"]["field_count"] == 107
    assert validate_pg331_juice_shop_plan(plan)["valid"] is True
    serialized = json.dumps(plan, ensure_ascii=False)
    assert "/rest/" not in serialized
    assert '"wire":' not in serialized
    assert all(set(episode["roles"]) == set(ROLES) for episode in plan["episodes"])
    assert all(episode["training_eligible"] is False for episode in plan["episodes"])


def test_plan_has_explicit_not_observed_manifest_and_post_ask_lane() -> None:
    plan = build_pg331_juice_shop_source_plan()
    post_episodes = [episode for episode in plan["episodes"] if episode["method"] == "POST"]
    get_episodes = [episode for episode in plan["episodes"] if episode["method"] == "GET"]
    assert len(post_episodes) == len(get_episodes) == len(SEEDS) * 3
    for episode in plan["episodes"]:
        contract = episode["observation_contract"]
        assert set(contract["axis_presence"].values()) == {"not_observed"}
        assert all(set(fields.values()) == {"not_observed"} for fields in contract["field_capture_manifest"].values())
        assert sum(len(fields) for fields in contract["field_capture_manifest"].values()) == 107
    assert all(episode["model_context_projection"]["next_action"] == "ask_typed" for episode in post_episodes)
    assert all(episode["model_context_projection"]["safe_to_send"] is False for episode in plan["episodes"])
    assert all(episode["lane"] == "unsupported_post_ask" for episode in post_episodes)


def test_fresh_identity_is_unique_per_route_role_and_replay() -> None:
    plan = build_pg331_juice_shop_source_plan(seeds=(33101,))
    episode = plan["episodes"][0]
    identities = {entry["container_identity_sha256"] for entry in episode["roles"].values()}
    assert len(identities) == 4
    assert all(entry["fresh_reset_required"] is True for entry in episode["roles"].values())
    assert all(entry["fresh_reset_observed"] is False for entry in episode["roles"].values())
    assert all(entry["external_network"] is False and entry["network_mode"] == "none" for entry in episode["roles"].values())


def test_evaluator_binding_hashes_literal_probes_off_record_and_fails_closed() -> None:
    binding = build_evaluator_binding(
        seed=33101,
        route_id="juice-track-order-xss-get",
        reset=_reset(),
        candidate=_role("candidate", True),
        reference=_role("reference", True),
        negative=_role("negative", False),
        literal_probes={"candidate": "candidate literal", "reference": "reference literal", "negative": "safe literal"},
        replay_consistent=True,
        reference_agreement=True,
        negative_control_clean=True,
        replay_evidence_sha256=sha256_json("replay-evidence"),
    )
    serialized = json.dumps(binding, ensure_ascii=False)
    assert "candidate literal" not in serialized
    assert "reference literal" not in serialized
    assert "safe literal" not in serialized
    assert binding["evaluator_sidecar"]["confirmed_positive"] is True
    assert binding["promotion"]["training_allowed"] is False
    assert validate_evaluator_binding(binding)["valid"] is True


def test_missing_negative_or_replay_remains_ask_and_binding_is_valid_diagnostic() -> None:
    binding = build_evaluator_binding(
        seed=33101,
        route_id="juice-products-search-post-unsupported",
        reset=_reset(),
        candidate=_role("candidate", True),
        reference=_role("reference", True),
        negative={**_role("negative", False), "sent": False},
        replay_consistent=False,
    )
    assert binding["evaluator_sidecar"]["confirmed_positive"] is False
    assert binding["model_context"]["typed_available"] is False
    assert binding["model_context"]["negative_control"] is False
    assert validate_evaluator_binding(binding)["valid"] is True


def test_bind_source_row_uses_strict_collector_without_operator_promotion() -> None:
    from app.pg331_source_row import validate_pg331_source_row

    evaluator = {
        "typed_available": True,
        "negative_control": True,
        "reference_present": True,
        "candidate_present": True,
        "fresh_reset": True,
        "evidence_hash": "d" * 64,
        "confirmed_positive": False,
        "effect_class": "dom_effect",
        "evaluator_version": "fixture-juice-evaluator-v1",
    }
    row = bind_source_row(
        seed=33101,
        route_id="juice-track-order-xss-get",
        role="candidate",
        observation=_complete_observation(),
        reset=_reset(),
        evaluator=evaluator,
        field_capture_manifest=_manifest(),
    )
    assert row["training_eligible"] is False
    assert row["operator_reviewed"] is False
    assert row["promotion"]["memory_promotion_allowed"] is False
    assert validate_pg331_source_row(row)["valid"] is True
    assert "/rest/" not in json.dumps(row, ensure_ascii=False)


def test_unsupported_post_source_binding_rejects_send_target() -> None:
    try:
        bind_source_row(
            seed=33101,
            route_id="juice-login-post-unsupported",
            role="candidate",
            observation={},
            reset=_reset(),
            evaluator={},
            field_capture_manifest={},
            target_projection={"safe_to_send": True},
        )
    except ValueError as error:
        assert "ASK-only" in str(error)
    else:  # pragma: no cover - explicit fail-closed assertion
        raise AssertionError("unsupported POST binding accepted safe_to_send=true")
