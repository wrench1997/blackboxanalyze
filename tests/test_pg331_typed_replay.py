from __future__ import annotations

from copy import deepcopy
import json

import pytest

from app.pg331_evaluator_sidecar import sha256_json
from scripts.run_pg331_typed_replay import (
    IMAGE,
    NETWORK_MODE,
    ROUTES,
    build_pg331_typed_replay_plan,
    build_pg331_typed_replay_record,
    validate_pg331_typed_replay_plan,
    validate_pg331_typed_replay_record,
)


def _reset() -> dict[str, object]:
    return {
        "reset_id": "pg331-reset-route-role",
        "fresh_reset": True,
        "target_instance_digest": sha256_json("container"),
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
        "effect_class": "result_shape" if typed else "none",
        "projection": {
            "status_class": "2xx",
            "content_type_class": "html",
            "body_shape": "html",
            "response_shape_changed": typed,
            "non_destructive": True,
        },
        "evidence_sha256": sha256_json({"role": role, "typed": typed}),
    }


def _record(*, probes: bool = True) -> dict[str, object]:
    probes_map = None
    if probes:
        # These values model evaluator-only literal probes.  The returned
        # record must contain neither string.
        probes_map = {"candidate": "candidate literal", "reference": "reference literal", "negative": "safe literal"}
    return build_pg331_typed_replay_record(
        route_id="pg331-sql-string-get",
        seed=33101,
        reset=_reset(),
        candidate=_role("candidate", True),
        reference=_role("reference", True),
        negative=_role("negative", False),
        literal_probes=probes_map,
        replay_consistent=True,
        reference_agreement=True,
        negative_control_clean=True,
    )


def test_plan_is_pure_three_route_get_post_allowlist() -> None:
    plan = build_pg331_typed_replay_plan()
    assert plan["execution"]["image"] == IMAGE
    assert plan["execution"]["network_mode"] == NETWORK_MODE
    assert plan["execution"]["real_execution"] is False
    assert len(plan["routes"]) == 3
    assert {item["route_id"] for item in plan["routes"]} == {item["id"] for item in ROUTES}
    assert {item["method"] for item in plan["routes"]} == {"GET", "POST"}
    assert all(set(item["role_containers"]) == {"candidate", "reference", "negative"} for item in plan["routes"])
    assert validate_pg331_typed_replay_plan(plan)["valid"] is True
    assert "/vul/" not in json.dumps(plan, ensure_ascii=False)


def test_record_binds_role_hashes_and_excludes_literal_probes() -> None:
    record = _record()
    encoded = json.dumps(record, ensure_ascii=False)
    assert "candidate literal" not in encoded
    assert "reference literal" not in encoded
    assert "safe literal" not in encoded
    assert record["evaluator_sidecar"]["roles"]["candidate"]["evidence_scope"] == "record_role_bound"
    assert record["evaluator_sidecar"]["confirmed_positive"] is True
    assert record["training_eligible"] is False
    assert validate_pg331_typed_replay_record(record)["valid"] is True


def test_missing_replay_or_negative_is_ask_incomplete() -> None:
    record = build_pg331_typed_replay_record(
        route_id="pg331-sql-id-post",
        seed=33101,
        reset=_reset(),
        candidate=_role("candidate", True),
        reference=_role("reference", True),
        negative={**_role("negative", False), "sent": False},
        replay_consistent=False,
        reference_agreement=True,
        literal_probes=None,
    )
    sidecar = record["evaluator_sidecar"]
    assert sidecar["confirmed_positive"] is False
    assert "negative_present" in sidecar["reasons"]
    assert "replay_consistent" in sidecar["reasons"]
    assert record["model_context"]["negative_control"] is False
    assert record["model_context"]["typed_available"] is False
    assert record["model_context"]["replay_ready"] is False
    assert validate_pg331_typed_replay_record(record)["valid"] is True


def test_unknown_route_and_tampered_plan_fail_closed() -> None:
    with pytest.raises(ValueError, match="allowlisted"):
        build_pg331_typed_replay_record(
            route_id="not-allowlisted",
            seed=33101,
            reset=_reset(),
            candidate=_role("candidate", True),
            reference=_role("reference", True),
            negative=_role("negative", False),
        )
    plan = deepcopy(build_pg331_typed_replay_plan())
    plan["execution"]["network_mode"] = "bridge"
    assert validate_pg331_typed_replay_plan(plan)["valid"] is False


def test_raw_role_projection_is_rejected() -> None:
    candidate = _role("candidate", True)
    candidate["projection"] = {"response_body": "raw"}
    with pytest.raises(ValueError, match="raw"):
        build_pg331_typed_replay_record(
            route_id="pg331-sql-search-get",
            seed=33101,
            reset=_reset(),
            candidate=candidate,
            reference=_role("reference", True),
            negative=_role("negative", False),
        )
