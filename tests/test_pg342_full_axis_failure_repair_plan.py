from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.plan_pg342_full_axis_failure_repair import (
    ROOT,
    build_pg342_plan,
    validate_pg342_plan,
    write_plan,
)


def test_plan_is_planning_only_and_has_get_post_and_full_roles() -> None:
    plan = build_pg342_plan()
    assert validate_pg342_plan(plan) == {"valid": True, "failures": []}
    assert plan["status"] == "planning_only"
    assert plan["execution"]["docker_started"] is False
    assert {lane["method"] for lane in plan["lanes"]} == {"GET", "POST"}
    assert len(plan["episodes"]) == 18
    for episode in plan["episodes"]:
        assert set(episode["roles"]) == {"candidate", "reference", "negative", "replay"}
        assert episode["observation_contract"]["required_axis_count"] == 7
        assert episode["observation_contract"]["required_field_count"] == 107
        assert episode["training_eligible"] is False


def test_failure_repair_and_negative_contract_is_explicit() -> None:
    plan = build_pg342_plan()
    for episode in plan["episodes"]:
        contract = episode["failure_repair_contract"]
        assert contract["failure_action_change_required"] is True
        assert contract["previous_action_must_differ_from_next"] is True
        assert contract["negative_action"] == "abstain_after_failure"
        assert episode["typed_contract"]["role_bound_evidence_sha256_required"] is True


def test_tampering_or_training_promotion_fails_closed() -> None:
    plan = build_pg342_plan()
    plan["episodes"][0]["training_eligible"] = True
    assert validate_pg342_plan(plan)["valid"] is False
    assert "training" in validate_pg342_plan(plan)["failures"]


def test_write_plan_is_workspace_bound_and_round_trips(tmp_path: Path) -> None:
    output = ROOT / "research" / "pg342_full_axis_failure_repair_plan_test.json"
    try:
        written = write_plan(output)
        loaded = json.loads(output.read_text(encoding="utf-8"))
        assert loaded["plan_sha256"] == written["plan_sha256"]
        assert validate_pg342_plan(loaded)["valid"] is True
    finally:
        output.unlink(missing_ok=True)
    with pytest.raises(ValueError):
        write_plan(tmp_path / "outside.json")
