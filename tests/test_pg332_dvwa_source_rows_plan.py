from __future__ import annotations

import json
from pathlib import Path

from scripts.plan_pg332_dvwa_source_rows import ROOT, ROLES, SEEDS, build_pg332_dvwa_source_plan, validate_pg332_dvwa_source_plan, write_plan


def test_dvwa_plan_is_static_network_none_and_all_ask() -> None:
    plan = build_pg332_dvwa_source_plan()
    assert plan["status"] == "planning_only"
    assert plan["execution"]["docker_started"] is False and plan["execution"]["network_contacted"] is False
    assert plan["execution"]["network_mode"] == "none" and plan["execution"]["loopback_relay_required"] is True
    assert len(plan["episodes"]) == len(SEEDS) * 3
    assert validate_pg332_dvwa_source_plan(plan)["valid"] is True
    assert "/vulnerabilities/" not in json.dumps(plan)
    assert all(episode["model_projection"] == {"next_action": "ask_typed", "safe_to_send": False} for episode in plan["episodes"])


def test_roles_manifest_and_post_gate_are_fail_closed() -> None:
    plan = build_pg332_dvwa_source_plan(seeds=(33204,))
    post = [episode for episode in plan["episodes"] if episode["method"] == "POST"]
    assert len(post) == 1 and post[0]["typed_contract"]["post_typed_available"] == "unknown_until_evaluator"
    assert all(episode["model_projection"]["next_action"] == "ask_typed" and episode["model_projection"]["safe_to_send"] is False for episode in post)
    assert post[0]["stateful_disposable"] is True
    assert post[0]["state_contract"]["evaluator_side_state_delta_only"] is True
    assert post[0]["state_contract"]["model_context_state_or_payload_allowed"] is False
    for episode in plan["episodes"]:
        assert set(episode["roles"]) == set(ROLES)
        assert len({item["target_identity_sha256"] for item in episode["roles"].values()}) == 4
        if episode["method"] == "POST":
            assert all(item["state_reset_before_required"] and item["state_reset_after_required"] and item["database_clean_attestation_required"] and item["teardown_required"] for item in episode["roles"].values())
        manifest = episode["observation_contract"]["field_capture_manifest"]
        assert sum(len(fields) for fields in manifest.values()) == 107
        assert all(status == "not_observed" for fields in manifest.values() for status in fields.values())
        assert episode["training_eligible"] is False


def test_optional_utf8_output_stays_in_workspace_and_revalidates() -> None:
    target = ROOT / ".pytest-pg332-plan-output.json"
    try:
        written = write_plan(target)
        reloaded = json.loads(target.read_text(encoding="utf-8"))
        assert reloaded["plan_sha256"] == written["plan_sha256"]
        assert validate_pg332_dvwa_source_plan(reloaded)["valid"] is True
    finally:
        target.unlink(missing_ok=True)
    try:
        write_plan(Path("C:/pg332-outside-workspace.json"))
    except ValueError as error:
        assert "inside the workspace" in str(error)
    else:
        raise AssertionError("outside output path accepted")
