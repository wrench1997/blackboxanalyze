from __future__ import annotations

import json

from scripts import run_pg388_logic_supplement_canary_local as local


def test_supplemental_canary_replays_all_candidate_only_cases_with_role_resets() -> None:
    report = local.run()
    assert report["status"] == "completed_local_supplemental_canary_diagnostic"
    assert report["counts"] == {
        "cases": 10,
        "seeds": 3,
        "roles": 4,
        "role_rows": 120,
        "fresh_resets_before": 120,
        "fresh_resets_after": 120,
        "setup_observations": 60,
        "typed_observations": 120,
        "candidate_effects": 30,
        "replay_effects": 30,
        "negative_control_clean": 30,
        "negative_violation": 0,
        "unsafe_allow": 0,
    }
    assert len(report["rows"]) == 120
    assert {row["role"] for row in report["rows"]} == {"candidate", "reference", "negative", "replay"}
    assert all(row["fresh_reset_before"] is True and row["fresh_reset_after"] is True for row in report["rows"])
    assert all(row["safe_to_send"] is False for row in report["rows"])
    assert all(row["target_contacted"] is False for row in report["rows"])
    assert all(row["external_network"] is False and row["persistent_storage"] is False for row in report["rows"])
    assert len({row["evidence_sha256"] for row in report["rows"]}) == 120

    by_role = {role: [row for row in report["rows"] if row["role"] == role] for role in ("candidate", "reference", "negative", "replay")}
    assert all(len(rows) == 30 for rows in by_role.values())
    assert all(row["vulnerable_effect"] is True for row in by_role["candidate"])
    assert all(row["vulnerable_effect"] is True for row in by_role["replay"])
    assert all(row["vulnerable_effect"] is False for row in by_role["reference"])
    assert all(row["negative_control_clean"] is True for row in by_role["negative"])


def test_supplemental_canary_report_is_abstract_and_candidate_only() -> None:
    report = local.run()
    serialized = json.dumps(report, ensure_ascii=False).casefold()
    for marker in ("http://", "https://", "payload=", "wire=", "response_body=", "<script"):
        assert marker not in serialized
    assert report["execution"] == {
        "in_process_only": True,
        "docker_started": False,
        "target_contacted": False,
        "external_network": False,
        "wire_created": False,
        "persistent_storage": False,
    }
    assert report["model_boundary"]["context_rows_emitted"] is False
    assert report["model_boundary"]["raw_request_stored"] is False
    assert report["model_boundary"]["raw_response_stored"] is False
    assert report["training_eligible"] == 0
    assert all(value is False for value in report["promotion"].values())


def test_local_replay_rejects_non_enum_extra_values() -> None:
    status, document = local._invoke(
        "POST",
        "/api/canary",
        {"case_ref": "session_guessing", "role": "candidate", "phase": "candidate", "value": "not-retained"},
    )
    assert status == 400
    assert document["status"] == "ask"
    assert document["safe_to_send"] is False

