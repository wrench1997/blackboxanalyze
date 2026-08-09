from __future__ import annotations

import json

from scripts.run_pg389_js_chain_local_replay import run_local_replay


def test_pg389_local_fixture_replay_has_typed_positive_and_clean_negative() -> None:
    report = run_local_replay()
    assert report["status"] == "completed_local_fixture_typed_diagnostic"
    assert report["counts"] == {
        "cases": 3,
        "roles": 4,
        "rows": 12,
        "fresh_resets": 12,
        "typed_effect": 9,
        "baseline_filtered": 12,
        "failure_action_changed": 12,
        "negative_violation": 0,
        "evidence_rows": 12,
    }
    assert all(item["negative_clean"] for item in report["case_reports"])
    assert all(item["candidate_typed"] and item["reference_typed"] and item["replay_typed"] for item in report["case_reports"])
    assert report["execution"]["local_fixture_contacted"] is True
    assert report["execution"]["target_contacted"] is False
    assert report["execution"]["external_network"] is False


def test_pg389_replay_does_not_persist_fixture_values_or_wires() -> None:
    report = run_local_replay()
    text = json.dumps(report, ensure_ascii=False, sort_keys=True).casefold()
    for marker in ("http://", "https://", "wire=", "payload=", "response_body=", "raw_value=", "pg385_cand", "pg385_ref", "pg385_neg", "pg385_replay"):
        assert marker not in text
    assert all(row["raw_wire_stored"] is False and row["raw_value_stored"] is False for row in report["rows"])
    assert report["training_eligible"] == 0
    assert all(flag is False for flag in report["promotion"].values())
