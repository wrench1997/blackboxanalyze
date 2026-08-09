import json
from pathlib import Path

from app.pg221_boolean_oracle import build_boolean_value, evaluate_boolean_effect


ROOT = Path(__file__).resolve().parents[1]


def test_pg221_runtime_boolean_builder_leaves_application_closing_delimiter() -> None:
    true_value = build_boolean_value(truth=True)
    false_value = build_boolean_value(truth=False)
    assert true_value.startswith("kobe'")
    assert false_value.startswith("kobe'")
    assert not true_value.endswith("'")
    assert not false_value.endswith("'")
    assert true_value != false_value


def test_pg221_oracle_is_fail_closed_without_pair_differential() -> None:
    base = {"response_projection": {"row_marker_count": 0, "result_shape": "record_absent"}}
    reset = {
        "fresh_target": True,
        "container_recreated": True,
        "container_restart_used": False,
        "volume_mount_count": 0,
        "database_health_gate": "mysqli_root_pikachu_ok",
    }
    result = evaluate_boolean_effect(
        route={"path": "/vul/sqli/sqli_blind_b.php", "method": "GET"},
        true_candidate=base,
        false_candidate=base,
        true_reference=base,
        false_reference=base,
        negative=base,
        reset=reset,
        source_hash="0" * 64,
    )
    assert result["boolean_effect_confirmed"] is False
    assert result["confirmed_positive"] is False
    assert "candidate_boolean_differential_missing" in result["reasons"]


def test_pg221_report_records_replay_outcome_without_claiming_site_wide_vulnerability() -> None:
    report = json.loads((ROOT / "research" / "pg221_pikachu_boolean_blind_oracle_report_v1.json").read_text(encoding="utf-8-sig"))
    assert report["counts"]["fresh_container_count"] == 2
    assert report["counts"]["boolean_effect_confirmed_count"] == 2
    assert report["counts"]["false_positive_count"] == 0
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert all(row["raw_payload_strings_stored"] is False for row in report["results"])
    assert all(row["raw_response_bodies_stored"] is False for row in report["results"])
