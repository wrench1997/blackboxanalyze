import json
from pathlib import Path

from app.pg217_pikachu_typed_sql_oracle import evaluate_pikachu_sql_effect


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def _projection(*, body_bucket: str, sql_error: bool) -> dict:
    return {
        "response_projection": {"backend_state": "backend_response_observed", "status_class": "2xx", "body_length_bucket": body_bucket, "shape": {"kind": "html"}},
        "signal": {"sql_error_shape": sql_error, "marker_reflected": False},
    }


def test_pg217_oracle_requires_negative_and_reference_agreement() -> None:
    route = {"path": "/vul/sqli/sqli_str.php", "method": "GET", "fields": ["name", "submit"]}
    baseline = _projection(body_bucket="4096-65535", sql_error=False)
    negative = _projection(body_bucket="4096-65535", sql_error=False)
    candidate = _projection(body_bucket="1-255", sql_error=True)
    reset = {"reset_id": "test", "fresh_target": True, "container_recreated": True, "container_restart_used": False, "volume_mount_count": 0, "state_change_allowed": False, "database_health_gate": "mysqli_root_pikachu_ok"}
    positive = evaluate_pikachu_sql_effect(route, baseline=baseline, negative=negative, candidate=candidate, reference=candidate, reset=reset, source_hash="0" * 64)
    assert positive["typed_effect_confirmed"] is True
    assert positive["confirmed_positive"] is True
    assert positive["vulnerability_claim_allowed"] is False
    negative_bad = evaluate_pikachu_sql_effect(route, baseline=baseline, negative=candidate, candidate=candidate, reference=candidate, reset=reset, source_hash="0" * 64)
    assert negative_bad["confirmed_positive"] is False
    assert "negative_control_has_sql_error_shape" in negative_bad["reasons"]


def test_pg217_report_has_real_ai_reference_and_route_abstention() -> None:
    report = _load("research/pg217_pikachu_typed_sql_oracle_report_v1.json")
    counts = report["counts"]
    assert report["status"] == "completed_local_typed_effect_oracle"
    assert counts["fresh_container_count"] == 14
    assert counts["get_episode_count"] == 10
    assert counts["post_episode_count"] == 4
    assert counts["database_health_gate_count"] == 14
    assert counts["negative_send_count"] == 14
    assert counts["ai_candidate_send_count"] == 14
    assert counts["reference_send_count"] == 14
    assert counts["typed_effect_confirmed_count"] == 8
    assert counts["confirmed_positive_count"] == 8
    assert counts["abstain_count"] == 6
    assert counts["false_positive_count"] == 0
    assert counts["docker_restart_used_count"] == 0
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert all(row["ai"]["raw_payload_stored"] is False for row in report["results"])
    assert all(row["reference"]["raw_payload_stored"] is False for row in report["results"])
    assert all(row["typed_oracle"]["evidence_hash"] for row in report["results"])
