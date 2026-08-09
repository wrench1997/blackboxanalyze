import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def test_pg214_replays_get_post_with_clean_database_gates() -> None:
    report = _load("research/pg214_pikachu_fixed_sql_loop_report_v1.json")
    counts = report["counts"]
    assert report["status"] == "completed_backend_response_shape_evaluator_only"
    assert counts["fresh_container_count"] == 14
    assert counts["episode_count"] == 14
    assert counts["get_episode_count"] == 10
    assert counts["post_episode_count"] == 4
    assert counts["database_health_gate_count"] == 14
    assert counts["database_clean_reset_verified_count"] == 14
    assert counts["database_unavailable_count"] == 0
    assert counts["sql_evaluator_typed_available_count"] == 14
    assert counts["ai_candidate_send_count"] == 14
    assert counts["reference_send_count"] == 14
    assert counts["ai_reference_shape_agreement_count"] == 14
    assert counts["confirmed_positive_count"] == 0
    assert counts["false_positive_count"] == 0
    assert report["safety"]["fresh_container_per_episode"] is True
    assert report["safety"]["no_volume_or_bind_mount"] is True
    assert report["safety"]["docker_restart_used_count"] == 0
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert all(row["database_clean_reset_verified"] for row in report["episodes"])
    assert all(row["reset"]["container_restart_used"] is False for row in report["episodes"])
    assert all(row["reset"]["volume_mount_count"] == 0 for row in report["episodes"])
    assert all(row["reset"]["database_health_gate"] == "mysqli_root_pikachu_ok" for row in report["episodes"])


def test_pg214_artifacts_keep_raw_payload_and_response_out() -> None:
    report = _load("research/pg214_pikachu_fixed_sql_loop_report_v1.json")
    trace = _load("research/pg214_pikachu_fixed_sql_loop_trace_v1.json")
    assert report["promotion"]["raw_payload_strings_stored"] is False
    assert report["promotion"]["raw_response_bodies_stored"] is False
    assert trace["raw_payload_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
    assert all(row["ai"]["raw_payload_stored"] is False for row in report["episodes"])
    assert all(row["reference"]["raw_payload_stored"] is False for row in report["episodes"])
