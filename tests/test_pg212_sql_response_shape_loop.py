import json
from pathlib import Path

import httpx

from app.pg212_sql_response_oracle import build_sql_probe_values, compare_sql_shapes, project_sql_response


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def test_pg212_runtime_sql_probe_is_bounded_and_non_timing() -> None:
    values = build_sql_probe_values(field_names=["name", "submit"], marker="pg212-test", probe_class="syntax_shape")
    assert values["submit"] == "submit"
    assert values["name"].startswith("pg212-test")
    assert "sleep" not in values["name"].casefold()
    assert "benchmark" not in values["name"].casefold()


def test_pg212_shape_difference_is_not_a_vulnerability_claim() -> None:
    control = {"response_projection": {"status_class": "2xx", "body_length_bucket": "4096-65535", "shape": {"kind": "html"}, "backend_state": "backend_response_observed"}, "signal": {"sql_error_shape": False}}
    candidate = {"response_projection": {"status_class": "5xx", "body_length_bucket": "1-255", "shape": {"kind": "html"}, "backend_state": "backend_response_observed"}, "signal": {"sql_error_shape": True}}
    result = compare_sql_shapes(control, candidate)
    assert result["response_shape_differential"] is True
    assert result["positive"] is False
    assert result["vulnerability_claim_allowed"] is False


def test_pg212_does_not_misclassify_a_normal_help_text_reference() -> None:
    response = httpx.Response(
        200,
        request=httpx.Request("GET", "http://127.0.0.1/"),
        content="根据实际情况修改 inc/config.inc.php 里面的数据库连接配置;",
    )
    projected = project_sql_response(response, marker="pg212-help", baseline_status=200)
    assert projected["oracle"]["backend_state"] == "backend_response_observed"
    assert projected["oracle"]["typed_available"] is True


def test_pg212_report_keeps_pikachu_sql_abstain_status() -> None:
    report = _load("research/pg212_pikachu_sql_response_shape_loop_report_v1.json")
    assert report["status"] == "completed_sql_backend_unavailable_abstain"
    assert report["counts"]["fresh_container_count"] == 14
    assert report["counts"]["episode_count"] == 14
    assert report["counts"]["get_episode_count"] == 10
    assert report["counts"]["post_episode_count"] == 4
    assert report["counts"]["reference_send_count"] == 14
    assert report["counts"]["database_unavailable_count"] == 14
    assert report["counts"]["ai_candidate_send_count"] == 0
    assert report["counts"]["abstain_count"] == 14
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert report["safety"]["fresh_container_per_episode"] is True
    assert report["safety"]["no_volume_or_bind_mount"] is True
    assert all(row["database_clean_reset_verified"] for row in report["episodes"])
    assert all(row["reset"]["container_restart_used"] is False for row in report["episodes"])
    assert all(row["reset"]["volume_mount_count"] == 0 for row in report["episodes"])
