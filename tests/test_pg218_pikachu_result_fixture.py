import json
from pathlib import Path

from app.pg218_pikachu_result_oracle import evaluate_result_fixture, project_result_response


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


class _Response:
    status_code = 200
    content = b"<p class='notice'>your uid:1 email is: fixture</p>"
    text = content.decode()


def test_pg218_result_oracle_requires_positive_and_negative_fixture() -> None:
    route = {"path": "/vul/sqli/sqli_id.php", "method": "POST", "fields": ["id", "submit"]}
    positive = project_result_response(_Response(), route=route, fixture_kind="known_record_id_1_or_user_fixture")
    negative = {"response_projection": {"row_marker_count": 0, "result_shape": "record_absent"}}
    typed = {"typed_effect_confirmed": True}
    reset = {"fresh_target": True, "container_recreated": True, "container_restart_used": False, "volume_mount_count": 0, "database_health_gate": "mysqli_root_pikachu_ok"}
    result = evaluate_result_fixture(route=route, positive=positive, negative=negative, typed_effect=typed, reset=reset)
    assert result["result_fixture_verified"] is True
    assert result["evidence_hash"]
    assert result["vulnerability_claim_allowed"] is False


def test_pg218_report_has_read_only_result_pairs() -> None:
    report = _load("research/pg218_pikachu_result_fixture_report_v1.json")
    assert report["status"] == "completed_read_only_result_fixture_validation"
    counts = report["counts"]
    assert counts["fresh_container_count"] == 14
    assert counts["known_positive_fixture_record_count"] == 12
    assert counts["negative_fixture_clean_count"] == 14
    assert counts["result_fixture_verified_count"] == 8
    assert counts["typed_effect_confirmed_count"] == 8
    assert counts["false_positive_count"] == 0
    assert counts["docker_restart_used_count"] == 0
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert all(row["fixture"]["raw_payload_stored"] is False for row in report["results"])
    assert all(row["fixture"]["raw_response_stored"] is False for row in report["results"])
