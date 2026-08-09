import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pg226_ai_sql_validation_has_get_post_typed_and_result_pairs() -> None:
    report = json.loads((ROOT / "research" / "pg226_ai_sql_payload_validation_report_v1.json").read_text(encoding="utf-8-sig"))
    dataset = json.loads((ROOT / "research" / "pg226_ai_sql_payload_validation_dataset_v1.json").read_text(encoding="utf-8-sig"))
    counts = report["counts"]
    assert report["status"] == "completed_ai_selected_sql_typed_result_validation"
    assert counts["fresh_container_count"] == 8
    assert counts["get_episode_count"] == 6
    assert counts["post_episode_count"] == 2
    assert counts["ai_candidate_send_count"] == counts["reference_send_count"] == counts["negative_send_count"] == 8
    assert counts["typed_effect_confirmed_count"] == 8
    assert counts["result_fixture_verified_count"] == 8
    assert counts["training_candidate_count"] == 8
    assert counts["false_positive_count"] == 0
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert all("<RUNTIME_SQL_SHAPE>" in row["ai"]["wire_placeholder"] for row in report["results"])
    assert all(row["raw_payload_strings_stored"] is False for row in dataset["rows"])
    assert all(row["raw_response_bodies_stored"] is False for row in dataset["rows"])

