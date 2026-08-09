import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pg227_keeps_dom_effect_separate_from_xss_and_open_redirect() -> None:
    report = json.loads((ROOT / "research" / "pg227_ai_dom_redirect_validation_report_v1.json").read_text(encoding="utf-8-sig"))
    dataset = json.loads((ROOT / "research" / "pg227_ai_dom_redirect_validation_dataset_v1.json").read_text(encoding="utf-8-sig"))
    counts = report["counts"]
    assert report["status"] == "completed_ai_selected_dom_redirect_surface_validation"
    assert counts["fresh_container_count"] == 14
    assert counts["route_count"] == 7
    assert counts["ai_candidate_send_count"] == counts["reference_send_count"] == counts["negative_send_count"] == 14
    assert counts["xss_positive_count"] == 0
    assert counts["open_redirect_positive_count"] == 0
    assert counts["false_positive_count"] == 0
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    for row in report["results"]:
        oracle = row["oracle"]
        if row["dom_surface_effect_confirmed"]:
            assert oracle["candidate_reference_agreement"] is True
            assert oracle["negative_clean"] is True
            assert oracle["script_execution"] is False
        assert row["xss_positive"] is False
        assert row["open_redirect_positive"] is False
        assert row["raw_payload_strings_stored"] is False
        assert row["raw_response_bodies_stored"] is False
    assert all(row["raw_payload_strings_stored"] is False for row in dataset["rows"])
    assert all(row["raw_response_bodies_stored"] is False for row in dataset["rows"])

