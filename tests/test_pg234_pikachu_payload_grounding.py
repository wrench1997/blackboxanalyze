import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pg234_wire_catalog_is_ai_selected_and_runtime_grounded() -> None:
    report = json.loads((ROOT / "research" / "pg234_pikachu_payload_grounding_report_v1.json").read_text(encoding="utf-8-sig"))
    catalog = json.loads((ROOT / "research" / "pg234_pikachu_payload_wire_catalog_v1.json").read_text(encoding="utf-8-sig"))
    assert report["status"] == "completed_pikachu_ai_wire_shape_grounding_report"
    assert report["counts"] == {
        "wire_row_count": 22,
        "sql_row_count": 8,
        "dom_redirect_row_count": 14,
        "get_row_count": 20,
        "post_row_count": 2,
        "ai_candidate_sent_count": 22,
        "typed_sql_result_confirmed_count": 8,
        "dom_surface_effect_confirmed_count": 4,
        "redirect_effect_count": 0,
        "false_positive_count": 0,
    }
    assert catalog["contract"]["ai_candidate_reference_negative_comparison"] is True
    assert catalog["contract"]["runtime_values_not_retained"] is True
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert report["safety"]["raw_payload_strings_stored"] is False
    assert report["safety"]["raw_response_bodies_stored"] is False
    assert all("<LOOPBACK_ORIGIN>" in row["ai_wire_placeholder"] for row in catalog["rows"])
    assert all("<RUNTIME_" in row["ai_wire_placeholder"] for row in catalog["rows"])
    assert all(row["vulnerability_claim_allowed"] is False for row in catalog["rows"])
    assert all(row["raw_payload_strings_stored"] is False for row in catalog["rows"])

