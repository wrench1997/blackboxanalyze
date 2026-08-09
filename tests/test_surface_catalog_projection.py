from pathlib import Path

from app.research_ops import build_payload_review, build_research_ops_snapshot


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_get_and_post_parameter_channels_are_not_dropped():
    snapshot = build_research_ops_snapshot()
    catalog = snapshot["surface_catalog"]
    assert catalog["counts"]["routes"] >= 70
    assert catalog["counts"]["with_parameter_context"] >= 40
    assert catalog["counts"]["parameterized_response_observed"] == 0

    routes = catalog["routes"]
    reflected_get = next(route for route in routes if route["path"] == "/vul/xss/xss_reflected_get.php")
    assert reflected_get["form_params"] == ["message", "submit"]
    assert reflected_get["has_parameter_context"] is True

    numeric_post = next(route for route in routes if route["path"] == "/vul/sqli/sqli_id.php")
    assert numeric_post["post_form_params"] == ["id", "submit"]
    assert numeric_post["has_parameter_context"] is True


def test_manifest_wire_projection_stays_unverified_and_loopback_only():
    review = build_payload_review()
    manifest_entries = [entry for entry in review["entries"] if str(entry.get("source", "")).startswith("PG-179")]
    assert len(manifest_entries) >= 20
    assert all(entry["validation_status"] == "oracle_gap" for entry in manifest_entries)
    assert all(entry["training_eligible"] is False for entry in manifest_entries)
    assert all("<LOOPBACK_ORIGIN>" in entry["ai"]["request"]["wire"] for entry in manifest_entries)
    assert all(entry["oracle_evidence"]["matched"] is False for entry in manifest_entries)
    assert review["target_scope"]["arbitrary_target_input"] is False
