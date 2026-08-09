import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def test_pg210_ai_and_reference_probes_are_sent_and_effects_agree() -> None:
    report = _load("research/pg210_ai_reference_payload_validation_report_v1.json")
    assert report["status"] == "completed_ai_reference_local_payload_validation"
    assert report["model"]["base_parameter_count"] > 100_000_000
    counts = report["counts"]
    assert counts["fresh_container_count"] == 2
    assert counts["episode_count"] == 12
    assert counts["ai_candidate_sent_count"] == 12
    assert counts["ai_surface_effect_count"] == 4
    assert counts["reference_surface_effect_count"] == 4
    assert counts["ai_reference_effect_agreement_count"] == 12
    assert counts["ai_payload_hash_bound_count"] == 12
    assert counts["reference_payload_hash_bound_count"] == 12
    assert counts["false_positive_count"] == 0
    assert all(row["ai"]["sent"] for row in report["episodes"])
    assert all(row["ai_reference_effect_agreement"] for row in report["episodes"])


def test_pg210_request_anatomy_view_explains_send_without_raw_payloads() -> None:
    report = _load("research/pg210_ai_reference_payload_validation_report_v1.json")
    view = _load("research/pg210_request_anatomy_view_v1.json")
    protocol = _load("research/pg210_ai_reference_payload_validation_protocol_v1.json")
    rules = _load("research/improvement_rules.json")
    serialized = json.dumps(report, ensure_ascii=False) + json.dumps(view, ensure_ascii=False)
    assert "<script" not in serialized.casefold()
    assert "response_body" not in serialized
    assert view["raw_payload_strings_stored"] is False
    assert view["raw_response_bodies_stored"] is False
    assert all(row["ai_request"]["method"] == "GET" for row in view["rows"])
    assert all(row["ai_request"]["placement"] == "query" for row in view["rows"])
    assert all(row["ai_request"]["binding_sha256"] for row in view["rows"] if row["ai_sent"])
    assert protocol["ai_participates_in_send"] is True
    assert protocol["reference_probe_independent"] is True
    assert protocol["sql_oracle"] == "unavailable; abstain"
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert rules["pg210_ai_reference_payload_validation"]["ai_candidate_sent_count"] == 12
    assert rules["pg210_ai_reference_payload_validation"]["reference_effect_agreement_count"] == 12

