import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg41_effect_confirmed_unknown_semantics_fail_closed():
    report = _load("pg41_safe_unknown_router_report_v1.json")
    metrics = report["metrics"]
    assert report["status"] == "diagnostic_only"
    assert metrics["pair_count"] == 480
    assert metrics["known_positive_count"] == 48
    assert metrics["unknown_positive_count"] == 48
    assert metrics["known_family_recall"] == 1.0
    assert metrics["unknown_effect_recall"] == 1.0
    assert metrics["negative_effect_false_accept_count"] == 0
    assert metrics["unknown_misname_count"] == 0
    assert metrics["unknown_not_abstain_count"] == 0
    assert metrics["unknown_strict_abstain"] is True
    assert report["safe_routing_gate"]["claim_allowed"] is True
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg41_protocol_keeps_safe_route_gate_separate_from_model_promotion():
    protocol = _load("pg41_safe_unknown_router_protocol_v1.json")
    assert protocol["model_input_contract"]["long_term_memory_write"] is False
    assert protocol["promotion_gate"]["zero_unknown_misname"] is True
    assert protocol["promotion_gate"]["unknown_surface_strict_abstain"] is True
    assert protocol["run_result"]["get_post_covered"] is True
    assert protocol["run_result"]["safe_routing_gate_status"] == "passed"
    assert protocol["run_result"]["capability_claim_allowed"] is False
    assert protocol["run_result"]["training_allowed"] is False
    assert protocol["status"] == "run_completed_safe_routing_gate_passed_no_promotion"


def test_pg41_report_contains_no_raw_probe_or_response_body():
    report_text = (ROOT / "research" / "pg41_safe_unknown_router_report_v1.json").read_text(encoding="utf-8").casefold()
    assert "<script" not in report_text
    assert "onerror" not in report_text
    assert "union select" not in report_text
    assert "raw_probe_strings_stored\": false" in report_text
    assert "raw_response_bodies_stored\": false" in report_text
