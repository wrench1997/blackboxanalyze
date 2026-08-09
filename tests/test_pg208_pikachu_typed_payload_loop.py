import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def test_pg208_ai_sends_crawl_grounded_xss_candidates_and_replays_them() -> None:
    report = _load("research/pg208_pikachu_typed_payload_loop_report_v1.json")
    assert report["status"] == "completed_crawl_parameter_grounding_and_typed_loop"
    assert report["model"]["base_parameter_count"] > 100_000_000
    assert report["crawl"]["source_request_surface_count"] == 112
    assert report["crawl"]["unique_route_entry_count"] == 112
    assert report["crawl"]["active_replay_eligible_count"] == 15
    counts = report["counts"]
    assert counts["fresh_container_count"] == 2
    assert counts["route_replay_count"] == 26
    assert counts["get_route_count"] == 22
    assert counts["post_route_count"] == 4
    assert counts["candidate_generated_count"] == 26
    assert counts["candidate_send_count"] == 12
    assert counts["get_candidate_send_count"] == 12
    assert counts["post_candidate_send_count"] == 0
    assert counts["typed_dom_candidate_send_count"] == 12
    assert counts["dual_oracle_agreement_count"] == 8
    assert counts["fresh_replay_effect_count"] == 12
    assert counts["unknown_oracle_abstain_count"] == 14
    assert counts["false_positive_count"] == 0
    assert all(row["candidate_sent"] is False or row["method"] == "GET" for row in report["route_runs"])
    assert all(row["raw_payload_strings_stored"] is False for row in report["route_runs"])
    assert all(row["raw_response_bodies_stored"] is False for row in report["route_runs"])


def test_pg208_unknown_sql_lane_and_promotion_are_quarantined() -> None:
    report = _load("research/pg208_pikachu_typed_payload_loop_report_v1.json")
    protocol = _load("research/pg208_pikachu_typed_payload_loop_protocol_v1.json")
    catalog = _load("research/pg208_pikachu_parameter_catalog_v1.json")
    rules = _load("research/improvement_rules.json")
    serialized = json.dumps(report, ensure_ascii=False)
    assert "<span" not in serialized
    assert "response_body" not in serialized
    assert report["promotion"]["training_eligible"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert protocol["parameter_authority"].startswith("PG-179")
    assert protocol["unknown_sql_oracle_action"] == "abstain"
    assert protocol["raw_payload_and_response_excluded"] is True
    assert catalog["raw_request_values_stored"] is False
    assert catalog["raw_response_bodies_stored"] is False
    assert rules["pg208_pikachu_typed_payload_loop"]["candidate_send_count"] == 12
    assert rules["pg208_pikachu_typed_payload_loop"]["sql_unknown_oracle_abstain"] is True

