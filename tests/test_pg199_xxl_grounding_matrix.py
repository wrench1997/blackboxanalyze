import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8-sig"))


def test_pg199_101m_xxl_actually_drives_grounding_matrix() -> None:
    report = _load("research/pg199_xxl_grounding_matrix_report_v1.json")
    assert report["status"] == "completed_101m_xxl_crawl_surface_grounding"
    assert report["device"] == "cuda"
    assert report["model"]["base_parameter_count"] > 100_000_000
    assert report["crawl_source"]["persisted_request_surface_count"] == 112
    assert report["crawl_source"]["selected_safe_surface_count"] == 15
    assert report["crawl_source"]["candidate_plan_route_count"] == 15
    assert report["crawl_source"]["candidate_plan_candidate_count"] == 63
    assert report["counts"]["route_replay_count"] == 30
    assert report["counts"]["xxl_candidate_allow_count"] == 12
    assert report["counts"]["ai_candidate_send_count"] == 12
    assert report["counts"]["grounded_payload_hash_match_count"] == 12
    assert report["counts"]["method_binding_match_count"] == 12
    assert report["counts"]["typed_dom_dual_agreement_count"] == 8
    assert report["counts"]["unknown_oracle_abstain_count"] == 18
    assert report["counts"]["false_positive_count"] == 0


def test_pg199_route_coverage_preserves_exclusions_and_raw_quarantine() -> None:
    report = _load("research/pg199_xxl_grounding_matrix_report_v1.json")
    protocol = _load("research/pg199_xxl_grounding_matrix_protocol_v1.json")
    serialized = json.dumps(report, ensure_ascii=False)
    assert len(report["route_inventory"]["selected"]) == 15
    assert len(report["route_inventory"]["excluded"]) == 29
    assert len(report["candidate_plan"]) == 15
    assert all(item["candidates"] for item in report["candidate_plan"])
    assert all(item["raw_probe_strings_stored"] is False for item in report["candidate_plan"])
    assert report["crawl_source"]["incomplete_source_surfaces_remain"] is True
    assert report["promotion"]["training_eligible"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert report["safety"]["raw_payload_strings_stored"] is False
    assert report["safety"]["raw_response_bodies_stored"] is False
    assert "<span" not in serialized
    assert "response_body" not in serialized
    assert protocol["model"] == "PG-197 101M XXL risk-aware decoder"
    assert protocol["raw_payload_and_response_excluded"] is True


def test_pg199_every_sent_candidate_has_method_and_payload_evidence() -> None:
    report = _load("research/pg199_xxl_grounding_matrix_report_v1.json")
    sent = [row for row in report["route_runs"] if row["candidate_sent"]]
    assert len(sent) == 12
    for row in sent:
        result = row["candidate_result"]
        assert result["candidate"]["payload_sha256"] == result["evidence"]["payload_sha256"]
        assert result["binding"]["method"] == result["candidate"]["method"] == row["method"]
        assert result["binding"]["path"] == result["candidate"]["path"] == row["path"]
        assert result["raw_probe_stored"] is False
        assert result["raw_response_stored"] is False
        assert result["promotion"]["training_eligible"] is False


def test_pg199_model_gate_abstains_unknown_oracle_routes() -> None:
    report = _load("research/pg199_xxl_grounding_matrix_report_v1.json")
    unknown = [row for row in report["route_runs"] if not row["model_decision"]["state"]["typed_available"]]
    assert len(unknown) == 18
    assert all(row["candidate_sent"] is False for row in unknown)
    assert all(row["abstain_reason"] == "xxl_model_or_candidate_gate_abstain" for row in unknown)


def test_pg199_rule_is_registered() -> None:
    rules = _load("research/improvement_rules.json")
    rule = rules["pg199_xxl_grounding_matrix"]
    assert rule["base_parameter_count"] > 100_000_000
    assert rule["route_replay_count"] == 30
    assert rule["ai_candidate_send_count"] == 12
    assert rule["grounded_payload_hash_match_count"] == 12
    assert rule["training_promotion_allowed"] is False
    assert rule["memory_promotion_allowed"] is False
