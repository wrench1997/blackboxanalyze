import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "research" / "pg191_pikachu_surface_matrix_large_report_v1.json"
TRACE = ROOT / "research" / "pg191_pikachu_surface_matrix_large_trace_v1.json"
PROTOCOL = ROOT / "research" / "pg191_pikachu_surface_matrix_large_protocol_v1.json"
MANIFEST = ROOT / "research" / "pg191_pikachu_surface_matrix_manifest_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def test_pg191_uses_full_browser_crawl_matrix_and_unseen_get_post_routes() -> None:
    report = _load(REPORT)
    protocol = _load(PROTOCOL)
    manifest = _load(MANIFEST)
    assert report["status"] == "completed_crawled_surface_matrix_and_large_replay"
    assert report["source"]["parameterized_surface_count"] == 44
    assert report["source"]["selected_route_count"] == 8
    assert protocol["parameterized_surface_count"] == 44
    assert len(protocol["selected_routes"]) == 8
    assert {row["method"] for row in protocol["selected_routes"]} == {"GET", "POST"}
    assert manifest["parameterized_surface_count"] == 44
    assert len(manifest["surfaces"]) == 44
    assert manifest["raw_values_stored"] is False
    assert manifest["raw_payloads_stored"] is False


def test_pg191_selects_101m_variant_and_sends_bounded_get_post_canaries() -> None:
    report = _load(REPORT)
    assert report["selection"]["selected_variant"] == "xxl"
    assert report["counts"]["sent_get_count"] == 15
    assert report["counts"]["sent_post_count"] == 4
    assert report["counts"]["candidate_sent_count"] == 3
    assert report["counts"]["manifest_validation_failure_count"] == 0
    assert report["counts"]["typed_positive_count"] == 0
    assert report["selection"]["vulnerability_claim_allowed"] is False
    assert report["selection"]["training_artifact_promotion_allowed"] is False


def test_pg191_large_and_xxl_gate_holdout_is_fail_closed() -> None:
    report = _load(REPORT)
    for result in report["model_variants"]:
        assert result["holdout"]["expected_abstain_count"] > 0
        assert result["holdout"]["abstain_recall"] >= 0.95
        assert result["holdout"]["unsafe_allow_count"] == 0
        assert result["parameter_count"] > 19_000_000
    assert report["model_variants"][1]["parameter_count"] > 100_000_000


def test_pg191_every_sent_route_is_fresh_and_keeps_raw_values_out() -> None:
    report = _load(REPORT)
    trace = _load(TRACE)
    protocol = _load(PROTOCOL)
    assert len(report["fresh_targets"]) == 8
    assert all(item["fresh_container"] for item in report["fresh_targets"])
    for run in report["runs"]:
        assert run["fresh_container"] is True
        assert run["manifest_validation_failure_count"] == 0
        assert run["target_instance_hash"]
        for step in run["steps"]:
            if step.get("action_manifest"):
                assert step["action_manifest"]["payload_sha256"]
                assert step["action_manifest"]["manifest_sha256"]
                assert step["action_manifest"]["safety"]["does_not_execute"] is True
            assert step.get("long_term_memory_write", False) is False
    assert trace["training_eligible"] is False
    assert trace["raw_payload_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
    assert protocol["manifest_validator_before_send"] is True
    assert protocol["typed_oracle_required_before_positive"] is True


def test_pg191_rule_preserves_crawl_authority_and_blocks_unverified_payload_claim() -> None:
    rules = _load(ROOT / "research" / "improvement_rules.json")
    rule = rules["pg191_pikachu_surface_matrix_large_replay"]
    assert rule["parameterized_surface_count"] == 44
    assert rule["selected_unseen_route_count"] == 8
    assert rule["selected_variant"] == "xxl"
    assert rule["browser_crawl_parameter_authority"] is True
    assert rule["model_input_route_fields_forbidden"] is True
    assert rule["typed_oracle_required_before_positive"] is True
    assert rule["raw_payload_strings_stored"] is False
    assert rule["vulnerability_claim_allowed"] is False
