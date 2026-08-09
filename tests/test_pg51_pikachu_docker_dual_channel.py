import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg51_real_docker_catalog_is_dual_channel_fresh_and_evaluation_only():
    catalog = _load("pg51_pikachu_docker_dual_channel_catalog_v1.json")
    assert len(catalog["samples"]) == 28
    assert catalog["methods"] == ["GET", "POST"]
    assert catalog["typed_positive_count"] == 0
    assert catalog["negative_control_count"] == 28
    assert catalog["fresh_reset_count"] == 28
    assert catalog["source_count"] == 1
    assert catalog["target_instance_count"] == 2
    assert catalog["trace_episode_count"] == 7
    assert catalog["accepted_evaluation_episode_count"] == 7
    assert catalog["training_eligible"] is False
    assert catalog["training_artifact_generated"] is False
    assert catalog["raw_probe_strings_stored"] is False
    assert catalog["raw_response_bodies_stored"] is False
    assert catalog["sources"][0]["container_image_digest"].startswith("sha256:")
    assert all(row["reset"]["fresh_target"] and row["reset"]["completed"] for row in catalog["samples"])
    assert all(row["decision"]["training_action"] == "abstain" for row in catalog["samples"])


def test_pg51_trace_abstains_without_execution_or_ast_or_redirect_oracle():
    trace = _load("pg51_pikachu_docker_dual_channel_trace_v1.json")
    assert trace["episode_count"] == 7
    assert trace["accepted_evaluation_episode_count"] == 7
    assert trace["methods"] == ["GET", "POST"]
    assert len(trace["steps"]) == 14
    assert all(step["decision"] == "abstain" for step in trace["steps"])
    assert all(step["fresh_reset"]["fresh_target"] for step in trace["steps"])
    assert all(step["online_weight_update"] is False and step["long_term_memory_write"] is False for step in trace["steps"])
    serialized = json.dumps(trace, ensure_ascii=False).casefold()
    assert "<script" not in serialized
    assert "union select" not in serialized
    assert "sleep(" not in serialized
    assert trace["raw_probe_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False


def test_pg51_report_keeps_real_docker_signal_separate_from_vulnerability_claim():
    report = _load("pg51_pikachu_docker_dual_channel_report_v1.json")
    assert report["status"] == "diagnostic_only"
    assert report["target"]["loopback_only"] is True
    assert report["target"]["external_network"] is False
    assert report["catalog"]["get_post_covered"] is True
    assert report["oracle_contract"]["execution_oracle_available"] is False
    assert report["oracle_contract"]["sql_ast_oracle_available"] is False
    assert report["oracle_contract"]["external_redirect_oracle_available"] is False
    assert report["oracle_contract"]["vulnerability_claim_allowed"] is False
    assert report["trace"]["abstain_count"] == 14
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["formal_capability_claim_allowed"] is False


def test_pg51_protocol_freezes_no_promotion_contract():
    protocol = _load("pg51_pikachu_docker_dual_channel_protocol_v1.json")
    assert protocol["target_contract"]["methods"] == ["GET", "POST"]
    assert protocol["oracle_contract"]["surface_signal_is_not_vulnerability_confirmation"] is True
    assert protocol["model_contract"]["required_decision_without_authoritative_oracle"] == "abstain"
    assert protocol["run_result"]["row_count"] == 28
    assert protocol["run_result"]["typed_positive_count"] == 0
    assert protocol["run_result"]["abstain_step_count"] == 14
    assert protocol["run_result"]["vulnerability_claim_allowed"] is False
    assert protocol["run_result"]["training_allowed"] is False
    assert protocol["run_result"]["memory_promotion_allowed"] is False
    assert protocol["status"] == "run_completed_real_docker_shadow_abstain_no_promotion"
