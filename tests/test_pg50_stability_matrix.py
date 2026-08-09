import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg50_catalog_covers_independent_matrix_with_hard_replay_context():
    catalog = _load("pg50_stability_matrix_catalog_v1.json")
    assert len(catalog["samples"]) == 7200
    assert catalog["typed_positive_count"] == 405
    assert catalog["negative_control_count"] == 6795
    assert catalog["fresh_reset_count"] == 7200
    assert catalog["source_count"] == 3
    assert catalog["target_instance_count"] == 7200
    assert catalog["trace_episode_count"] == 450
    assert catalog["accepted_evaluation_episode_count"] == 450
    assert catalog["methods"] == ["GET", "POST"]
    assert catalog["implementations"] == ["ember", "frost", "quartz"]
    assert catalog["seeds"] == [503, 509, 521, 523, 541]
    assert catalog["surface_variants"] == ["plain", "wrapped", "framed"]
    assert len(catalog["families"]) == 10
    assert catalog["training_artifact_generated"] is False
    assert catalog["raw_probe_strings_stored"] is False
    assert catalog["raw_response_bodies_stored"] is False
    assert all(row["reset"]["fresh_target"] and row["reset"]["completed"] for row in catalog["samples"])
    assert all(row["evidence"]["evidence_hash"] for row in catalog["samples"])
    assert all(row["payload_manifest"]["method"] in {"GET", "POST"} for row in catalog["samples"])
    assert "<script" not in json.dumps(catalog["families"], ensure_ascii=False).casefold()
    assert "union select" not in json.dumps(catalog["families"], ensure_ascii=False).casefold()


def test_pg50_preprobe_policy_is_stable_on_two_unseen_implementations():
    report = _load("pg50_stability_matrix_report_v1.json")
    assert report["status"] == "diagnostic_only"
    assert report["training"]["implementation"] == "ember"
    assert report["training"]["holdout_implementations_used_for_training"] == []
    assert report["training"]["response_projection_consumed_by_policy"] is False
    assert report["model"]["response_projection_consumed_by_policy"] is False
    assert report["train_metrics"]["positive_recall"] == 1.0
    assert report["dev_metrics"]["positive_recall"] == 1.0
    for implementation in ("frost", "quartz"):
        metric = report["implementation_holdouts"][implementation]
        assert metric["episode_count"] == 150
        assert metric["effect_success_rate"] == 1.0
        assert metric["known_family_recall"] == 1.0
        assert metric["unknown_positive_count"] == 15
        assert metric["unknown_safe_abstain_count"] == 15
        assert metric["unknown_strict_abstain"] is True
        assert metric["negative_false_accept_count"] == 0
        assert metric["get_post_covered"] is True
        assert metric["accepted_trace_episode_count"] == 150
    assert report["safe_gate"]["status"] == "passed"
    assert report["safe_gate"]["claim_allowed"] is True
    assert report["formal_capability_claim_allowed"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg50_active_trace_is_projection_only_and_all_holdout_episodes_validate():
    trace = _load("pg50_stability_matrix_active_trace_v1.json")
    assert trace["episode_count"] == 300
    assert trace["accepted_evaluation_episode_count"] == 300
    assert trace["methods"] == ["GET", "POST"]
    assert trace["raw_probe_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
    assert len(trace["implementation_traces"]) == 2
    assert all(step["fresh_reset"]["fresh_target"] for step in trace["steps"])
    assert all(step["online_weight_update"] is False and step["long_term_memory_write"] is False for step in trace["steps"])
    assert any(step["decision"] == "abstain" for step in trace["steps"])
    serialized = json.dumps(trace["steps"][:20], ensure_ascii=False).casefold()
    assert "<script" not in serialized
    assert "onerror" not in serialized
    assert "union select" not in serialized


def test_pg50_protocol_records_safety_gate_without_promotion():
    protocol = _load("pg50_stability_matrix_protocol_v1.json")
    assert protocol["target"]["holdout_implementations"] == ["frost", "quartz"]
    assert protocol["model_input_contract"]["response_projection_consumed_by_policy"] is False
    assert protocol["run_result"]["catalog_rows"] == 7200
    assert protocol["run_result"]["catalog_accepted_episode_count"] == 450
    assert protocol["run_result"]["frost_effect_success_rate"] == 1.0
    assert protocol["run_result"]["quartz_effect_success_rate"] == 1.0
    assert protocol["run_result"]["frost_negative_false_accept_count"] == 0
    assert protocol["run_result"]["quartz_negative_false_accept_count"] == 0
    assert protocol["run_result"]["safe_gate_status"] == "passed"
    assert protocol["run_result"]["formal_capability_claim_allowed"] is False
    assert protocol["run_result"]["training_allowed"] is False
    assert protocol["run_result"]["memory_promotion_allowed"] is False
    assert protocol["status"] == "run_completed_stability_gate_passed_no_promotion"
