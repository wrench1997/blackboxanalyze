import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg48_catalog_has_real_dual_channel_replay_context():
    catalog = _load("pg48_compositional_preprobe_catalog_v1.json")
    samples = catalog["samples"]
    assert len(samples) == 1536
    assert sum(bool(row["oracle_projection"].get("positive", False)) for row in samples) == 84
    assert sum(row["sample_role"] == "negative_control" for row in samples) == 1452
    assert catalog["runtime_replay"] is True
    assert catalog["independent_target_implementation"] is True
    assert catalog["methods"] == ["GET", "POST"]
    assert len(catalog["sources"]) == 2
    assert catalog["fresh_reset_count"] == 1536
    assert catalog["trace_episode_count"] == 96
    assert catalog["accepted_evaluation_episode_count"] == 96
    assert catalog["training_artifact_generated"] is False
    serialized = json.dumps(catalog, ensure_ascii=False).casefold()
    assert "<script" not in serialized
    assert "onerror" not in serialized
    assert "union select" not in serialized
    assert catalog["raw_probe_strings_stored"] is False
    assert catalog["raw_response_bodies_stored"] is False


def test_pg48_preprobe_head_selects_method_on_frost_without_response_features():
    report = _load("pg48_compositional_preprobe_report_v1.json")
    holdout = report["frost_holdout"]
    assert report["status"] == "diagnostic_only"
    assert report["training"]["pg48_frost_used_for_training"] is False
    assert report["training"]["response_projection_consumed_by_policy"] is False
    assert report["model"]["response_projection_consumed_by_policy"] is False
    assert report["train_metrics"]["effect_recall"] == 1.0
    assert report["seed_holdout_metrics"]["effect_recall"] == 1.0
    assert holdout["episode_count"] == 48
    assert holdout["effect_success_rate"] == 1.0
    assert holdout["known_family_recall"] == 1.0
    assert holdout["unknown_positive_count"] == 6
    assert holdout["unknown_safe_abstain_count"] == 6
    assert holdout["unknown_strict_abstain"] is True
    assert holdout["negative_false_accept_count"] == 0
    assert holdout["mean_queries"] == 3.125
    assert holdout["get_post_covered"] is True
    assert holdout["accepted_trace_episode_count"] == 48
    assert report["safe_gate"]["status"] == "passed"
    assert report["safe_gate"]["claim_allowed"] is True
    assert report["formal_capability_claim_allowed"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg48_trace_is_fresh_projection_only_and_abstains_on_unbound_slot():
    trace = _load("pg48_compositional_preprobe_active_trace_v1.json")
    assert trace["episode_count"] == 48
    assert trace["accepted_evaluation_episode_count"] == 48
    assert len(trace["steps"]) == 150
    assert trace["methods"] == ["GET", "POST"]
    assert all(step["fresh_reset"]["fresh_target"] for step in trace["steps"])
    assert all(step["online_weight_update"] is False and step["long_term_memory_write"] is False for step in trace["steps"])
    assert any(step["decision"] == "abstain" for step in trace["steps"])
    serialized = json.dumps(trace, ensure_ascii=False).casefold()
    assert "<script" not in serialized
    assert "onerror" not in serialized
    assert "union select" not in serialized
    assert trace["raw_probe_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False


def test_pg48_protocol_blocks_promotion_after_safety_gate():
    protocol = _load("pg48_compositional_preprobe_protocol_v1.json")
    assert protocol["slot_contract"]["excluded"] == [
        "typed_oracle",
        "positive",
        "family",
        "response_projection",
        "raw_probe",
        "raw_response_body",
        "evidence_sha256",
    ]
    assert protocol["run_result"]["safe_gate_status"] == "passed"
    assert protocol["run_result"]["formal_capability_claim_allowed"] is False
    assert protocol["run_result"]["training_allowed"] is False
    assert protocol["run_result"]["memory_promotion_allowed"] is False
    assert protocol["status"] == "run_completed_preprobe_safety_gate_passed_no_promotion"
