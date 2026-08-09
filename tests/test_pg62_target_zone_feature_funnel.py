import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg62_feature_funnel_keeps_only_stable_pre_oracle_feature():
    report = _read("pg62_target_zone_feature_funnel_report_v1.json")
    contract = report["input_contract_audit"]
    funnel = report["funnel"]
    assert report["status"] == "diagnostic_only"
    assert contract["feature_function_present"] is True
    assert contract["oracle_is_label_not_feature"] is True
    assert contract["target_method_is_label_not_feature"] is True
    assert contract["layout_id_is_not_feature"] is True
    assert contract["raw_request_response_is_not_feature"] is True
    assert funnel["accepted_features"] == ["channel_hint"]
    rejected = {row["feature"] for row in funnel["rows"] if row["funnel_decision"] == "reject"}
    assert rejected == {"surface_class", "response_shape", "route_depth", "parameter_count_bucket"}
    channel = next(row for row in funnel["rows"] if row["feature"] == "channel_hint")
    assert channel["stable_utility_across_layouts_and_seeds"] is True
    assert channel["mean_layout_utility_drop"] > 0.5


def test_pg62_baseline_safety_is_separate_from_permutation_diagnostics():
    report = _read("pg62_target_zone_feature_funnel_report_v1.json")
    baseline = report["baseline_holdout"]
    gate = report["hard_gate"]
    assert baseline["target_success_rate"] == 1.0
    assert baseline["negative_false_accept_count"] == 0
    assert baseline["unknown_strict_abstain"] is True
    assert baseline["selected_action_entropy"] >= 0.5
    assert gate["status"] == "passed"
    assert gate["claim_allowed"] is False
    assert gate["training_allowed"] is False
    assert gate["memory_promotion_allowed"] is False
    permuted = report["permutation_ablation"]["channel_hint"]["global"]
    assert permuted["target_success_rate"] < baseline["target_success_rate"]
    assert permuted["negative_false_accept_count"] > 0


def test_pg62_protocol_requires_non_resonant_seed_stability_audit():
    protocol = _read("pg62_target_zone_feature_funnel_protocol_v1.json")
    assert protocol["authorized_scope"]["target_host"] == "127.0.0.1"
    assert protocol["authorized_scope"]["external_network"] is False
    assert protocol["required_stages"]["seed_bucket_modulus"] == 3
    assert protocol["required_stages"]["codex_review_required"] is True
    assert protocol["input_contract"]["pre_oracle_only"] is True
    assert protocol["input_contract"]["oracle_is_label_only"] is True
    assert protocol["run_result"]["accepted_features"] == ["channel_hint"]
    text = json.dumps(protocol, ensure_ascii=False).casefold()
    assert "<script" not in text
    assert "union select" not in text
    assert protocol["authorized_scope"]["raw_probe_persistence"] is False
    assert protocol["authorized_scope"]["raw_response_body_persistence"] is False
