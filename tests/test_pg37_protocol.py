import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg37_protocol_freezes_counterfactual_split_and_model_input_boundary():
    protocol = _load("pg37_counterfactual_protocol_v1.json")
    assert protocol["scope"]["methods"] == ["GET", "POST"]
    assert protocol["scope"]["surface_variants"] == ["compact", "nested", "headerized"]
    assert protocol["fixture"]["planned_sample_count"] == 2880
    assert protocol["fixture"]["planned_typed_positive_count"] == 288
    assert protocol["counterfactual_contract"]["same_family_across_variants"] is True
    assert protocol["counterfactual_contract"]["surface_variant_is_not_family_label"] is True
    assert protocol["model_input_contract"]["typed_oracle_is_label_not_feature"] is True
    assert "surface_variant_label" in protocol["model_input_contract"]["excluded"]
    assert protocol["promotion_gate"]["surface_holdout_required"] is True
    assert protocol["promotion_gate"]["family_holdout_required"] is True
    result = protocol["collection_result"]
    assert result["sample_count"] == 2880
    assert result["typed_positive_count"] == 288
    assert result["fresh_reset_count"] == 2880
    assert result["max_workers"] == 8
    assert result["raw_probe_strings_stored"] is False
    assert protocol["status"] == "ablation_completed_capability_gate_failed_no_promotion"
    ablation = protocol["ablation_result"]
    assert ablation["training_count_per_ablation"] == 256
    assert ablation["results"]["counterfactual_paired"]["pair_agreement"] == 0.625
    assert ablation["results"]["counterfactual_paired"]["family_holdout_typed_recall"] == 0.0
    assert ablation["all_capability_gates"] == "blocked"
    assert ablation["capability_claim_allowed"] is False
    assert ablation["training_allowed"] is False


def test_pg37_improvement_rule_requires_pair_ablation_and_blocks_failed_promotion():
    rules = _load("improvement_rules.json")
    policy = rules["pg37_counterfactual_representation_policy"]
    assert policy["same_family_multi_surface_required"] is True
    assert policy["typed_oracle_is_label_not_feature"] is True
    assert policy["surface_variant_label_excluded_from_model"] is True
    assert policy["required_ablations"] == ["surface_only", "counterfactual_paired", "phase_only"]
    assert policy["family_holdout_and_source_holdout_required"] is True
    assert policy["training_allowed_on_failure"] is False
    assert policy["memory_promotion_on_failure"] is False
