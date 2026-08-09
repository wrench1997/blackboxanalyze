import json
from pathlib import Path


def _read(name):
    return json.loads(Path("research", name).read_text(encoding="utf-8"))


def test_pg55_feature_funnel_and_training_splits_are_isolated():
    funnel = _read("pg55_invariant_feature_funnel_dataset_v1.json")
    report = _read("pg55_invariant_feature_funnel_report_v1.json")
    candidate = _read("pg55_invariant_rule_ir_candidate_report_v1.json")
    assert len(funnel["rows"]) == 324
    assert len(funnel["candidate_feature_names"]) == 54
    assert funnel["review_decision"] == "approved_for_downstream_ood_experiment"
    assert len(funnel["accepted_features"]) == 7
    assert report["audit"]["stage_counts"] == {
        "candidate": 54,
        "observable_safe": 37,
        "quality": 30,
        "source_leakage": 11,
        "seed_stability": 11,
        "label_utility_audit": 9,
        "redundancy_pruned": 7,
    }
    assert candidate["training"]["train_rows"] == 324
    assert candidate["training"]["dev_rows"] == 108
    assert candidate["training"]["holdout_rows"] == 120
    assert candidate["training"]["oracle_in_features"] is False
    assert candidate["training"]["family_label_in_features"] is False
    assert Path(candidate["training"]["checkpoint"]).exists()


def test_pg55_raw_failure_and_density_safety_are_not_capability_claims():
    candidate = _read("pg55_invariant_rule_ir_candidate_report_v1.json")
    dev = candidate["dev"]["metrics"]
    raw = candidate["holdout"]["raw"]
    gated = candidate["holdout"]["density_gated"]
    assert dev["known_family_recall"] == 0.083333
    assert dev["negative_effect_false_accept_count"] == 0
    assert raw["known_family_recall"] == 0.125
    assert raw["known_wrong_family_count"] == 84
    assert raw["unknown_misname_count"] == 12
    assert raw["negative_effect_false_accept_count"] == 0
    assert gated["known_family_recall"] == 0.0
    assert gated["unknown_misname_count"] == 0
    assert gated["negative_effect_false_accept_count"] == 0
    assert gated["abstain_rate"] == 1.0
    assert candidate["promotion"]["training_allowed"] is False
    assert candidate["promotion"]["memory_promotion_allowed"] is False
    assert candidate["promotion"]["formal_claim_allowed"] is False
