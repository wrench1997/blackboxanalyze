import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg37_ablation_is_oracle_blind_and_no_variant_is_promoted():
    report = _load("pg37_representation_ablation_report_v1.json")
    assert report["status"] == "diagnostic_only"
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    for name, item in report["ablations"].items():
        assert item["model"]["typed_oracle_consumed_by_model"] is False
        assert item["model"]["surface_variant_label_consumed_by_model"] is False
        assert item["capability_gate"]["claim_allowed"] is False
        assert item["promotion"]["training_allowed"] is False
        assert item["promotion"]["memory_promotion_allowed"] is False
    assert report["comparison"]["family_holdout_typed_recall"] == {"surface_only": 0.0, "counterfactual_paired": 0.0, "phase_only": 0.0}
    assert report["comparison"]["unknown_false_positive_rate"] == {"surface_only": 0.0, "counterfactual_paired": 0.0, "phase_only": 0.0}


def test_pg37_ablation_policy_rejects_vacuous_abstention_as_success():
    rules = _load("improvement_rules.json")
    policy = rules["pg37_ablation_result_policy"]
    assert policy["surface_holdout_required"] is True
    assert policy["counterfactual_pair_gain_required"] is True
    assert policy["family_holdout_recall_gain_required"] is True
    assert policy["phase_only_abstain_is_not_success"] is True
    assert policy["pair_loss_without_recall_gain_is_failure"] is True
    assert policy["all_ablation_gates_blocked"] is True
    assert policy["training_allowed_on_failure"] is False
