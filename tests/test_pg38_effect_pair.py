import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg38_effect_head_is_oracle_blind_and_partial_transfer_is_not_promoted():
    report = _load("pg38_effect_pair_candidate_report_v1.json")
    assert report["status"] == "diagnostic_only"
    assert report["model"]["effect_head_family_agnostic"] is True
    assert report["model"]["typed_oracle_consumed_by_model"] is False
    assert report["model"]["raw_hashes_consumed"] is False
    assert report["splits"]["family_holdout"]["effect_recall_any_family"] == 0.5
    assert report["splits"]["ood_source"]["effect_recall_any_family"] == 0.5
    assert report["splits"]["source_holdout"]["effect_recall_any_family"] == 0.666667
    assert report["splits"]["family_holdout"]["effect_false_positive_rate"] == 0.0
    assert report["splits"]["negative_control"]["effect_false_positive_rate"] == 0.0
    assert report["capability_gate"]["claim_allowed"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg38_policy_separates_effect_transfer_from_rule_ir_family_success():
    protocol = _load("pg38_effect_pair_protocol_v1.json")
    assert protocol["pair_contract"]["effect_head_is_family_agnostic"] is True
    assert protocol["model_input_contract"]["typed_oracle_is_label_not_feature"] is True
    assert protocol["run_result"]["capability_gate_status"] == "blocked"
    assert protocol["run_result"]["capability_claim_allowed"] is False
    assert protocol["status"] == "run_completed_capability_gate_failed_no_promotion"
    rules = json.loads((ROOT / "research" / "improvement_rules.json").read_text(encoding="utf-8"))
    policy = rules["pg38_run_result_policy"]
    assert policy["effect_recall_is_not_family_recall"] is True
    assert policy["partial_effect_transfer_does_not_authorize_promotion"] is True
    assert policy["next_experiment"].startswith("PG-39")
