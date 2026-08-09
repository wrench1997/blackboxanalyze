import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg39_coarse_delta_transfers_effect_detection_but_not_family_naming():
    report = _load("pg39_coarse_delta_candidate_report_v1.json")
    assert report["status"] == "diagnostic_only"
    assert report["model"]["coarse_delta_dim"] == 32
    assert report["model"]["effect_head_family_agnostic"] is True
    assert report["model"]["typed_oracle_consumed_by_model"] is False
    assert report["splits"]["surface_holdout"]["effect_recall_any_family"] == 1.0
    assert report["splits"]["family_holdout"]["effect_recall_any_family"] == 1.0
    assert report["splits"]["ood_source"]["effect_recall_any_family"] == 1.0
    assert report["splits"]["source_holdout"]["effect_recall_any_family"] == 1.0
    assert report["splits"]["negative_control"]["effect_false_positive_rate"] == 0.0
    assert report["splits"]["family_holdout"]["typed_recall"] == 0.0
    assert report["capability_gate"]["claim_allowed"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg39_protocol_and_policy_keep_effect_and_family_gates_separate():
    protocol = _load("pg39_coarse_delta_protocol_v1.json")
    assert protocol["coarse_feature_contract"]["effect_head_family_agnostic"] is True
    assert protocol["coarse_feature_contract"]["typed_oracle_is_label_not_feature"] is True
    assert protocol["run_result"]["effect_false_positive_rate"] == 0.0
    assert protocol["run_result"]["capability_gate_status"] == "blocked"
    assert protocol["status"] == "run_completed_effect_gain_family_gate_failed_no_promotion"
    rules = json.loads((ROOT / "research" / "improvement_rules.json").read_text(encoding="utf-8"))
    policy = rules["pg39_run_result_policy"]
    assert policy["effect_recall_gain_is_real_but_partial"] is True
    assert policy["typed_family_recall_still_required"] is True
    assert policy["effect_head_cannot_name_unseen_family"] is True
    assert policy["next_experiment"].startswith("PG-40")
