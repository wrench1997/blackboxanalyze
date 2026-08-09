import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def test_pg36_formal_candidate_is_oracle_blind_and_quarantined():
    report = _load("research/pg36_formal_rule_ir_candidate_report_v1.json")
    assert report["status"] == "diagnostic_only"
    assert report["model"]["typed_oracle_consumed_by_model"] is False
    assert report["training"]["count"] == 128
    assert report["capability_gate"]["claim_allowed"] is False
    assert report["capability_gate"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["splits"]["family_holdout"]["typed_recall"] == 0.0
    assert report["splits"]["ood_source"]["typed_recall"] == 0.0
    assert report["source_holdout"]["false_positive_rate"] == 0.0
    assert report["unknown_abstain"]["false_positive_rate"] == 0.0


def test_pg36_formal_policy_freezes_independent_split_and_no_promotion():
    rules = _load("research/improvement_rules.json")
    policy = rules["pg36_formal_rule_ir_policy"]
    assert policy["typed_oracle_is_label_not_feature"] is True
    assert policy["training_implementation"] == "north"
    assert policy["source_holdout_implementation"] == "south"
    assert policy["family_holdout_required"] == ["logic", "url_redirect"]
    assert policy["seed_holdout_required"] is True
    assert policy["training_allowed_on_gate_failure"] is False
    assert policy["memory_promotion_on_failure"] is False
