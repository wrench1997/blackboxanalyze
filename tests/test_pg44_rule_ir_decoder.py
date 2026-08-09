import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg44_decoder_keeps_known_family_binding_and_unknown_route_separate():
    report = _load("pg44_rule_ir_decoder_report_v1.json")
    assert report["status"] == "diagnostic_only"
    assert report["training"]["typed_oracle_consumed_by_model"] is False
    assert report["training"]["pg42_used_for_training"] is False
    assert report["training"]["semantic_reference_contains_family_name"] is False
    assert report["pg40_seed_holdout"]["typed_recall"] == 1.0
    assert report["pg42_splits"]["implementation_holdout"]["known_family_recall"] == 1.0
    assert report["pg42_splits"]["implementation_holdout"]["unknown_effect_recall"] == 1.0
    assert report["pg42_splits"]["implementation_holdout"]["unknown_strict_abstain"] is True
    assert report["pg42_splits"]["family_holdout"]["known_positive_count"] == 0
    assert report["pg42_splits"]["family_holdout"]["unknown_positive_count"] == 36
    assert report["pg42_splits"]["family_holdout"]["unknown_strict_abstain"] is True
    assert report["pg42_splits"]["negative_control"]["false_positive_rate"] == 0.0
    assert report["formal_capability_claim_allowed"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg44_protocol_records_formal_claim_blocker():
    protocol = _load("pg44_rule_ir_decoder_protocol_v1.json")
    assert protocol["model_input_contract"]["typed_oracle_is_target_only"] is True
    assert protocol["holdout"]["strict_family_ood"].startswith("template-boundary")
    assert protocol["run_result"]["pg42_independent_known_family_recall"] == 1.0
    assert protocol["run_result"]["pg42_independent_unknown_misname_count"] == 0
    assert protocol["run_result"]["strict_new_family_named_recall"] == 0.0
    assert protocol["run_result"]["formal_capability_claim_allowed"] is False
    assert protocol["run_result"]["training_allowed"] is False
    assert protocol["status"] == "run_completed_known_binding_passed_unknown_abstain_no_promotion"
