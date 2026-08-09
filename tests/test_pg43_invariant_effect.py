import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg42_triage_identifies_zero_variance_normalization_failure():
    triage = _load("pg42_independent_ood_triage_v1.json")
    assert triage["status"] == "diagnostic_only"
    assert 15 in triage["checkpoint"]["zero_variance_feature_indices"]
    assert triage["groups"]["pg42_dev"]["surface_variant_metrics"]["framed"]["zero_variance_shift_count"] == 48
    assert triage["groups"]["pg42_dev"]["surface_variant_metrics"]["framed"]["effect_recall"] == 0.0
    assert triage["training_allowed"] is False
    assert triage["memory_promotion_allowed"] is False


def test_pg43_invariant_effect_recovers_independent_effect_detection():
    report = _load("pg43_invariant_effect_candidate_report_v1.json")
    assert report["status"] == "diagnostic_only"
    assert report["training_source"]["pg42_used_for_training"] is False
    assert report["model"]["typed_oracle_consumed_by_model"] is False
    assert report["pg42_splits"]["implementation_holdout"]["effect_recall_any_family"] == 1.0
    assert report["pg42_splits"]["family_holdout"]["effect_recall_any_family"] == 1.0
    assert report["pg42_splits"]["negative_control"]["effect_false_positive_rate"] == 0.0
    assert report["candidate_effect_gate"]["status"] == "passed"
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg43_safe_router_passes_effect_and_unknown_abstain_but_not_formal_claim():
    report = _load("pg43_pg42_safe_router_report_v1.json")
    metrics = report["routing"]["metrics"]
    assert metrics["pair_count"] == 1440
    assert metrics["effect_recall_any_family"] == 1.0
    assert metrics["known_family_recall"] == 1.0
    assert metrics["unknown_effect_recall"] == 1.0
    assert metrics["negative_effect_false_accept_count"] == 0
    assert metrics["unknown_misname_count"] == 0
    assert metrics["unknown_strict_abstain"] is True
    assert report["effect_gate"]["claim_allowed"] is True
    assert report["formal_capability_claim_allowed"] is False
    assert report["promotion"]["training_allowed"] is False


def test_pg43_protocol_keeps_effect_gate_separate_from_formal_promotion():
    protocol = _load("pg43_invariant_effect_protocol_v1.json")
    assert protocol["training_contract"]["pg42_used_for_training"] is False
    assert protocol["run_result"]["effect_gate_status"] == "passed"
    assert protocol["run_result"]["formal_capability_claim_allowed"] is False
    assert protocol["run_result"]["training_allowed"] is False
    assert protocol["promotion_gate"]["formal_family_decoder_still_required"] is True
    assert protocol["status"] == "run_completed_effect_gate_passed_no_promotion"
