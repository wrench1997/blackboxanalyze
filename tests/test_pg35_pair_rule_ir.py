import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg35_pair_candidate_uses_two_heads_and_remains_quarantined():
    report = _load("pg35_pair_rule_ir_report_v1.json")
    model = report["model"]
    assert report["status"] == "diagnostic_only"
    assert model["visible_projection_labels"] is False
    assert model["typed_oracle_consumed_by_model"] is False
    assert model["positive_authority"] is False
    assert report["training_allowed"] is False
    assert report["memory_promotion_allowed"] is False
    assert report["checkpoint_selection"] == "minimum_supervised_plus_pair_objective"
    assert report["training"]["pair_count"] >= 24
    assert report["capability_gate"]["status"] == "no_proven_gain"
    assert report["capability_gate"]["claim_allowed"] is False


def test_pg35_pair_candidate_keeps_negative_controls_safe_and_reports_holdout_failure():
    report = _load("pg35_pair_rule_ir_report_v1.json")
    assert report["splits"]["negative_control"]["false_positive_rate"] == 0.0
    assert report["splits"]["family_holdout"]["typed_recall"] == 0.0
    assert report["splits"]["ood_source"]["typed_recall"] == 0.0
    for metrics in report["pair_consistency"].values():
        assert metrics["pair_count"] > 0
        assert 0.0 <= metrics["agreement_rate"] <= 1.0
