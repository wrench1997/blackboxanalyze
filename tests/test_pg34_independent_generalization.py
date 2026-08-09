import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg33_checkpoint_is_evaluated_source_holdout_without_weight_updates():
    report = _load("pg34_independent_generalization_report_v1.json")
    assert report["source"]["independent_target_implementation"] is True
    assert report["source"]["checkpoint_source"] == "PG-33-only"
    assert report["source"]["weights_updated"] is False
    assert report["aggregate"]["uncalibrated"]["false_positive_rate"] == 0.9
    assert report["aggregate"]["uncalibrated"]["typed_recall"] == 0.0
    assert report["aggregate"]["calibrated"]["typed_recall"] == 0.0
    assert report["aggregate"]["calibrated"]["false_positive_rate"] == 0.0
    assert report["capability_gate"]["status"] == "no_proven_gain"
    assert report["capability_gate"]["claim_allowed"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
