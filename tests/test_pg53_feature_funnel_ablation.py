import json
import re
from pathlib import Path


REPORT = Path("research/pg53_feature_funnel_ablation_report_v1.json")


def test_feature_funnel_ablation_is_same_split_and_diagnostic_only():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["split"]["train"] == "pg35 seeds 5301/5307"
    assert report["split"]["dev"] == "pg35 seed 5311"
    assert report["split"]["holdout"] == "pg36 all seeds"
    assert report["all_safe_candidate"]["feature_count"] == 37
    assert set(report["reviewed_funnel"]["features"]) == {
        "geometry_change_presence_control",
        "geometry_true_boolean_delta_ratio_control",
        "geometry_array_item_count",
        "geometry_nonzero_numeric_count",
        "geometry_numeric_count",
        "geometry_array_count",
    }
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["formal_claim_allowed"] is False
    assert re.fullmatch(r"[0-9a-f]{64}", report["funnel_review_evidence_sha256"])


def test_feature_funnel_ablation_does_not_hide_zero_recall_or_false_accepts():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    for feature_set in (report["all_safe_candidate"], report["reviewed_funnel"]):
        metrics = feature_set["holdout"]["calibrated"]
        assert metrics["false_accept_count"] == 0
        assert 0.0 <= metrics["typed_recall"] <= 1.0
        assert 0.0 <= metrics["abstain_rate"] <= 1.0
