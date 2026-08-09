import json
import re
from pathlib import Path


REPORT = Path("research/pg53_mixed_source_seed_holdout_report_v1.json")


def test_mixed_source_seed_holdout_is_explicit_and_quarantined():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["split"]["train"] == "PG-35 + PG-36, seeds 5301/5307"
    assert report["split"]["dev"] == "PG-35, seed 5311"
    assert report["split"]["holdout"] == "PG-36, seed 5311"
    assert report["split"]["train_rows"] == 180
    assert report["split"]["dev_rows"] == 36
    assert report["split"]["holdout_rows"] == 36
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["formal_claim_allowed"] is False
    assert re.fullmatch(r"[0-9a-f]{64}", report["report_sha256"])


def test_mixed_source_holdout_does_not_convert_abstain_into_success():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    metrics = report["holdout"]["calibrated"]
    assert metrics["false_accept_count"] == 0
    assert 0.0 <= metrics["typed_recall"] <= 1.0
    assert metrics["abstain_rate"] >= 0.0
