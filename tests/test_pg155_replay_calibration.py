from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_pg155_calibration_controls_false_stop_without_label_based_selection() -> None:
    report = json.loads(Path("research/pg155_replay_calibration_report_v1.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed_pg155_replay_calibration"
    assert report["device"] == "cuda"
    assert report["data_policy"]["selection_used_labels"] is False
    assert report["data_policy"]["calibration_fit_on_holdout"] is False
    variants = {row["variant"]: row for row in report["variants"]}
    assert variants["balanced_replay"]["selection"]["source_counts"] == {"pg149_synthetic": 600, "pg136_real": 600}
    assert variants["selective_replay"]["selection"]["source_counts"] == {"pg149_synthetic": 1200}
    assert variants["balanced_replay"]["calibrated_real_pg136_holdout"]["false_stop_count"] == 0
    assert variants["selective_replay"]["calibrated_real_pg136_holdout"]["false_stop_count"] == 0
    assert variants["balanced_replay"]["calibrated_real_pg136_holdout"]["coverage"] < 1.0
    assert variants["selective_replay"]["calibrated_real_pg136_holdout"]["coverage"] < 1.0
    assert all(row["language_canary"]["catastrophic_forgetting_detected"] is False for row in variants.values())
    assert report["promotion"]["capability_claim_allowed"] is False


def test_pg155_report_hash_is_recomputable() -> None:
    report = json.loads(Path("research/pg155_replay_calibration_report_v1.json").read_text(encoding="utf-8"))
    declared = report.pop("report_sha256")
    actual = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual
