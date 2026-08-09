from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_pg148_runs_multiple_large_model_posttraining_strategies() -> None:
    report = json.loads(Path("research/pg148_large_model_posttraining_report_v1.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed_pg148_large_model_posttraining"
    assert report["device"] == "cuda"
    assert {row["variant"] for row in report["variants"]} == {"scratch_large", "frozen_large", "low_lr_large", "adapter_large", "joint_xl"}
    scratch = next(row for row in report["variants"] if row["variant"] == "scratch_large")
    joint = next(row for row in report["variants"] if row["variant"] == "joint_xl")
    assert scratch["holdout"]["accuracy"] >= 0.80
    assert joint["language_forgetting"]["catastrophic_forgetting_detected"] is True
    assert report["capability_claim_allowed"] is False


def test_pg148_report_hash_is_recomputable() -> None:
    report = json.loads(Path("research/pg148_large_model_posttraining_report_v1.json").read_text(encoding="utf-8"))
    declared = report.pop("report_sha256")
    actual = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual

