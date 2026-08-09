from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _load(name: str) -> dict:
    return json.loads((Path("research") / name).read_text(encoding="utf-8"))


def test_pg156_keeps_real_surface_evaluation_only_and_compares_three_variants() -> None:
    report = _load("pg156_calibration_surface_anchor_report_v1.json")
    assert report["status"] == "completed_pg156_calibration_surface_anchor"
    assert report["device"] == "cuda"
    assert report["data_policy"]["pg145_training_eligible"] is False
    assert report["data_policy"]["pg146_training_labels_used"] is False
    assert report["data_policy"]["surface_anchor_labels"] is False
    assert report["promotion"]["capability_claim_allowed"] is False
    assert report["source"]["surface_anchor_count"] == 600
    variants = {row["variant"]: row for row in report["variants"]}
    assert set(variants) == {"lm_anchor_baseline", "calibration_aware", "surface_anchor_calibrated"}
    assert variants["calibration_aware"]["raw_real_pg136_holdout"]["false_stop_count"] == 0
    assert variants["surface_anchor_calibrated"]["raw_real_pg136_holdout"]["false_stop_count"] == 0
    assert variants["surface_anchor_calibrated"]["calibrated_real_pg136_holdout"]["false_stop_count"] == 0
    assert variants["surface_anchor_calibrated"]["surface_lm"]["perplexity"] < variants["calibration_aware"]["surface_lm"]["perplexity"]
    assert variants["surface_anchor_calibrated"]["calibrated_synthetic_holdout"]["false_stop_count"] > 0
    assert all(row["language_canary"]["catastrophic_forgetting_detected"] is False for row in variants.values())


def test_pg156_report_hash_is_recomputable() -> None:
    report = _load("pg156_calibration_surface_anchor_report_v1.json")
    declared = report.pop("report_sha256")
    actual = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual


def test_pg156_protocol_forbids_evaluation_label_leakage() -> None:
    protocol = _load("pg156_calibration_surface_anchor_protocol_v1.json")
    assert protocol["surface_proxy"] == "PG-147 train only; no PG-145/PG-146 rows"
    assert protocol["threshold_fit"] == "highest dev coverage with zero dev false_stop"
    assert protocol["promotion"]["long_term_memory_promotion_allowed"] is False
