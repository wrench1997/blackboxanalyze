from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _load(name: str) -> dict:
    return json.loads((Path("research") / name).read_text(encoding="utf-8"))


def test_pg160_adapter_only_is_best_offline_candidate_without_body_drift() -> None:
    report = _load("pg160_hierarchical_lr_early_stop_report_v1.json")
    assert report["status"] == "completed_pg160_hierarchical_lr_early_stop"
    assert report["device"] == "cuda"
    assert report["data_policy"]["holdout_labels_in_training"] is False
    assert report["data_policy"]["unseen_authorization_family_training_used"] is False
    assert report["promotion"]["capability_claim_allowed"] is False
    variants = {row["variant"]: row for row in report["variants"]}
    assert set(variants) == {"shared_adapter_source", "adapter_only_2e", "body_lr_1e5_2e", "body_lr_3e5_early"}
    candidate = variants["adapter_only_2e"]
    assert candidate["best_epoch"] == 2
    assert candidate["synthetic_holdout"]["false_stop_count"] == 0
    assert candidate["real_pg136_holdout"]["false_stop_count"] == 0
    assert candidate["unseen_authorization_family_holdout"]["accuracy"] == 0.75
    assert candidate["calibrated_unseen_authorization_family_holdout"]["coverage"] == 1.0
    assert candidate["surface_lm"]["perplexity"] == variants["shared_adapter_source"]["surface_lm"]["perplexity"]
    assert variants["body_lr_1e5_2e"]["synthetic_holdout"]["false_stop_count"] > 0
    assert variants["body_lr_3e5_early"]["surface_lm"]["perplexity"] > candidate["surface_lm"]["perplexity"]
    assert all(row["language_canary"]["catastrophic_forgetting_detected"] is False for row in variants.values())


def test_pg160_report_hash_is_recomputable() -> None:
    report = _load("pg160_hierarchical_lr_early_stop_report_v1.json")
    declared = report.pop("report_sha256")
    actual = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual


def test_pg160_protocol_uses_dev_only_early_stop() -> None:
    protocol = _load("pg160_hierarchical_lr_early_stop_protocol_v1.json")
    assert protocol["early_stop"]["holdout_used"] is False
    assert protocol["early_stop"]["patience"] == 1
    assert protocol["promotion"]["long_term_memory_promotion_allowed"] is False
