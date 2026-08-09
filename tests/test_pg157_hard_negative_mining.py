from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _load(name: str) -> dict:
    return json.loads((Path("research") / name).read_text(encoding="utf-8"))


def test_pg157_mining_is_train_split_only_and_keeps_candidates_offline() -> None:
    report = _load("pg157_hard_negative_mining_report_v1.json")
    assert report["status"] == "completed_pg157_hard_negative_mining"
    assert report["device"] == "cuda"
    assert report["source"]["hard_negative_count"] == 800
    assert report["source"]["hard_negative_source_counts"] == {"pg136_real": 400, "pg149_synthetic": 400}
    assert report["data_policy"]["selection_used_dev_labels"] is False
    assert report["data_policy"]["selection_used_holdout_labels"] is False
    assert report["data_policy"]["pg146_training_labels_used"] is False
    assert report["promotion"]["capability_claim_allowed"] is False
    variants = {row["variant"]: row for row in report["variants"]}
    assert set(variants) == {"lm_anchor_baseline", "focal_hard_mining", "contrastive_hard_mining"}
    assert variants["focal_hard_mining"]["calibrated_synthetic_holdout"]["false_stop_count"] == 0
    assert variants["contrastive_hard_mining"]["calibrated_synthetic_holdout"]["false_stop_count"] > 0
    assert variants["focal_hard_mining"]["surface_lm"]["perplexity"] > variants["lm_anchor_baseline"]["surface_lm"]["perplexity"]
    assert all(row["language_canary"]["catastrophic_forgetting_detected"] is False for row in variants.values())


def test_pg157_report_hash_is_recomputable() -> None:
    report = _load("pg157_hard_negative_mining_report_v1.json")
    declared = report.pop("report_sha256")
    actual = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual


def test_pg157_protocol_declares_independent_mining_pool() -> None:
    protocol = _load("pg157_hard_negative_mining_protocol_v1.json")
    assert protocol["mining_pool"] == "PG-154 action_train only; dev/holdout labels excluded"
    assert protocol["threshold_fit"] == "highest dev coverage with zero dev false_stop"
    assert protocol["promotion"]["long_term_memory_promotion_allowed"] is False
