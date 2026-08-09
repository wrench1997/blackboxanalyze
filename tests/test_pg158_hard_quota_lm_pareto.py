from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _load(name: str) -> dict:
    return json.loads((Path("research") / name).read_text(encoding="utf-8"))


def test_pg158_pareto_keeps_unseen_family_out_of_training() -> None:
    report = _load("pg158_hard_quota_lm_pareto_report_v1.json")
    assert report["status"] == "completed_pg158_hard_quota_lm_pareto"
    assert report["device"] == "cuda"
    assert report["source"]["unseen_authorization_family_count"] == 12
    assert report["data_policy"]["unseen_authorization_family_training_used"] is False
    assert report["data_policy"]["selection_used_dev_labels"] is False
    assert report["data_policy"]["selection_used_holdout_labels"] is False
    assert report["promotion"]["capability_claim_allowed"] is False
    variants = {row["variant"]: row for row in report["variants"]}
    assert set(variants) == {"lm_anchor_baseline", "q400_lm025", "q800_lm050", "q1200_lm100"}
    assert variants["q400_lm025"]["hard_negative_count"] == 400
    assert variants["q800_lm050"]["hard_negative_count"] == 800
    assert variants["q1200_lm100"]["hard_negative_count"] == 1200
    assert variants["q1200_lm100"]["surface_lm"]["perplexity"] < variants["q400_lm025"]["surface_lm"]["perplexity"]
    assert all(row["calibrated_real_pg136_holdout"]["false_stop_count"] == 0 for row in variants.values())
    assert all(row["language_canary"]["catastrophic_forgetting_detected"] is False for row in variants.values())


def test_pg158_report_hash_is_recomputable() -> None:
    report = _load("pg158_hard_quota_lm_pareto_report_v1.json")
    declared = report.pop("report_sha256")
    actual = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual


def test_pg158_protocol_declares_family_holdout_and_pareto_grid() -> None:
    protocol = _load("pg158_hard_quota_lm_pareto_protocol_v1.json")
    assert protocol["hard_quota_grid"]["per_source"] == [0, 200, 400, 600]
    assert protocol["unseen_family"].startswith("authorization appears only")
    assert protocol["promotion"]["long_term_memory_promotion_allowed"] is False
