from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_pg154_replay_and_guard_matrix_is_source_separated() -> None:
    report = json.loads(Path("research/pg154_multisource_action_replay_report_v1.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed_pg154_multisource_action_replay"
    assert report["device"] == "cuda"
    assert report["source"]["action_mix_count"] == 8533
    assert report["source"]["lm_replay_count"] == 1200
    assert report["data_policy"]["pg146_training_labels_used"] is False
    variants = {row["variant"]: row for row in report["variants"]}
    assert variants["action_only"]["synthetic_holdout"]["false_stop_count"] == 24
    assert variants["lm_anchor"]["synthetic_holdout"]["false_stop_count"] == 0
    assert variants["lm_anchor"]["real_pg136_holdout"]["accuracy"] == 0.86363636
    assert variants["false_stop_guard"]["real_pg136_holdout"]["false_stop_count"] == 0
    assert all(row["evaluation_only_surface_unknown"]["unknown_abstain_rate"] == 1.0 for row in variants.values())
    assert all(row["language_canary"]["catastrophic_forgetting_detected"] is False for row in variants.values())
    assert report["promotion"]["capability_claim_allowed"] is False


def test_pg154_report_hash_is_recomputable() -> None:
    report = json.loads(Path("research/pg154_multisource_action_replay_report_v1.json").read_text(encoding="utf-8"))
    declared = report.pop("report_sha256")
    actual = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual
