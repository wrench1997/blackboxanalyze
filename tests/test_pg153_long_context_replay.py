from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_pg153_replay_preserves_old_canary_and_long_context_improves_fit() -> None:
    report = json.loads(Path("research/pg153_long_context_replay_report_v1.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed_pg153_long_context_replay"
    assert report["device"] == "cuda"
    assert report["source"]["replay_count"] == 1200
    assert report["source"]["long_holdout_count"] == 416
    variants = {row["variant"]: row for row in report["variants"]}
    assert variants["short_context_replay"]["replay_rows"] == 1200
    assert variants["long_context_replay"]["replay_rows"] == 1200
    assert variants["long_context_replay"]["after_new_long_holdout"]["perplexity"] < variants["short_context_replay"]["after_new_long_holdout"]["perplexity"]
    assert variants["long_context_replay"]["old_canary_forgetting"]["catastrophic_forgetting_detected"] is False
    assert variants["long_context_no_replay"]["old_canary_forgetting"]["catastrophic_forgetting_detected"] is True
    assert report["promotion"]["capability_claim_allowed"] is False


def test_pg153_report_hash_is_recomputable() -> None:
    report = json.loads(Path("research/pg153_long_context_replay_report_v1.json").read_text(encoding="utf-8"))
    declared = report.pop("report_sha256")
    actual = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual
