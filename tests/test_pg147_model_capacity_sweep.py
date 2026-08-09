from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_pg147_capacity_matrix_contains_multiple_real_checkpoints() -> None:
    report = json.loads(Path("research/pg147_model_capacity_sweep_report_v1.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed_pg147_model_capacity_sweep"
    assert report["device"] == "cuda"
    assert report["corpus"]["generated_count"] == 12000
    assert len(report["variants"]) == 4
    assert max(row["parameter_count"] for row in report["variants"]) == 57064106
    best = min(report["variants"], key=lambda row: row["holdout"]["perplexity"])
    assert best["variant"] == "large_transformer"
    assert report["capability_claim_allowed"] is False


def test_pg147_report_hash_is_recomputable() -> None:
    report = json.loads(Path("research/pg147_model_capacity_sweep_report_v1.json").read_text(encoding="utf-8"))
    declared = report.pop("report_sha256")
    actual = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual

