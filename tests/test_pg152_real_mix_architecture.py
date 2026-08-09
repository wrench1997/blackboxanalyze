from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_pg152_keeps_real_mix_fixed_and_measures_both_bodies() -> None:
    report = json.loads(Path("research/pg152_real_mix_architecture_report_v1.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed_pg152_real_mix_architecture"
    assert report["device"] == "cuda"
    assert report["source"]["mixed_train_count"] == 8533
    assert report["source"]["real_rows_in_mixed_train"] == 2133
    variants = {row["variant"]: row for row in report["variants"]}
    assert set(variants) == {"dense_large_real25", "moe_large4_real25"}
    assert variants["dense_large_real25"]["real_pg136_holdout"]["accuracy"] == 0.86363636
    assert variants["moe_large4_real25"]["real_pg136_holdout"]["accuracy"] == 0.86363636
    assert variants["moe_large4_real25"]["synthetic_holdout"]["accuracy"] > variants["dense_large_real25"]["synthetic_holdout"]["accuracy"]
    assert variants["moe_large4_real25"]["real_surface_lm"]["perplexity"] < variants["dense_large_real25"]["real_surface_lm"]["perplexity"]
    assert report["promotion"]["capability_claim_allowed"] is False


def test_pg152_report_hash_is_recomputable() -> None:
    report = json.loads(Path("research/pg152_real_mix_architecture_report_v1.json").read_text(encoding="utf-8"))
    declared = report.pop("report_sha256")
    actual = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual
