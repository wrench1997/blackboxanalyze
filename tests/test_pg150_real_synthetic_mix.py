from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_pg150_real_trace_mix_improves_unseen_real_holdout() -> None:
    report = json.loads(Path("research/pg150_real_synthetic_mix_report_v1.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed_pg150_real_synthetic_mix"
    assert report["device"] == "cuda"
    results = {row["variant"]: row for row in report["variants"]}
    assert results["synthetic_only"]["real_pg136_holdout"]["accuracy"] == 0.22727273
    assert results["real_5_percent"]["real_pg136_holdout"]["accuracy"] == 0.80681818
    assert results["real_25_percent"]["real_pg136_holdout"]["accuracy"] == 0.86363636
    assert results["real_25_percent"]["synthetic_holdout"]["accuracy"] == 0.865
    assert report["capability_claim_allowed"] is False


def test_pg150_report_hash_is_recomputable() -> None:
    report = json.loads(Path("research/pg150_real_synthetic_mix_report_v1.json").read_text(encoding="utf-8"))
    declared = report.pop("report_sha256")
    actual = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual

