from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_pg149_synthetic_alignment_is_separated_from_real_trace_transfer() -> None:
    report = json.loads(Path("research/pg149_causal_action_alignment_report_v1.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed_pg149_causal_action_alignment"
    assert report["device"] == "cuda"
    assert report["corpus"]["generated_count"] == 8000
    assert max(row["synthetic_holdout"]["accuracy"] for row in report["variants"]) >= 0.85
    assert max(row["real_pg136_holdout"]["accuracy"] for row in report["variants"]) == 0.20454545
    assert report["capability_claim_allowed"] is False


def test_pg149_report_hash_is_recomputable() -> None:
    report = json.loads(Path("research/pg149_causal_action_alignment_report_v1.json").read_text(encoding="utf-8"))
    declared = report.pop("report_sha256")
    actual = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual

