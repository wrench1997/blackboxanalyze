from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_pg347_multi_impl_slot import _sha_report, audit_report


ROOT = Path(__file__).resolve().parents[1]


def test_pg347_audit_blocks_observed_holdout_failures() -> None:
    report = json.loads((ROOT / "research" / "pg347_a800_multi_impl_slot_smoke_v1.json").read_text(encoding="utf-8"))
    audit = audit_report(report)
    assert audit["status"] == "passed_observation_blocked_capability"
    assert {"negative_false_allow", "ask_recall", "repair_recall"}.issubset(audit["failures"])


def test_pg347_report_hash_is_recomputed_without_mutation() -> None:
    report = json.loads((ROOT / "research" / "pg347_a800_multi_impl_slot_smoke_v1.json").read_text(encoding="utf-8"))
    assert report["report_sha256"] == _sha_report(report)
