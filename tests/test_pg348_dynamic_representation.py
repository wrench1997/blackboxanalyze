from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_pg348_dynamic_representation import _sha_report, audit_report


ROOT = Path(__file__).resolve().parents[1]


def test_pg348_dynamic_representation_audit_is_capability_blocked() -> None:
    report = json.loads((ROOT / "research" / "pg348_dynamic_context_a800_representation_v1.json").read_text(encoding="utf-8"))
    audit = audit_report(report)
    assert audit["status"] == "passed_representation_observation_blocked_capability"
    assert audit["information_gate_status"] == "diagnostic_only"
    assert audit["promotion"]["vulnerability_claim_allowed"] is False


def test_pg348_dynamic_representation_report_hash_matches() -> None:
    report = json.loads((ROOT / "research" / "pg348_dynamic_context_a800_representation_v1.json").read_text(encoding="utf-8"))
    assert report["report_sha256"] == _sha_report(report)
