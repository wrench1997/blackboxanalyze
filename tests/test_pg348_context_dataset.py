from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pg348_context_dataset_keeps_520_rows_and_holdout_split() -> None:
    data = json.loads((ROOT / "research" / "pg348_context_only_dataset_v1.json").read_text(encoding="utf-8"))
    assert data["counts"] == {"rows": 520, "train_rows": 240, "implementation_holdout_rows": 280, "training_eligible_rows": 0}
    assert all(row["context_firewall"]["forbidden_token_count"] == 0 for row in data["records"])
    assert all(row["training_eligible"] is False for row in data["records"])


def test_pg348_context_audit_is_diagnostic_not_promotion() -> None:
    audit = json.loads((ROOT / "research" / "pg348_context_only_information_audit_v1.json").read_text(encoding="utf-8"))
    assert audit["status"] == "diagnostic_only"
    assert audit["counts"]["implementation_split_leaks"] == 0
    assert audit["promotion"] == {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}
