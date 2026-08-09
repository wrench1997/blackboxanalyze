from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_pg360_slotwise_dataset import audit
from scripts.build_pg360_slotwise_dataset import build


ROOT = Path(__file__).resolve().parents[1]


def test_pg360_audit_passes_complete_slotwise_view() -> None:
    source = json.loads((ROOT / "research" / "pg359_context_index_dataset_v1.json").read_text(encoding="utf-8"))
    dataset = build(source, input_sha256="a" * 64, input_path="pg359.json")
    report = audit(dataset, dataset_sha256="b" * 64, dataset_path="pg360.json")
    assert report["status"] == "diagnostic_candidate_only"
    assert report["failures"] == []
    assert report["counts"]["source_rows"] == 1832
    assert report["information_preservation"]["target_information_added_to_context"] is False


def test_pg360_audit_rejects_wrong_query_slot() -> None:
    source = json.loads((ROOT / "research" / "pg359_context_index_dataset_v1.json").read_text(encoding="utf-8"))
    dataset = build(source, input_sha256="a" * 64, input_path="pg359.json")
    dataset["records"][0]["context_tokens"][-2] = "slot_query=next_action"
    report = audit(dataset, dataset_sha256="b" * 64, dataset_path="pg360.json")
    assert report["status"] == "blocked_incomplete"
    assert any("query" in failure for failure in report["failures"])
