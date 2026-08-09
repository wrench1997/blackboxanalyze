from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_pg359_context_index_dataset import audit
from scripts.build_pg359_context_index_dataset import build


ROOT = Path(__file__).resolve().parents[1]


def test_pg359_audit_passes_append_only_view() -> None:
    source_path = ROOT / "research" / "pg351_ask_oracle_composition_dataset_v2.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    dataset = build(source, input_sha256="a" * 64, input_path="source.json")
    report = audit(dataset, dataset_sha256="b" * 64, dataset_path="pg359.json")
    assert report["status"] == "diagnostic_candidate_only"
    assert report["failures"] == []
    assert report["context_index"]["full_original_context_preserved"] is True
    assert report["context_index"]["target_information_added"] is False


def test_pg359_audit_rejects_tampered_index() -> None:
    source = json.loads((ROOT / "research" / "pg351_ask_oracle_composition_dataset_v2.json").read_text(encoding="utf-8"))
    dataset = build(source, input_sha256="a" * 64, input_path="source.json")
    dataset["records"][0]["context_tokens"][-2] = "index_typed=present"
    report = audit(dataset, dataset_sha256="b" * 64, dataset_path="pg359.json")
    assert report["status"] == "blocked_incomplete"
    assert any("index_derivation" in failure for failure in report["failures"])
