from __future__ import annotations

import json
from pathlib import Path

from scripts.build_pg360_slotwise_dataset import SLOTS, build


ROOT = Path(__file__).resolve().parents[1]


def _source() -> dict:
    return json.loads((ROOT / "research" / "pg359_context_index_dataset_v1.json").read_text(encoding="utf-8"))


def test_slotwise_expansion_preserves_context_and_uses_schema_query_only() -> None:
    source = _source()
    result = build(source, input_sha256="a" * 64, input_path="pg359.json")
    assert result["status"] == "diagnostic_candidate_only"
    assert result["counts"]["records"] == len(source["records"]) * len(SLOTS)
    row = result["records"][0]
    original = source["records"][0]["context_tokens"]
    assert row["context_tokens"][: len(original)] == original
    assert row["context_tokens"][-3:] == ["[SLOT_QUERY_BOS]", "slot_query=question", "[SLOT_QUERY_EOS]"]
    assert row["target_tokens"][0] == "[TARGET_BOS]"
    assert row["target_tokens"][-1] == "[TARGET_EOS]"
    assert row["slot_query_contract"]["target_value_in_context"] is False


def test_slotwise_rows_cover_every_slot_and_keep_raw_firewall_closed() -> None:
    result = build(_source(), input_sha256="b" * 64, input_path="pg359.json")
    assert {row["slot"] for row in result["records"]} == set(SLOTS)
    assert all(row["context_firewall"] == {"forbidden_token_count": 0, "sidecars_off_context": True} for row in result["records"])
    assert all("payload=" not in " ".join(row["context_tokens"]) for row in result["records"])


def test_slotwise_builder_quarantines_raw_context() -> None:
    source = _source()
    source["records"][0]["context_tokens"] = ["raw_payload=bad"]
    result = build(source, input_sha256="c" * 64, input_path="pg359.json")
    assert result["status"] == "blocked_incomplete"
    assert result["promotion"]["training_allowed"] is False
