from __future__ import annotations

import json
from pathlib import Path

from scripts.extend_pg350_oracle_vocabulary import extend


def test_oracle_vocabulary_is_append_only_and_abstract() -> None:
    root = Path(__file__).parents[1]
    base = json.loads((root / "research/pg349_dynamic_typed_vocabulary_v8.json").read_text(encoding="utf-8"))
    result = extend(base, base_sha256="a" * 64, dataset_path=root / "research/pg350_oracle_slot_source_rows_v1.json")
    assert set(base["context_tokens"]).issubset(result["context_tokens"])
    assert set(base["target_tokens"]).issubset(result["target_tokens"])
    assert "oracle_ref=typed_effect" in result["target_tokens"]
    assert "negative_control_presence_ref=matched_triplet" in result["target_tokens"]
    assert result["vocabulary_policy"]["raw_literal_tokens_allowed"] is False
    assert result["promotion"]["payload"] is False
    assert all("<script" not in token.casefold() for token in result["target_tokens"])
