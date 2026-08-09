from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from scripts.build_pg350_oracle_slot_dataset import build


def _source_dataset() -> dict:
    path = Path(__file__).parents[1] / "research" / "pg349_dynamic_typed_source_rows_v5.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_builder_adds_abstract_oracle_and_negative_presence_without_raw_values() -> None:
    source = _source_dataset()
    result = build(source, input_sha256=hashlib.sha256(b"fixture-input").hexdigest())
    assert result["status"] == "diagnostic_only"
    assert result["counts"]["records"] == result["counts"]["input_records"]
    assert result["counts"]["training_eligible_rows"] == 0
    assert result["promotion"]["payload_catalog_promotion_allowed"] is False
    first = result["records"][0]
    assert "oracle_ref=typed_effect" in first["target_tokens"]
    assert "negative_control_presence_ref=matched_triplet" in first["target_tokens"]
    encoded = json.dumps(result, ensure_ascii=False)
    assert "raw_payload=" not in encoded
    assert "response_body=" not in encoded


def test_builder_keeps_negative_and_failure_targets_abstract() -> None:
    source = _source_dataset()
    result = build(source, input_sha256="a" * 64)
    negative = next(row for row in result["records"] if row["target_projection"]["probe_variant_ref"] == "negative_control")
    failure = next(row for row in result["records"] if row["target_projection"]["question"] == "ask_failure")
    assert negative["target_projection"]["oracle_ref"] == "negative_no_effect"
    assert failure["target_projection"]["oracle_ref"] == "unknown"
    assert negative["target_projection"]["negative_control_presence_ref"] == "matched_triplet"


def test_builder_quarantines_raw_context_or_target() -> None:
    source = _source_dataset()
    row = copy.deepcopy(source["records"][0])
    row["context_tokens"] = list(row["context_tokens"]) + ["raw_payload=bad"]
    source["records"] = [row]
    result = build(source, input_sha256="b" * 64)
    assert result["status"] == "blocked_incomplete"
    assert result["counts"]["records"] == 0
    assert result["counts"]["invalid_records"] == 1
