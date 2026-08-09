from __future__ import annotations

import json
from pathlib import Path

from scripts.build_pg359_context_index_dataset import INDEX_BEGIN, INDEX_END, build, context_index


ROOT = Path(__file__).resolve().parents[1]


def _dataset() -> dict:
    return json.loads((ROOT / "research" / "pg351_ask_oracle_composition_dataset_v2.json").read_text(encoding="utf-8"))


def test_context_index_is_append_only_and_does_not_read_target() -> None:
    source = _dataset()
    result = build(source, input_sha256="a" * 64, input_path="source.json")
    assert result["status"] == "diagnostic_candidate_only"
    assert result["counts"]["records"] == len(source["records"])
    row = result["records"][0]
    original = source["records"][0]["context_tokens"]
    assert row["context_tokens"][: len(original)] == original
    assert row["context_tokens"][len(original)] == INDEX_BEGIN
    assert row["context_tokens"][-1] == INDEX_END
    assert row["context_index"]["target_tokens_read"] is False
    assert row["context_index"]["evaluator_sidecar_read"] is False


def test_context_index_has_observation_only_tokens_and_no_raw_fragments() -> None:
    index = context_index(["belief_typed_available=present", "belief_fresh_reset=unknown", "belief_process_step=failure", "failure_failure_class=blocked_variant", "javascript_presence=observed"])
    assert index[0] == INDEX_BEGIN
    assert index[-1] == INDEX_END
    assert "index_typed=present" in index
    assert "index_fresh=unknown" in index
    assert "index_process=failure" in index
    assert "index_failure=blocked_variant" in index
    assert "index_javascript_axis=observed" in index
    assert all("payload=" not in token and "wire=" not in token for token in index)


def test_invalid_raw_source_is_quarantined() -> None:
    source = _dataset()
    source["records"][0]["context_tokens"] = ["raw_payload=bad"]
    result = build(source, input_sha256="b" * 64, input_path="source.json")
    assert result["status"] == "blocked_incomplete"
    assert result["counts"]["invalid_records"] >= 1
    assert result["promotion"]["training_allowed"] is False
