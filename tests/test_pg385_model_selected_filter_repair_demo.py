from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_pg385_model_selected_filter_repair_demo import (
    _context_from_projection,
    _rule_from_prediction,
    run_demo,
)

ROOT = Path(__file__).resolve().parents[1]


def test_context_projection_is_abstract_and_bounded() -> None:
    tokens = _context_from_projection({"filter_state": "filtered", "filter_class": "encoding_filter"})
    assert "filter_state=filtered" in tokens
    assert "filter_class=encoding_filter" in tokens
    assert not any(value.startswith(("http://", "https://", "payload=", "wire=")) for value in tokens)


def test_model_prediction_expands_only_to_abstract_rule_ir() -> None:
    rule = _rule_from_prediction({
        "question": "none",
        "encoding_ref": "double_layer_order_sensitive",
        "probe_variant_ref": "one_variable_repair",
        "next_action": "repair",
        "repair_action": "encoding",
        "safe_to_send": "1",
    })
    assert rule["encoding_ref"] == "double_layer_order_sensitive"
    assert rule["probe_variant_ref"] == "one_variable_repair"
    assert "raw_value" not in rule
    assert "payload" not in rule


@pytest.mark.skipif(not (ROOT / "artifacts/pg385-variant-selector/pg385_variant_seed_38503.pt").exists(), reason="A800 candidate checkpoint is not present")
def test_a800_selector_checkpoint_drives_local_typed_replay() -> None:
    report, wires = run_demo(checkpoint=ROOT / "artifacts/pg385-variant-selector/pg385_variant_seed_38503.pt", show_wire=True)
    assert report["status"] == "completed_model_selected_filter_repair_loopback_only"
    assert report["counts"] == {
        "baseline_filtered": 1,
        "model_variant_selected": 1,
        "candidate_typed": 1,
        "reference_typed": 1,
        "negative_violation": 0,
        "replay_typed": 1,
    }
    assert report["model_boundary"]["model_emits_raw_string"] is False
    assert len(wires) == 5
