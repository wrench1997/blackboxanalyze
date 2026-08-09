from __future__ import annotations

import json

from app.pg385_filter_canary_fixture import evaluate_raw_request
from scripts.run_pg385_filter_repair_demo import run_demo


def test_fixture_rejects_single_layer_and_accepts_second_layer_canary() -> None:
    blocked = evaluate_raw_request(method="GET", raw_query="q=PG385_BASE_0001%3A")
    assert blocked["filter_state"] == "filtered"
    assert blocked["filter_class"] == "encoding_filter"
    assert blocked["typed_effect_confirmed"] is False

    accepted = evaluate_raw_request(method="GET", raw_query="q=PG385_CAND_0002%25253A")
    assert accepted["filter_state"] == "typed_effect"
    assert accepted["typed_effect_confirmed"] is True
    assert accepted["external_network"] is False


def test_repair_reasoner_changes_one_abstract_axis_and_negative_stays_clean() -> None:
    report, wires = run_demo(show_wire=False)
    assert report["status"] == "completed_filter_repair_loopback_only"
    assert report["counts"] == {
        "baseline_filtered": 1,
        "model_repair_selected": 1,
        "action_changed": 1,
        "candidate_typed": 1,
        "reference_typed": 1,
        "negative_violation": 0,
        "replay_typed": 1,
    }
    assert wires == []
    assert report["model_boundary"]["model_raw_value"] is False
    assert report["model_boundary"]["evaluator_last_hop_canary_binding"] is True
    text = json.dumps(report, ensure_ascii=False, sort_keys=True)
    assert "PG385_CAND_0002" not in text
    assert "GET http://" not in text


def test_show_wire_is_ephemeral_only_and_contains_bounded_local_marker() -> None:
    report, wires = run_demo(show_wire=True)
    assert report["counts"]["candidate_typed"] == 1
    assert len(wires) == 5
    assert all("127.0.0.1" in wire for wire in wires)
    assert any("%25253A" in wire for wire in wires)
    assert all("external" not in wire.casefold() for wire in wires)
