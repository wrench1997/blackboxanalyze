from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _rules() -> dict:
    return json.loads((ROOT / "research" / "improvement_rules.json").read_text(encoding="utf-8"))


def test_runtime_wire_rule_keeps_real_generation_at_evaluator_boundary() -> None:
    rule = _rules()["pg350_runtime_wire_generation"]
    assert rule["evaluator_binding"]["source_attested_template_required"] is True
    assert rule["evaluator_binding"]["single_runtime_marker_required"] is True
    assert rule["evaluator_binding"]["allowlisted_methods"] == ["GET", "POST"]
    lifecycle = rule["raw_string_lifecycle"]
    assert lifecycle["generate_at"] == "evaluator_last_hop_only"
    assert lifecycle["model_context"] is False
    assert lifecycle["training_records"] is False
    assert lifecycle["long_term_memory"] is False
    assert all(value is False for value in rule["promotion"].values())


def test_pg351_candidate_report_is_not_payload_capability_promotion() -> None:
    rule = _rules()["pg351_ask_oracle_composition_candidate"]
    assert rule["counts"]["training_eligible"] == 0
    assert rule["counts"]["raw_payload_in_context"] is False
    assert rule["worst_seed_observation"]["negative_false_allow_max"] == 0
    assert rule["worst_seed_observation"]["repair_recall_min"] == 0.0
    assert rule["worst_seed_observation"]["positive_action_recall_min"] == 0.0
    assert all(value is False for value in rule["promotion"].values())
