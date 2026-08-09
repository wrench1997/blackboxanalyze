"""Regression checks for the PG-380 abstract-reasoning and layered-entropy rules."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _rules() -> dict:
    with (ROOT / "research" / "improvement_rules.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_entropy_hard_gate_is_first_stage_only_and_later_layers_are_diagnostic() -> None:
    rules = _rules()
    preservation = rules["research_goal_v2"]["information_preservation_contract"]
    limits = preservation["compression_limits"]

    assert limits["entropy_hard_gate_scope"] == "first_stage_next_token_pretraining_only"
    assert limits["max_drop_from_fixed_holdout_predictive_entropy"] == 0.25
    later = limits["later_layer_entropy"]
    assert later["role"] == "diagnostic_only"
    assert later["must_report"] is True
    assert later["hard_gate"] is False
    assert {
        "finite_logits",
        "non_empty_class_support",
        "no_holdout_leakage",
        "slot_coverage",
        "ASK_repair_negative_gates",
    }.issubset(set(later["required_checks"]))


def test_pg380_is_abstract_only_and_never_authorizes_raw_payload_or_promotion() -> None:
    rules = _rules()
    pg380 = rules["pg361_payload_shape_syntax_slot_contract"]["pg380_abstract_adversarial_reasoning"]

    assert pg380["status"] == "completed_abstract_adversarial_candidate_only"
    assert pg380["abstract_training"]["abstract_reasoning_sft_candidate_allowed"] is True
    assert pg380["abstract_training"]["capability_training_allowed"] is False
    assert pg380["abstract_training"]["training_eligible"] == 0
    assert pg380["safety"]["raw_payload_in_context"] is False
    assert pg380["safety"]["concrete_wire"] == "evaluator_template_only"
    assert all(value is False for value in pg380["promotion"].values())


def test_raw_payload_generation_policy_remains_evaluator_only() -> None:
    policy = _rules()["pg361_payload_shape_syntax_slot_contract"]["raw_payload_generation_policy"]
    assert policy["model_output"] == "abstract Rule-IR slots and allowlisted variant references only"
    assert policy["raw_generation_authority"] == "evaluator_only; model cannot supply arbitrary literal bytes"
    assert policy["raw_string_training"] is False
    assert policy["arbitrary_target"] is False
    assert policy["fresh_replay_required"] is True


def test_pg385_abstract_adversarial_lane_allows_only_reviewed_last_hop_binding() -> None:
    rules = _rules()
    lane = rules["pg361_payload_shape_syntax_slot_contract"]["pg385_abstract_adversarial_evaluator_fast_lane"]

    assert lane["status"] == "active_candidate_only"
    assert lane["dataset_scope"]["methods"] == ["GET", "POST"]
    assert lane["dataset_scope"]["roles"] == ["candidate", "reference", "negative", "replay"]
    assert lane["dataset_scope"]["implementation_holdout"] is True
    assert lane["model_output"]["raw_literal_bytes"] is False
    assert lane["model_output"]["arbitrary_waf_bypass_string"] is False
    assert lane["model_output"]["arbitrary_target_or_external_callback"] is False
    assert lane["concrete_binding"]["authority"] == "reviewed evaluator last hop only"
    assert lane["concrete_binding"]["benign_canary_wire_allowed"] is True
    assert lane["concrete_binding"]["model_can_emit_raw_canary"] is False
    assert lane["concrete_binding"]["evaluator_may_bind_canary_last_hop"] is True
    assert "no external callback" in lane["concrete_binding"]["benign_canary_wire_scope"]
    assert lane["concrete_binding"]["candidate_reference_negative_replay_typed"] is True
    assert lane["stateful_evaluator"]["state_delta_is_evaluator_only"] is True
    assert lane["entropy_and_compression"]["nested_compression_or_adapter_entropy"] == "diagnostic_only; do not block solely on this metric"
    assert all(value is False for value in lane["promotion"].values())
