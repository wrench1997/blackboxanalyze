from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from app.pg139_parser_variant import SCHEMA_VERSION as PARSER_SCHEMA, alternate_tokens
from app.pg139_safety_value_head import ActionSafetyValueHead, CausalSafetyValuePolicy, SCHEMA_VERSION


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg139_value_head_and_parser_are_bounded_and_decoupled():
    assert SCHEMA_VERSION == "pg139-action-conditioned-safety-value-v1"
    assert PARSER_SCHEMA == "pg139-independent-parser-variant-v1"
    head = ActionSafetyValueHead(64)
    assert head.policy[-1].out_features == 7
    policy_logits, safety_logits = head(torch.zeros((2, 64)))
    assert tuple(policy_logits.shape) == (2, 7)
    assert tuple(safety_logits.shape) == (2, 7)
    assert CausalSafetyValuePolicy
    tokens = alternate_tokens([
        {
            "source_token_layers": [{"modality": "html", "tokens": [{"kind": "tag", "value": "form"}]}],
            "ir_layer": {"tokens": [{"slot_id": "failure.kind", "value": "unknown", "weight": 1.0}]},
        }
    ])
    assert len(tokens) < 384
    assert all("<script" not in token.casefold() for token in tokens)


def test_pg139_raw_value_guarded_and_parser_ood_are_separate():
    report = _load("pg139_value_head_loio_report_v1.json")
    assert report["hard_gates_passed"] is False
    assert report["training_eligible"] is False
    assert report["checks"]["value_unknown_abstain"] is True
    assert report["checks"]["value_nontrivial_action_rate"] is True
    assert report["checks"]["value_safety_floor"] is False
    assert report["checks"]["parser_ood_safety_floor"] is False
    assert report["promotion_checks"]["action_gain_in_both_loio_folds"] is False
    selected = report["selected_summary"]
    assert selected[0]["variant"] == "scratch_value"
    assert selected[0]["standard"]["pg127"]["raw"]["safety_compliance_rate"] == 0.75
    assert selected[0]["standard"]["pg127"]["value"]["safety_compliance_rate"] == 0.75
    assert selected[0]["standard"]["pg127"]["value_guarded"]["safety_compliance_rate"] == 0.916667
    assert selected[0]["parser_ood"]["pg127"]["value"]["safety_compliance_rate"] == 0.125
    assert selected[1]["variant"] == "pretrained_joint_value"
    assert selected[1]["standard"]["pg125"]["raw"]["safety_compliance_rate"] == 0.8125
    assert selected[1]["standard"]["pg125"]["value"]["safety_compliance_rate"] == 0.8125
    assert selected[1]["standard"]["pg125"]["value_guarded"]["safety_compliance_rate"] == 0.9375
    assert selected[1]["parser_ood"]["pg125"]["value"]["safety_compliance_rate"] == 0.375
    assert report["input_contract"]["typed_contract_used_as_training_target_only"] is True
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg139_report_hash_is_recomputable():
    report = _load("pg139_value_head_loio_report_v1.json")
    declared = report.pop("report_sha256")
    actual = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual
