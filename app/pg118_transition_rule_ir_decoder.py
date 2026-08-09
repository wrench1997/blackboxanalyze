"""Fresh small Rule IR decoder with one generic transition-delta slot.

PG-118 keeps the PG-115/116 family- and oracle-blind features, then adds only
bounded observable transition facts (for example, a location delta).  It does
not add route names, target IDs, evaluator labels, or raw probe values.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn

from .pg115_small_rule_ir_decoder import (
    PG115_DECISIONS,
    RULE_IR_BY_DECISION,
    model_input_feature_vector as base_feature_vector,
)
from .rule_ir_decoder import validate_abstract_rule_ir


PG118_DECISIONS = PG115_DECISIONS
BASE_FEATURE_DIM = 40
FEATURE_DIM = 44
SCHEMA_VERSION = "pg118-transition-rule-ir-decoder-v1"

for _rule in RULE_IR_BY_DECISION.values():
    validate_abstract_rule_ir(_rule)


def canonical_model_input(model_input: dict[str, Any]) -> dict[str, Any]:
    """Persist PG-115 visible fields plus generic transition delta flags."""

    action = model_input.get("action_manifest") or {}
    baseline = model_input.get("baseline_projection") or {}
    response = model_input.get("response_projection") or {}
    belief = model_input.get("belief_before") or {}
    bsp = response.get("bsp_core_projection") or {}
    return {
        "action_manifest": {
            "method": action.get("method"),
            "placement": action.get("placement"),
            "encoding_chain": list(action.get("encoding_chain") or []),
            "safety": dict(action.get("safety") or {}),
        },
        "baseline_projection": {
            "body_length_bucket": baseline.get("body_length_bucket"),
            "status_class": baseline.get("status_class"),
        },
        "response_projection": {
            "candidate_signal": bool(response.get("candidate_signal")),
            "noise_bucket": int(response.get("noise_bucket", 0) or 0),
            "policy_header_changed": bool(response.get("policy_header_changed")),
            "shape_changed": bool(response.get("shape_changed")),
            "location_changed": bool(response.get("location_changed")),
            "transition_delta": response.get("transition_delta") if response.get("transition_delta") in {"location", "none"} else "unknown",
            "shape_class": response.get("shape_class"),
            "status_class": response.get("status_class"),
            "bsp_core_projection": {
                "leaf_mass_error": float(bsp.get("leaf_mass_error", 0.0) or 0.0),
                "selected_leaf_count": len(bsp.get("selected_leaf_ids") or []),
                "topology_version": int(bsp.get("topology_version", 0) or 0),
            },
        },
        "belief_before": {key: float(belief.get(key, 0.0) or 0.0) for key in ("effect", "input_only", "no_effect", "unknown")},
    }


def model_input_feature_vector(model_input: dict[str, Any], *, prior_inputs: Iterable[dict[str, Any]] = ()) -> list[float]:
    """Return the 40 anonymous PG-116 features plus four delta slots."""

    # The base extractor ignores unknown keys and remains the established
    # family-free projection.  Only the new bounded transition flags are added.
    base = base_feature_vector(model_input, prior_inputs=prior_inputs)
    if len(base) != BASE_FEATURE_DIM:
        raise ValueError(f"unexpected PG-115 base feature dimension: {len(base)}")
    response = model_input.get("response_projection") or {}
    prior = list(prior_inputs)
    prior_responses = [row.get("response_projection") or {} for row in prior]
    current_location = bool(response.get("location_changed"))
    prior_location = any(bool(row.get("location_changed")) for row in prior_responses)
    method = str((model_input.get("action_manifest") or {}).get("method", "")).upper()
    method_changed = any(str((row.get("action_manifest") or {}).get("method", "")).upper() != method for row in prior)
    base.extend([
        float(current_location),
        float(prior_location),
        float(current_location and method_changed),
        float(current_location and bool(response.get("candidate_signal"))),
    ])
    return base


class TransitionRuleIRDecisionDecoder(nn.Module):
    """Fresh GPU-friendly MLP; no PG-115/116 weights are reused."""

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_dim: int = 48):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(hidden_dim, len(PG118_DECISIONS))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(features))

    @torch.inference_mode()
    def decode(self, features: torch.Tensor) -> list[dict[str, Any]]:
        probabilities = torch.softmax(self(features), dim=-1)
        values, indices = probabilities.max(dim=-1)
        outputs: list[dict[str, Any]] = []
        for confidence, index, row in zip(values.detach().cpu(), indices.detach().cpu(), probabilities.detach().cpu()):
            decision = PG118_DECISIONS[int(index)]
            outputs.append({
                "decision": decision,
                "confidence": round(float(confidence), 6),
                "rule_ir": RULE_IR_BY_DECISION[decision],
                "probabilities": {name: round(float(probability), 6) for name, probability in zip(PG118_DECISIONS, row)},
            })
        return outputs


def decision_index(decision: str) -> int:
    if decision not in PG118_DECISIONS:
        raise ValueError(f"unknown PG-118 decision: {decision}")
    return PG118_DECISIONS.index(decision)


__all__ = ["FEATURE_DIM", "PG118_DECISIONS", "SCHEMA_VERSION", "TransitionRuleIRDecisionDecoder", "canonical_model_input", "decision_index", "model_input_feature_vector"]
