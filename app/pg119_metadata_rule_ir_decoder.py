"""Fresh Rule IR decoder with location and metadata transition slots.

PG-119 adds only bounded, family-free observable metadata deltas to the
PG-118 projection.  The evaluator label, target/family identifiers, hashes,
and raw probe/response values remain outside the model input.
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


PG119_DECISIONS = PG115_DECISIONS
BASE_FEATURE_DIM = 40
LOCATION_SLOT_DIM = 4
METADATA_SLOT_DIM = 4
FEATURE_DIM = BASE_FEATURE_DIM + LOCATION_SLOT_DIM + METADATA_SLOT_DIM
METADATA_SLOT_START = BASE_FEATURE_DIM + LOCATION_SLOT_DIM
SCHEMA_VERSION = "pg119-metadata-rule-ir-decoder-v1"

for _rule in RULE_IR_BY_DECISION.values():
    validate_abstract_rule_ir(_rule)


def canonical_model_input(model_input: dict[str, Any]) -> dict[str, Any]:
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
            "metadata_changed": bool(response.get("metadata_changed")),
            "transition_delta": response.get("transition_delta") if response.get("transition_delta") in {"location", "metadata", "none"} else "unknown",
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
    """Return 40 established features plus location and metadata delta slots."""

    base = base_feature_vector(model_input, prior_inputs=prior_inputs)
    if len(base) != BASE_FEATURE_DIM:
        raise ValueError(f"unexpected PG-115 base feature dimension: {len(base)}")
    response = model_input.get("response_projection") or {}
    prior = list(prior_inputs)
    prior_responses = [row.get("response_projection") or {} for row in prior]
    method = str((model_input.get("action_manifest") or {}).get("method", "")).upper()
    method_changed = any(str((row.get("action_manifest") or {}).get("method", "")).upper() != method for row in prior)
    current_location = bool(response.get("location_changed"))
    prior_location = any(bool(row.get("location_changed")) for row in prior_responses)
    current_metadata = bool(response.get("metadata_changed"))
    prior_metadata = any(bool(row.get("metadata_changed")) for row in prior_responses)
    base.extend([
        float(current_location),
        float(prior_location),
        float(current_location and method_changed),
        float(current_location and bool(response.get("candidate_signal"))),
        float(current_metadata),
        float(prior_metadata),
        float(current_metadata and method_changed),
        float(current_metadata and bool(response.get("candidate_signal"))),
    ])
    return base


class MetadataRuleIRDecisionDecoder(nn.Module):
    """Fresh GPU-friendly MLP; no PG-118 weights are reused."""

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_dim: int = 48):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(hidden_dim, len(PG119_DECISIONS))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(features))

    @torch.inference_mode()
    def decode(self, features: torch.Tensor) -> list[dict[str, Any]]:
        probabilities = torch.softmax(self(features), dim=-1)
        values, indices = probabilities.max(dim=-1)
        outputs: list[dict[str, Any]] = []
        for confidence, index, row in zip(values.detach().cpu(), indices.detach().cpu(), probabilities.detach().cpu()):
            decision = PG119_DECISIONS[int(index)]
            outputs.append({"decision": decision, "confidence": round(float(confidence), 6), "rule_ir": RULE_IR_BY_DECISION[decision], "probabilities": {name: round(float(probability), 6) for name, probability in zip(PG119_DECISIONS, row)}})
        return outputs


def decision_index(decision: str) -> int:
    if decision not in PG119_DECISIONS:
        raise ValueError(f"unknown PG-119 decision: {decision}")
    return PG119_DECISIONS.index(decision)


def ablate_metadata_slots(features: list[float]) -> list[float]:
    """Zero only the new slots for a causal representation ablation."""

    if len(features) != FEATURE_DIM:
        raise ValueError(f"expected {FEATURE_DIM} features, got {len(features)}")
    values = list(features)
    for index in range(METADATA_SLOT_START, FEATURE_DIM):
        values[index] = 0.0
    return values


__all__ = ["FEATURE_DIM", "METADATA_SLOT_START", "PG119_DECISIONS", "SCHEMA_VERSION", "MetadataRuleIRDecisionDecoder", "ablate_metadata_slots", "canonical_model_input", "decision_index", "model_input_feature_vector"]
