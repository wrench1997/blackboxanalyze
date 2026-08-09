"""Fresh Rule IR decoder with a generic authorization-transition slot.

PG-122 showed that a typed authorization change was invisible to PG-121's
metadata/location representation.  PG-123 adds four bounded transition
features; it does not add a family label, oracle label, target id, hash, or raw
probe/body.  The checkpoint is always trained from scratch.
"""

from __future__ import annotations

from typing import Any, Iterable

from torch import nn

from .pg115_small_rule_ir_decoder import PG115_DECISIONS
from .pg119_metadata_rule_ir_decoder import RULE_IR_BY_DECISION, decision_index
from .pg121_shape_sanitized_rule_ir_decoder import (
    FEATURE_DIM as PG121_FEATURE_DIM,
    MetadataRuleIRDecisionDecoder,
    model_input_feature_vector as pg121_feature_vector,
)
from .rule_ir_decoder import validate_abstract_rule_ir


PG123_DECISIONS = PG115_DECISIONS
BASE_FEATURE_DIM = PG121_FEATURE_DIM
AUTHORIZATION_SLOT_DIM = 4
FEATURE_DIM = BASE_FEATURE_DIM + AUTHORIZATION_SLOT_DIM
AUTHORIZATION_SLOT_START = BASE_FEATURE_DIM
SCHEMA_VERSION = "pg123-authorization-rule-ir-decoder-v1"

for _rule in RULE_IR_BY_DECISION.values():
    validate_abstract_rule_ir(_rule)


def canonical_model_input(model_input: dict[str, Any]) -> dict[str, Any]:
    """Keep PG-121's projection but expose one generic transition category."""

    from .pg121_shape_sanitized_rule_ir_decoder import canonical_model_input as pg121_canonical

    result = pg121_canonical(model_input)
    raw_response = model_input.get("response_projection") or {}
    response = result["response_projection"]
    response["authorization_changed"] = bool(raw_response.get("authorization_changed"))
    transition = raw_response.get("transition_delta")
    response["transition_delta"] = transition if transition in {"location", "metadata", "authorization", "none"} else "unknown"
    return result


def model_input_feature_vector(model_input: dict[str, Any], *, prior_inputs: Iterable[dict[str, Any]] = ()) -> list[float]:
    """Return PG-121's 48 sanitized features plus four auth slots."""

    current = canonical_model_input(model_input)
    prior = [canonical_model_input(value) for value in prior_inputs]
    values = pg121_feature_vector(current, prior_inputs=prior)
    if len(values) != BASE_FEATURE_DIM:
        raise ValueError(f"unexpected PG-121 feature dimension: {len(values)}")
    response = current.get("response_projection") or {}
    prior_responses = [row.get("response_projection") or {} for row in prior]
    method = str((current.get("action_manifest") or {}).get("method", "")).upper()
    method_changed = any(str((row.get("action_manifest") or {}).get("method", "")).upper() != method for row in prior)
    authorization_changed = bool(response.get("authorization_changed")) or response.get("transition_delta") == "authorization"
    prior_authorization = any(bool(row.get("authorization_changed")) or row.get("transition_delta") == "authorization" for row in prior_responses)
    values.extend([
        float(authorization_changed),
        float(prior_authorization),
        float(authorization_changed and method_changed),
        float(authorization_changed and bool(response.get("candidate_signal"))),
    ])
    return values


class AuthorizationRuleIRDecisionDecoder(MetadataRuleIRDecisionDecoder):
    """Fresh MLP; inherited module shape is initialized with 52 inputs."""

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_dim: int = 48):
        super().__init__(feature_dim=feature_dim, hidden_dim=hidden_dim)


__all__ = [
    "AUTHORIZATION_SLOT_DIM",
    "AUTHORIZATION_SLOT_START",
    "AuthorizationRuleIRDecisionDecoder",
    "FEATURE_DIM",
    "PG123_DECISIONS",
    "SCHEMA_VERSION",
    "canonical_model_input",
    "decision_index",
    "model_input_feature_vector",
]
