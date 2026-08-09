"""Fresh PG-121 decoder with shape-hash buckets removed.

PG-120 showed that the PG-119 model's four anonymous shape hash buckets could
turn an implementation spelling into a shortcut.  PG-121 keeps the same
capacity and transition slots but forces those four features to zero in both
training and evaluation.  This is a representation change, not a capacity
increase or a family label.
"""

from __future__ import annotations

from typing import Any, Iterable

from .pg119_metadata_rule_ir_decoder import (
    FEATURE_DIM,
    PG119_DECISIONS,
    MetadataRuleIRDecisionDecoder,
    canonical_model_input,
    decision_index,
)
from .pg119_metadata_rule_ir_decoder import model_input_feature_vector as metadata_feature_vector


SCHEMA_VERSION = "pg121-shape-sanitized-rule-ir-decoder-v1"
SHAPE_HASH_START = 36
SHAPE_HASH_END = 40


def model_input_feature_vector(model_input: dict[str, Any], *, prior_inputs: Iterable[dict[str, Any]] = ()) -> list[float]:
    """Use PG-119's generic projection while removing shape hash buckets."""

    values = metadata_feature_vector(model_input, prior_inputs=prior_inputs)
    if len(values) != FEATURE_DIM:
        raise ValueError(f"unexpected PG-119 feature dimension: {len(values)}")
    values[SHAPE_HASH_START:SHAPE_HASH_END] = [0.0] * (SHAPE_HASH_END - SHAPE_HASH_START)
    return values


def shape_hash_slots_zeroed(features: list[float]) -> bool:
    return len(features) == FEATURE_DIM and all(float(value) == 0.0 for value in features[SHAPE_HASH_START:SHAPE_HASH_END])


__all__ = ["FEATURE_DIM", "PG119_DECISIONS", "SCHEMA_VERSION", "MetadataRuleIRDecisionDecoder", "canonical_model_input", "decision_index", "model_input_feature_vector", "shape_hash_slots_zeroed"]
