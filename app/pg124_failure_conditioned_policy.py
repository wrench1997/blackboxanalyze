"""A bounded next-action policy for PG-124 failure-feedback ablation.

This is intentionally separate from the Rule IR decision decoder.  It learns
the next *safe abstract replay action* from the current projection and a
sanitized failure token.  Oracle authority, evidence hashes, raw probes,
response bodies, target ids and family names are excluded from the vector.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .pg123_authorization_rule_ir_decoder import model_input_feature_vector as pg123_feature_vector


SCHEMA_VERSION = "pg124-failure-conditioned-policy-v1"
FAILURE_KINDS = (
    "typed_positive",
    "candidate_without_typed_effect",
    "oracle_unavailable",
    "method_disagreement",
    "no_surface_delta",
    "budget_exhausted",
)
FAILED_GATES = (
    "typed_effect",
    "cross_channel_consistency",
    "matched_negative_control",
    "surface_delta",
    "bounded_probe_budget",
)
POLICY_ACTIONS = (
    "replay_other_method",
    "repeat_matched_negative_pair",
    "probe_candidate_other_method",
    "abstain_candidate_only",
    "abstain_unknown_oracle",
    "stop_confirmed_positive",
    "abstain_budget_exhausted",
)
PG123_FEATURE_DIM = 52
FAILURE_FEATURE_DIM = len(FAILURE_KINDS) + len(FAILED_GATES) + 2 + 4
FEATURE_DIM = PG123_FEATURE_DIM + FAILURE_FEATURE_DIM


def _one_hot(value: str, values: Sequence[str]) -> list[float]:
    return [float(value == item) for item in values]


def failure_feedback_vector(signature: Mapping[str, Any]) -> list[float]:
    """Project only bounded failure feedback; mask authority/oracle fields."""

    kind = str(signature.get("kind", ""))
    gate = str(signature.get("failed_gate", ""))
    method = str(signature.get("observed_method", "")).upper()
    methods = {str(item).upper() for item in signature.get("methods_seen", []) if str(item).upper() in {"GET", "POST"}}
    values = _one_hot(kind, FAILURE_KINDS)
    values.extend(_one_hot(gate, FAILED_GATES))
    values.extend([float(method == "GET"), float(method == "POST")])
    values.extend([
        float(bool(signature.get("candidate_signal"))),
        float(len(methods) >= 2),
        float("GET" in methods),
        float("POST" in methods),
    ])
    if len(values) != FAILURE_FEATURE_DIM:
        raise AssertionError("PG-124 failure feature dimension drift")
    return values


def policy_feature_vector(model_input: Mapping[str, Any], signature: Mapping[str, Any], *, prior_inputs: Sequence[Mapping[str, Any]] = (), failure_enabled: bool = True) -> list[float]:
    base = pg123_feature_vector(dict(model_input), prior_inputs=[dict(item) for item in prior_inputs])
    if len(base) != PG123_FEATURE_DIM:
        raise ValueError(f"unexpected PG-123 feature dimension: {len(base)}")
    feedback = failure_feedback_vector(signature) if failure_enabled else [0.0] * FAILURE_FEATURE_DIM
    return base + feedback


class FailureConditionedActionPolicy(nn.Module):
    """Small fresh MLP for abstract action selection; no Rule IR weights reused."""

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(hidden_dim, len(POLICY_ACTIONS))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(features))


def policy_index(action: str) -> int:
    if action not in POLICY_ACTIONS:
        raise ValueError(f"unknown PG-124 action: {action}")
    return POLICY_ACTIONS.index(action)


__all__ = [
    "FAILED_GATES",
    "FAILURE_FEATURE_DIM",
    "FAILURE_KINDS",
    "FEATURE_DIM",
    "FailureConditionedActionPolicy",
    "PG123_FEATURE_DIM",
    "POLICY_ACTIONS",
    "SCHEMA_VERSION",
    "failure_feedback_vector",
    "policy_feature_vector",
    "policy_index",
]
