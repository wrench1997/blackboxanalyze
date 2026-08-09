"""PG-126 surface-invariant policy using only sanitized failure feedback."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from .pg124_failure_conditioned_policy import (
    FAILED_GATES,
    FAILURE_FEATURE_DIM,
    FAILURE_KINDS,
    POLICY_ACTIONS,
    failure_feedback_vector,
    policy_index,
)


SCHEMA_VERSION = "pg126-failure-only-policy-v1"
FEATURE_DIM = FAILURE_FEATURE_DIM


def failure_only_feature_vector(signature: Mapping[str, Any], *, enabled: bool = True) -> list[float]:
    return failure_feedback_vector(signature) if enabled else [0.0] * FEATURE_DIM


class FailureOnlyActionPolicy(nn.Module):
    """Fresh action policy with no response-surface features."""

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


__all__ = ["FEATURE_DIM", "FailureOnlyActionPolicy", "SCHEMA_VERSION", "failure_only_feature_vector", "policy_index"]
