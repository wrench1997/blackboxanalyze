"""PG-127 failure-assembly policy.

The policy is deliberately not a vulnerability classifier.  It learns the
next bounded replay action from the *failure information* left by the last
step.  The scheduler assigns an auditable focus weight to six generic groups
(``failure_kind``, ``failed_gate``, ``candidate_signal``, ``observed_method``,
``methods_seen`` and ``probe_budget``); this module uses those weights to
assemble the input presented to a fresh action head.

No oracle authority, evidence hash, target id, raw probe, response body or
family label is represented in the model vector.  The evidence remains in the
local replay trace and is never a training shortcut.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from .failure_guided_scheduler import key_feature_weights_for_signature
from .pg124_failure_conditioned_policy import (
    FAILURE_FEATURE_DIM,
    POLICY_ACTIONS,
    failure_feedback_vector,
    policy_index,
)


SCHEMA_VERSION = "pg127-key-feature-assembly-policy-v1"
ASSEMBLY_FEATURE_DIM = FAILURE_FEATURE_DIM + 1
MAX_PROBE_BUDGET = 16.0
_WEIGHTED = "weighted"
_UNIFORM = "uniform"
_ZERO = "zero"
ASSEMBLY_MODES = (_WEIGHTED, _UNIFORM, _ZERO)


def _bounded_budget(signature: Mapping[str, Any]) -> float:
    remaining = max(0, min(int(signature.get("remaining_probe_budget", 0) or 0), int(MAX_PROBE_BUDGET)))
    return float(remaining) / MAX_PROBE_BUDGET


def failure_assembly_feature_vector(signature: Mapping[str, Any], *, mode: str = _WEIGHTED) -> list[float]:
    """Assemble bounded failure groups with scheduler-derived focus weights.

    ``weighted`` is the experiment condition.  ``uniform`` keeps the same
    failure groups and budget but removes the focus gate.  ``zero`` is a
    capacity-matched ablation.  The feature layout is intentionally fixed so
    the three conditions cannot change model capacity or feature count.
    """

    if mode not in ASSEMBLY_MODES:
        raise ValueError(f"unknown PG-127 assembly mode: {mode}")
    if mode == _ZERO:
        return [0.0] * ASSEMBLY_FEATURE_DIM
    values = failure_feedback_vector(signature)
    weights = key_feature_weights_for_signature(signature) if mode == _WEIGHTED else {key: 1.0 for key in ("failure_kind", "failed_gate", "candidate_signal", "observed_method", "methods_seen", "probe_budget")}
    # failure_feedback_vector layout: 6 kind, 5 gate, 2 method, candidate,
    # methods_seen aggregate + GET + POST.
    assembled = list(values)
    assembled[0:6] = [value * weights["failure_kind"] for value in assembled[0:6]]
    assembled[6:11] = [value * weights["failed_gate"] for value in assembled[6:11]]
    assembled[11:13] = [value * weights["observed_method"] for value in assembled[11:13]]
    assembled[13:14] = [value * weights["candidate_signal"] for value in assembled[13:14]]
    assembled[14:17] = [value * weights["methods_seen"] for value in assembled[14:17]]
    assembled.append(_bounded_budget(signature) * weights["probe_budget"])
    if len(assembled) != ASSEMBLY_FEATURE_DIM:
        raise AssertionError("PG-127 failure assembly feature dimension drift")
    return assembled


class KeyFeatureAssemblyActionPolicy(nn.Module):
    """Fresh MLP that consumes only the assembled failure state."""

    def __init__(self, feature_dim: int = ASSEMBLY_FEATURE_DIM, hidden_dim: int = 64):
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


def policy_index_for_assembly(action: str) -> int:
    return policy_index(action)


__all__ = [
    "ASSEMBLY_FEATURE_DIM",
    "ASSEMBLY_MODES",
    "KeyFeatureAssemblyActionPolicy",
    "SCHEMA_VERSION",
    "failure_assembly_feature_vector",
    "policy_index_for_assembly",
]
