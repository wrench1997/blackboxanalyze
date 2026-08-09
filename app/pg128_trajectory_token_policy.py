"""Trajectory-token policy for failure-guided exploration.

Each observed step contributes one sanitized failure token.  The token has a
bounded feature vector from PG-127's key-feature assembly and an explicit
trajectory weight.  The weight is derived only from recency, unresolved-gate
focus and remaining probe budget; it is not a disguised oracle label.  A
small GRU consumes the prefix of tokens seen so far and predicts one
allow-listed abstract replay action.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .failure_guided_scheduler import key_feature_weights_for_signature
from .pg124_failure_conditioned_policy import POLICY_ACTIONS, policy_index
from .pg127_key_feature_policy import ASSEMBLY_FEATURE_DIM, failure_assembly_feature_vector


SCHEMA_VERSION = "pg128-trajectory-token-policy-v1"
MAX_TRAJECTORY_TOKENS = 6
TOKEN_FEATURE_DIM = ASSEMBLY_FEATURE_DIM + 1
HIDDEN_DIM = 64
_WEIGHTED = "weighted"
_UNIFORM_TOKENS = "uniform_tokens"
_ZERO = "zero"
TRAJECTORY_MODES = (_WEIGHTED, _UNIFORM_TOKENS, _ZERO)


def _token_priority(signature: Mapping[str, Any], *, position: int, total: int) -> float:
    """Compute a causal, bounded priority before normalization."""

    focus = key_feature_weights_for_signature(signature)
    recency = 1.0 + float(position) / float(max(total - 1, 1))
    unresolved_gate = 1.0 + 1.5 * focus["failed_gate"] + 0.75 * focus["candidate_signal"]
    budget_signal = 1.0 + min(float(signature.get("remaining_probe_budget", 0) or 0), 4.0) / 4.0 * focus["probe_budget"]
    return recency * unresolved_gate * budget_signal


def trajectory_token_weights(signatures: Sequence[Mapping[str, Any]], *, mode: str = _WEIGHTED) -> list[float]:
    """Return one normalized weight per observed failure token."""

    if mode not in TRAJECTORY_MODES:
        raise ValueError(f"unknown PG-128 trajectory mode: {mode}")
    if not signatures:
        return []
    if mode == _ZERO:
        return [0.0 for _ in signatures]
    if mode == _UNIFORM_TOKENS:
        value = 1.0 / float(len(signatures))
        return [value for _ in signatures]
    priorities = [_token_priority(signature, position=index, total=len(signatures)) for index, signature in enumerate(signatures)]
    total = sum(priorities)
    if total <= 0.0:  # pragma: no cover - defensive, priorities are positive
        return [1.0 / float(len(signatures)) for _ in signatures]
    return [round(priority / total, 6) for priority in priorities]


def trajectory_feature_matrix(signatures: Sequence[Mapping[str, Any]], *, mode: str = _WEIGHTED) -> list[list[float]]:
    """Build a fixed six-token prefix matrix; future/padding tokens are zero."""

    if len(signatures) > MAX_TRAJECTORY_TOKENS:
        raise ValueError("PG-128 trajectory exceeds the bounded token window")
    weights = trajectory_token_weights(signatures, mode=mode)
    rows: list[list[float]] = []
    for signature, token_weight in zip(signatures, weights):
        if mode == _ZERO:
            rows.append([0.0] * TOKEN_FEATURE_DIM)
            continue
        assembled = failure_assembly_feature_vector(signature, mode="weighted")
        rows.append(assembled + [float(token_weight)])
    rows.extend([[0.0] * TOKEN_FEATURE_DIM for _ in range(MAX_TRAJECTORY_TOKENS - len(rows))])
    if len(rows) != MAX_TRAJECTORY_TOKENS or any(len(row) != TOKEN_FEATURE_DIM for row in rows):
        raise AssertionError("PG-128 trajectory feature shape drift")
    return rows


class TrajectoryTokenActionPolicy(nn.Module):
    """Order-sensitive policy with explicit token-weighted context pooling."""

    def __init__(self, token_feature_dim: int = TOKEN_FEATURE_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.token_encoder = nn.Sequential(
            nn.Linear(token_feature_dim, 48),
            nn.LayerNorm(48),
            nn.GELU(),
        )
        self.sequence = nn.GRU(48, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(POLICY_ACTIONS)),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != TOKEN_FEATURE_DIM:
            raise ValueError("PG-128 tokens must have shape [batch, tokens, feature_dim]")
        encoded = self.token_encoder(tokens)
        sequence, _ = self.sequence(encoded)
        weights = tokens[..., -1].clamp_min(0.0)
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = (sequence * weights.unsqueeze(-1)).sum(dim=1) / denominator
        lengths = (weights > 0.0).sum(dim=1).clamp(min=1, max=tokens.shape[1]) - 1
        last = sequence[torch.arange(sequence.shape[0], device=sequence.device), lengths]
        return self.classifier(torch.cat([pooled, last], dim=-1))


def policy_index_for_trajectory(action: str) -> int:
    return policy_index(action)


__all__ = [
    "HIDDEN_DIM",
    "MAX_TRAJECTORY_TOKENS",
    "SCHEMA_VERSION",
    "TOKEN_FEATURE_DIM",
    "TRAJECTORY_MODES",
    "TrajectoryTokenActionPolicy",
    "policy_index_for_trajectory",
    "trajectory_feature_matrix",
    "trajectory_token_weights",
]
