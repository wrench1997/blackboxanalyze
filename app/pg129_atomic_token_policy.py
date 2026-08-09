"""PG-129 atomic tokenization of failure trajectories.

The previous trajectory experiment represented one whole failure signature as
one dense token.  This module makes the representation explicit: every step
is decomposed into six bounded atomic tokens.  Each token carries its
categorical value, scheduler group weight, trajectory weight, normalized
position and a current-step flag.  Raw probes, response bodies, oracle
authority, target ids and family names never become tokens.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .failure_guided_scheduler import key_feature_weights_for_signature
from .pg124_failure_conditioned_policy import (
    FAILED_GATES,
    FAILURE_KINDS,
    POLICY_ACTIONS,
    policy_index,
)
from .pg128_trajectory_token_policy import MAX_TRAJECTORY_TOKENS


SCHEMA_VERSION = "pg129-atomic-token-policy-v1"
ATOMIC_GROUPS = ("failure_kind", "failed_gate", "candidate_signal", "observed_method", "methods_seen", "probe_budget")
GROUP_VALUE_TABLES = {
    "failure_kind": tuple(FAILURE_KINDS),
    "failed_gate": tuple(FAILED_GATES),
    "candidate_signal": ("false", "true"),
    "observed_method": ("", "GET", "POST"),
    "methods_seen": ("", "GET", "POST", "GET+POST"),
    "probe_budget": ("0", "1", "2", "3", "4+"),
}
TYPE_DIM = len(ATOMIC_GROUPS)
VALUE_DIM = max(len(values) for values in GROUP_VALUE_TABLES.values())
SCALAR_DIM = 4  # group focus, trajectory weight, token weight, position/current flag
ATOMIC_TOKEN_FEATURE_DIM = TYPE_DIM + VALUE_DIM + SCALAR_DIM
MAX_ATOMIC_TOKENS = MAX_TRAJECTORY_TOKENS * len(ATOMIC_GROUPS)
HIDDEN_DIM = 64
_WEIGHTED = "weighted"
_UNIFORM_TOKENS = "uniform_tokens"
_ZERO = "zero"
ATOMIC_MODES = (_WEIGHTED, _UNIFORM_TOKENS, _ZERO)
FAILURE_TOKEN_KINDS = frozenset({"candidate_without_typed_effect", "oracle_unavailable", "method_disagreement", "budget_exhausted"})


def is_failure_token(signature: Mapping[str, Any]) -> bool:
    """Only unresolved failure observations receive a focused weight."""

    kind = str(signature.get("kind", ""))
    if kind in FAILURE_TOKEN_KINDS:
        return True
    return kind == "no_surface_delta" and str(signature.get("failed_gate", "")) != "matched_negative_control"


def _forward_group_weights() -> dict[str, float]:
    return {group: 1.0 / float(len(ATOMIC_GROUPS)) for group in ATOMIC_GROUPS}


def _dynamic_temporal_weights(signatures: Sequence[Mapping[str, Any]], *, mode: str) -> list[float]:
    if not signatures:
        return []
    base = [1.0 + float(index) / float(max(len(signatures) - 1, 1)) for index in range(len(signatures))]
    if mode == _UNIFORM_TOKENS:
        total = sum(base)
        return [value / total for value in base]
    priorities: list[float] = []
    for index, signature in enumerate(signatures):
        priority = base[index]
        if is_failure_token(signature):
            focus = key_feature_weights_for_signature(signature)
            priority *= 1.0 + 2.0 * max(focus["failed_gate"], focus["candidate_signal"], focus["probe_budget"])
        # A forward token after a failure deliberately uses only the baseline
        # priority; the failure boost is not carried into the next inference.
        priorities.append(priority)
    total = sum(priorities)
    return [value / total for value in priorities]


def _methods_value(signature: Mapping[str, Any]) -> str:
    methods = {str(item).upper() for item in signature.get("methods_seen", []) if str(item).upper() in {"GET", "POST"}}
    return "GET+POST" if methods == {"GET", "POST"} else next(iter(methods), "") if methods else ""


def _group_value(group: str, signature: Mapping[str, Any]) -> str:
    if group == "failure_kind":
        return str(signature.get("kind", ""))
    if group == "failed_gate":
        return str(signature.get("failed_gate", ""))
    if group == "candidate_signal":
        return "true" if bool(signature.get("candidate_signal")) else "false"
    if group == "observed_method":
        return str(signature.get("observed_method", "")).upper()
    if group == "methods_seen":
        return _methods_value(signature)
    remaining = max(0, int(signature.get("remaining_probe_budget", 0) or 0))
    return str(min(remaining, 4)) if remaining < 4 else "4+"


def _scalar(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def atomic_trajectory_matrix(signatures: Sequence[Mapping[str, Any]], *, mode: str = _WEIGHTED) -> list[list[float]]:
    """Tokenize a prefix into a fixed 36-token matrix."""

    if mode not in ATOMIC_MODES:
        raise ValueError(f"unknown PG-129 atomic mode: {mode}")
    if len(signatures) > MAX_TRAJECTORY_TOKENS:
        raise ValueError("PG-129 trajectory exceeds the bounded token window")
    temporal_weights = _dynamic_temporal_weights(signatures, mode=mode) if mode != _ZERO else [0.0 for _ in signatures]
    rows: list[list[float]] = []
    for position, (signature, temporal_weight) in enumerate(zip(signatures, temporal_weights)):
        focus = key_feature_weights_for_signature(signature) if is_failure_token(signature) else _forward_group_weights()
        normalized_position = float(position) / float(max(MAX_TRAJECTORY_TOKENS - 1, 1))
        current_flag = 1.0 if position == len(signatures) - 1 else 0.0
        for group_index, group in enumerate(ATOMIC_GROUPS):
            type_vector = [1.0 if index == group_index else 0.0 for index in range(TYPE_DIM)]
            values = GROUP_VALUE_TABLES[group]
            value = _group_value(group, signature)
            value_vector = [1.0 if item == value else 0.0 for item in values]
            value_vector.extend([0.0] * (VALUE_DIM - len(value_vector)))
            group_focus = float(focus[group])
            token_weight = group_focus * float(temporal_weight)
            if mode == _ZERO:
                rows.append([0.0] * ATOMIC_TOKEN_FEATURE_DIM)
            else:
                rows.append(type_vector + value_vector + [_scalar(group_focus), _scalar(float(temporal_weight)), _scalar(token_weight), _scalar(normalized_position + 0.5 * current_flag)])
    rows.extend([[0.0] * ATOMIC_TOKEN_FEATURE_DIM for _ in range(MAX_ATOMIC_TOKENS - len(rows))])
    if len(rows) != MAX_ATOMIC_TOKENS or any(len(row) != ATOMIC_TOKEN_FEATURE_DIM for row in rows):
        raise AssertionError("PG-129 atomic token matrix shape drift")
    return rows


class AtomicTokenActionPolicy(nn.Module):
    """Small transformer over atomic failure tokens."""

    def __init__(self, feature_dim: int = ATOMIC_TOKEN_FEATURE_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(feature_dim, 48), nn.LayerNorm(48), nn.GELU())
        self.position = nn.Embedding(MAX_ATOMIC_TOKENS, 48)
        layer = nn.TransformerEncoderLayer(d_model=48, nhead=4, dim_feedforward=96, dropout=0.0, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.classifier = nn.Sequential(nn.Linear(48 * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, len(POLICY_ACTIONS)))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1] != MAX_ATOMIC_TOKENS or tokens.shape[-1] != ATOMIC_TOKEN_FEATURE_DIM:
            raise ValueError("PG-129 tokens must have shape [batch, max_atomic_tokens, feature_dim]")
        positions = torch.arange(MAX_ATOMIC_TOKENS, device=tokens.device).unsqueeze(0)
        encoded = self.embedding(tokens) + self.position(positions)
        padding = tokens.abs().sum(dim=-1) <= 1e-8
        # Transformer attention is undefined when every position is masked;
        # keep one inert sentinel for the capacity-matched zero ablation.
        all_padding = padding.all(dim=1)
        if bool(all_padding.any()):
            padding = padding.clone()
            padding[all_padding, 0] = False
        encoded = self.transformer(encoded, src_key_padding_mask=padding)
        token_weights = tokens[..., TYPE_DIM + VALUE_DIM + 2].clamp_min(0.0)
        denominator = token_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = (encoded * token_weights.unsqueeze(-1)).sum(dim=1) / denominator
        current = tokens[..., -1].clamp_min(0.0)
        current_denominator = current.sum(dim=1, keepdim=True).clamp_min(1e-6)
        current_context = (encoded * current.unsqueeze(-1)).sum(dim=1) / current_denominator
        return self.classifier(torch.cat([pooled, current_context], dim=-1))


def policy_index_for_atomic(action: str) -> int:
    return policy_index(action)


__all__ = [
    "ATOMIC_GROUPS",
    "ATOMIC_MODES",
    "ATOMIC_TOKEN_FEATURE_DIM",
    "AtomicTokenActionPolicy",
    "GROUP_VALUE_TABLES",
    "FAILURE_TOKEN_KINDS",
    "HIDDEN_DIM",
    "MAX_ATOMIC_TOKENS",
    "SCHEMA_VERSION",
    "TYPE_DIM",
    "VALUE_DIM",
    "atomic_trajectory_matrix",
    "is_failure_token",
    "policy_index_for_atomic",
]
