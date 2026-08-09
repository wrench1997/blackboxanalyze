"""PG-131 policy over layered Rule-IR tokens.

PG-130 established the compression contract.  PG-131 turns the resulting
Rule-IR slots into the actual sequence input for an action policy.  The model
sees slot/value tokens and their audited weights, never the source snapshot,
raw JavaScript, raw request/response, typed oracle authority or target
identity.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .pg124_failure_conditioned_policy import POLICY_ACTIONS, policy_index


SCHEMA_VERSION = "pg131-layered-ir-policy-v1"
IR_SLOT_IDS = (
    "surface.modalities",
    "transport.methods_seen",
    "response.transition_delta",
    "failure.kind",
    "failure.failed_gate",
    "failure.recovery_phase",
    "probe.remaining_budget",
    "trajectory.progress",
)
IR_VALUE_VOCAB = (
    "none",
    "html",
    "html+javascript+transport",
    "javascript",
    "transport",
    "GET",
    "POST",
    "GET+POST",
    "location",
    "metadata",
    "authorization",
    "visibility",
    "scope",
    "failure_adjusted",
    "forward_baseline",
    "0",
    "1-4",
    "5-16",
    "17+",
    "typed_positive",
    "candidate_without_typed_effect",
    "oracle_unavailable",
    "method_disagreement",
    "no_surface_delta",
    "budget_exhausted",
    "typed_effect",
    "cross_channel_consistency",
    "matched_negative_control",
    "surface_delta",
    "bounded_probe_budget",
    "step_1_of_4",
    "step_2_of_4",
    "step_3_of_4",
    "step_4_of_4",
    "step_1_of_6",
    "step_2_of_6",
    "step_3_of_6",
    "step_4_of_6",
    "step_5_of_6",
    "step_6_of_6",
    "unknown",
)
SLOT_DIM = len(IR_SLOT_IDS)
VALUE_DIM = len(IR_VALUE_VOCAB)
SCALAR_DIM = 4  # slot weight, token weight, current flag, step position
IR_TOKEN_FEATURE_DIM = SLOT_DIM + VALUE_DIM + SCALAR_DIM
MAX_STEPS = 6
TOKENS_PER_STEP = len(IR_SLOT_IDS)
MAX_IR_TOKENS = MAX_STEPS * TOKENS_PER_STEP
HIDDEN_DIM = 64
IR_MODES = ("weighted", "uniform", "no_failure_slots", "zero")


def _value_index(value: Any) -> int:
    text = str(value)
    try:
        return IR_VALUE_VOCAB.index(text)
    except ValueError:
        return IR_VALUE_VOCAB.index("unknown")


def layered_ir_token_matrix(prefix_layers: Sequence[Mapping[str, Any]], *, mode: str = "weighted") -> list[list[float]]:
    """Flatten a prefix of Rule-IR layers into a fixed token matrix."""

    if mode not in IR_MODES:
        raise ValueError(f"unknown PG-131 IR mode: {mode}")
    if len(prefix_layers) > MAX_STEPS:
        raise ValueError("PG-131 prefix exceeds the bounded IR token window")
    rows: list[list[float]] = []
    for step_index, layer in enumerate(prefix_layers):
        tokens = list((layer.get("tokens") or []))
        by_slot = {str(token.get("slot_id")): token for token in tokens}
        raw_weights: list[float] = []
        for slot in IR_SLOT_IDS:
            token = by_slot.get(slot) or {}
            weight = max(0.0, min(float(token.get("weight", 1.0)), 2.0))
            if mode == "uniform":
                weight = 1.0
            elif mode == "no_failure_slots" and slot.startswith("failure."):
                weight = 0.0
            elif mode == "zero":
                weight = 0.0
            raw_weights.append(weight)
        denominator = sum(raw_weights) or 1.0
        for slot_index, slot in enumerate(IR_SLOT_IDS):
            token = by_slot.get(slot) or {}
            value = token.get("value", "unknown")
            slot_weight = raw_weights[slot_index]
            token_weight = slot_weight / denominator
            type_vector = [1.0 if index == slot_index else 0.0 for index in range(SLOT_DIM)]
            value_vector = [1.0 if index == _value_index(value) else 0.0 for index in range(VALUE_DIM)]
            current = 1.0 if step_index == len(prefix_layers) - 1 else 0.0
            position = float(step_index) / float(max(MAX_STEPS - 1, 1))
            if mode == "zero":
                rows.append([0.0] * IR_TOKEN_FEATURE_DIM)
            else:
                rows.append(type_vector + value_vector + [slot_weight / 2.0, token_weight, current, position])
    rows.extend([[0.0] * IR_TOKEN_FEATURE_DIM for _ in range(MAX_IR_TOKENS - len(rows))])
    if len(rows) != MAX_IR_TOKENS or any(len(row) != IR_TOKEN_FEATURE_DIM for row in rows):
        raise AssertionError("PG-131 IR token shape drift")
    return rows


class LayeredIRActionPolicy(nn.Module):
    """Transformer action head over a bounded prefix of Rule-IR tokens."""

    def __init__(self, feature_dim: int = IR_TOKEN_FEATURE_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(feature_dim, 64), nn.LayerNorm(64), nn.GELU())
        self.position = nn.Embedding(MAX_IR_TOKENS, 64)
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.0, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.classifier = nn.Sequential(nn.Linear(128, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, len(POLICY_ACTIONS)))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1] != MAX_IR_TOKENS or tokens.shape[-1] != IR_TOKEN_FEATURE_DIM:
            raise ValueError("PG-131 IR tokens must have shape [batch, max_tokens, feature_dim]")
        positions = torch.arange(MAX_IR_TOKENS, device=tokens.device).unsqueeze(0)
        encoded = self.embedding(tokens) + self.position(positions)
        padding = tokens.abs().sum(dim=-1) <= 1e-8
        all_padding = padding.all(dim=1)
        if bool(all_padding.any()):
            padding = padding.clone()
            padding[all_padding, 0] = False
        encoded = self.transformer(encoded, src_key_padding_mask=padding)
        token_weights = tokens[..., SLOT_DIM + VALUE_DIM + 1].clamp_min(0.0)
        denominator = token_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = (encoded * token_weights.unsqueeze(-1)).sum(dim=1) / denominator
        current = tokens[..., SLOT_DIM + VALUE_DIM + 2].clamp_min(0.0)
        current_denominator = current.sum(dim=1, keepdim=True).clamp_min(1e-6)
        current_context = (encoded * current.unsqueeze(-1)).sum(dim=1) / current_denominator
        return self.classifier(torch.cat([pooled, current_context], dim=-1))


def policy_index_for_layered_ir(action: str) -> int:
    return policy_index(action)


__all__ = [
    "HIDDEN_DIM",
    "IR_MODES",
    "IR_SLOT_IDS",
    "IR_TOKEN_FEATURE_DIM",
    "IR_VALUE_VOCAB",
    "LayeredIRActionPolicy",
    "MAX_IR_TOKENS",
    "MAX_STEPS",
    "SCHEMA_VERSION",
    "TOKENS_PER_STEP",
    "layered_ir_token_matrix",
    "policy_index_for_layered_ir",
]
