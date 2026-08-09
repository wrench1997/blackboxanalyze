"""PG-143 model-visible oracle-availability/abstention head."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .pg139_safety_value_head import ActionSafetyValueHead, SCHEMA_VERSION as VALUE_SCHEMA


SCHEMA_VERSION = "pg143-oracle-availability-abstention-head-v1"


class OracleAvailabilityHead(nn.Module):
    """Predict whether the typed oracle is available, without authority."""

    def __init__(self, hidden_dim: int, *, head_dim: int = 48) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, 2),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.network(context)


class CausalSafetyAvailabilityPolicy(nn.Module):
    """Causal body + action/value head + explicit availability head."""

    def __init__(self, backbone: nn.Module, *, hidden_dim: int, head_seed: int = 14301) -> None:
        super().__init__()
        self.backbone = backbone
        torch.manual_seed(int(head_seed))
        self.value_head = ActionSafetyValueHead(hidden_dim, seed=head_seed)
        self.availability_head = OracleAvailabilityHead(hidden_dim)
        self._provenance: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "value_head_schema": VALUE_SCHEMA,
            "availability_head_fresh": True,
            "head_seed": int(head_seed),
            "typed_availability_label_only": True,
            "positive_authority_in_input": False,
            "evaluator_action_in_input": False,
        }

    def context(self, token_ids: torch.Tensor) -> torch.Tensor:
        output = self.backbone.contextualize(token_ids)
        mask = token_ids.ne(0)
        lengths = mask.to(torch.long).sum(dim=1).clamp_min(1)
        return output[torch.arange(output.shape[0], device=output.device), lengths - 1]

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.context(token_ids)
        policy_logits, safety_logits = self.value_head(context)
        availability_logits = self.availability_head(context)
        return policy_logits, safety_logits, availability_logits

    @property
    def provenance(self) -> dict[str, Any]:
        return {**self._provenance, **self.value_head.provenance}


__all__ = ["CausalSafetyAvailabilityPolicy", "OracleAvailabilityHead", "SCHEMA_VERSION"]

