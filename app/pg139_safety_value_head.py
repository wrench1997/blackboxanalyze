"""PG-139 learned action-conditioned safety value head."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .pg124_failure_conditioned_policy import POLICY_ACTIONS


SCHEMA_VERSION = "pg139-action-conditioned-safety-value-v1"


class ActionSafetyValueHead(nn.Module):
    """Predict safety for every abstract action given the causal context."""

    def __init__(self, hidden_dim: int, *, action_dim: int = 24, head_dim: int = 64, seed: int = 13901) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(len(POLICY_ACTIONS), action_dim)
        self.context_projection = nn.Linear(hidden_dim, head_dim)
        self.action_projection = nn.Linear(action_dim, head_dim)
        self.value = nn.Sequential(nn.LayerNorm(head_dim), nn.GELU(), nn.Linear(head_dim, 1))
        self.policy = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, len(POLICY_ACTIONS)))
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        with torch.no_grad():
            self.action_embedding.weight.normal_(mean=0.0, std=0.03, generator=generator)
        self._provenance: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "action_conditioned": True,
            "head_seed": int(seed),
            "typed_contract_mask_used_in_training": False,
            "evaluator_action_in_input": False,
        }

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        policy_logits = self.policy(context)
        context_projection = self.context_projection(context).unsqueeze(1)
        action_projection = self.action_projection(self.action_embedding.weight).unsqueeze(0)
        safety_logits = self.value(torch.tanh(context_projection + action_projection)).squeeze(-1)
        return policy_logits, safety_logits

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self._provenance)


class CausalSafetyValuePolicy(nn.Module):
    """Causal backbone plus policy logits and learned per-action safety values."""

    def __init__(self, backbone: nn.Module, *, hidden_dim: int, head_seed: int = 13901) -> None:
        super().__init__()
        self.backbone = backbone
        self.safety_head = ActionSafetyValueHead(hidden_dim, seed=head_seed)
        self._provenance = {"schema_version": SCHEMA_VERSION, "backbone_decoupled": True, "safety_value_head_fresh": True, "head_seed": int(head_seed)}

    def context(self, token_ids: torch.Tensor) -> torch.Tensor:
        output = self.backbone.contextualize(token_ids)
        mask = token_ids.ne(0)
        lengths = mask.to(torch.long).sum(dim=1).clamp_min(1)
        return output[torch.arange(output.shape[0], device=output.device), lengths - 1]

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.safety_head(self.context(token_ids))

    @property
    def provenance(self) -> dict[str, Any]:
        return {**self._provenance, **self.safety_head.provenance}


__all__ = ["ActionSafetyValueHead", "CausalSafetyValuePolicy", "SCHEMA_VERSION"]
