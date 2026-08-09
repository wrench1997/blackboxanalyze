"""PG-138 decoupled causal representation and safety action head."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .pg124_failure_conditioned_policy import POLICY_ACTIONS


SCHEMA_VERSION = "pg138-decoupled-safety-head-v1"
SAFETY_HEAD_DIM = 64


class DecoupledSafetyHead(nn.Module):
    """A fresh action head trained independently from the causal LM head."""

    def __init__(self, hidden_dim: int, *, bottleneck_dim: int = SAFETY_HEAD_DIM) -> None:
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(p=0.05),
        )
        self.action_classifier = nn.Linear(bottleneck_dim, len(POLICY_ACTIONS))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.action_classifier(self.adapter(context))


class DecoupledCausalSafetyPolicy(nn.Module):
    """Backbone and safety head are explicit modules with separate provenance."""

    def __init__(self, backbone: nn.Module, *, hidden_dim: int, head_seed: int = 13801) -> None:
        super().__init__()
        self.backbone = backbone
        self.safety_head = DecoupledSafetyHead(hidden_dim)
        generator = torch.Generator(device="cpu").manual_seed(int(head_seed))
        with torch.no_grad():
            for name, parameter in self.safety_head.named_parameters():
                if parameter.ndim > 1:
                    parameter.normal_(mean=0.0, std=0.02, generator=generator)
                elif "weight" in name:
                    parameter.fill_(1.0)
                else:
                    parameter.zero_()
        self._provenance: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "backbone_decoupled": True,
            "safety_head_fresh": True,
            "safety_head_seed": int(head_seed),
            "evaluator_action_in_input": False,
        }

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self._provenance)

    def context(self, token_ids: torch.Tensor) -> torch.Tensor:
        output = self.backbone.contextualize(token_ids)
        mask = token_ids.ne(0)
        lengths = mask.to(torch.long).sum(dim=1).clamp_min(1)
        return output[torch.arange(output.shape[0], device=output.device), lengths - 1]

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.safety_head(self.context(token_ids))


__all__ = ["DecoupledCausalSafetyPolicy", "DecoupledSafetyHead", "SAFETY_HEAD_DIM", "SCHEMA_VERSION"]
