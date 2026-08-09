"""Family-specific discriminator for sanitized shadow HTTP surfaces."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .rule_ir_decoder import DECODER_FAMILIES, FEATURE_DIM


SURFACE_START = 145
SURFACE_END = 189
SURFACE_DIM = SURFACE_END - SURFACE_START
CONTEXT_DIM = FEATURE_DIM - SURFACE_DIM


class SurfaceDiscriminator(nn.Module):
    """Two-tower classifier: response/action surface plus semantic context."""

    def __init__(self, hidden_dim: int = 160, dropout: float = 0.10):
        super().__init__()
        self.surface_tower = nn.Sequential(
            nn.Linear(SURFACE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.context_tower = nn.Sequential(
            nn.Linear(CONTEXT_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(DECODER_FAMILIES)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        surface = features[:, SURFACE_START:SURFACE_END]
        context = torch.cat((features[:, :SURFACE_START], features[:, SURFACE_END:]), dim=-1)
        surface_hidden = self.surface_tower(surface)
        context_hidden = self.context_tower(context)
        joined = torch.cat((surface_hidden, context_hidden), dim=-1)
        gate = self.gate(joined)
        fused = torch.cat((surface_hidden * gate, context_hidden * (1.0 - gate)), dim=-1)
        return self.classifier(fused)

    @torch.inference_mode()
    def probabilities(self, features: torch.Tensor) -> list[dict[str, float]]:
        values = torch.softmax(self(features), dim=-1).detach().cpu()
        return [
            {family: round(float(probability), 6) for family, probability in zip(DECODER_FAMILIES, row)}
            for row in values
        ]
