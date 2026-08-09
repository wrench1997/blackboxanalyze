"""PG-133 dual-layer token policy.

The policy consumes a single sequence made of bounded page/source atoms and
Rule-IR pair atoms.  Boundary tokens make the layer explicit; the scalar
channel carries the audited slot/count weight, current-step flag, position and
source-vs-IR flag.  It is a next-safe-action policy, never a vulnerability
oracle.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .layered_token_embedding import (
    DEFAULT_EMBEDDING_DIM,
    LayeredTokenEmbedding,
    MAX_LAYERED_TOKENS,
    PAD_ID,
    SCALAR_DIM,
    layered_token_inputs,
)
from .pg124_failure_conditioned_policy import POLICY_ACTIONS, policy_index


SCHEMA_VERSION = "pg133-layered-token-policy-v1"
HIDDEN_DIM = 64


class LayeredTokenActionPolicy(nn.Module):
    """Transformer over source-token + Rule-IR token IDs."""

    def __init__(self, *, embedding_dim: int = DEFAULT_EMBEDDING_DIM, hidden_dim: int = HIDDEN_DIM, embedding_seed: int = 13301) -> None:
        super().__init__()
        self.token_embedding = LayeredTokenEmbedding(embedding_dim=embedding_dim, seed=embedding_seed)
        self.token_projection = nn.Sequential(nn.Linear(embedding_dim, 64), nn.LayerNorm(64), nn.GELU())
        self.scalar_projection = nn.Sequential(nn.Linear(SCALAR_DIM, 64), nn.LayerNorm(64), nn.GELU())
        self.position = nn.Embedding(MAX_LAYERED_TOKENS, 64)
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.0, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.classifier = nn.Sequential(nn.Linear(128, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, len(POLICY_ACTIONS)))

    @property
    def embedding_provenance(self) -> dict[str, Any]:
        return self.token_embedding.provenance.as_dict()

    def forward(self, token_ids: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] != MAX_LAYERED_TOKENS:
            raise ValueError("PG-133 token ids have the wrong shape")
        if scalars.ndim != 3 or scalars.shape[1:] != (MAX_LAYERED_TOKENS, SCALAR_DIM):
            raise ValueError("PG-133 scalar channel has the wrong shape")
        encoded = self.token_projection(self.token_embedding(token_ids)) + self.scalar_projection(scalars)
        positions = torch.arange(MAX_LAYERED_TOKENS, device=token_ids.device).unsqueeze(0)
        encoded = encoded + self.position(positions)
        padding = token_ids.eq(PAD_ID)
        all_padding = padding.all(dim=1)
        if bool(all_padding.any()):
            padding = padding.clone()
            padding[all_padding, 0] = False
        encoded = self.transformer(encoded, src_key_padding_mask=padding)
        token_weights = scalars[..., 1].clamp_min(0.0)
        denominator = token_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = (encoded * token_weights.unsqueeze(-1)).sum(dim=1) / denominator
        current = scalars[..., 2].clamp_min(0.0)
        current_denominator = current.sum(dim=1, keepdim=True).clamp_min(1e-6)
        current_context = (encoded * current.unsqueeze(-1)).sum(dim=1) / current_denominator
        return self.classifier(torch.cat([pooled, current_context], dim=-1))


def policy_index_for_layered_tokens(action: str) -> int:
    return policy_index(action)


__all__ = [
    "HIDDEN_DIM",
    "LayeredTokenActionPolicy",
    "MAX_LAYERED_TOKENS",
    "SCHEMA_VERSION",
    "layered_token_inputs",
    "policy_index_for_layered_tokens",
]
