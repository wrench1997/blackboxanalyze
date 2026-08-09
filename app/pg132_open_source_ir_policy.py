"""PG-132 action policy backed by an open-source Rule IR token bridge.

PG-131 represented each slot/value pair as a hand-built one-hot vector.  This
variant keeps the same bounded Rule IR and safety contract, but replaces the
one-hot input with token IDs from :mod:`app.open_source_token_embedding` and a
standard PyTorch embedding layer.  The default matrix is fresh and seeded;
the provenance object makes that fact explicit.  A local pretrained matrix
is accepted only when its source, license and SHA-256 are supplied.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .open_source_token_embedding import (
    DEFAULT_EMBEDDING_DIM,
    MAX_IR_TOKENS,
    PAD_ID,
    RuleIRTokenEmbedding,
    SCALAR_DIM,
    open_source_ir_token_inputs,
)
from .pg124_failure_conditioned_policy import POLICY_ACTIONS, policy_index


SCHEMA_VERSION = "pg132-open-source-ir-policy-v1"
HIDDEN_DIM = 64


class OpenSourceIRActionPolicy(nn.Module):
    """Transformer action head over Rule IR token IDs and scalar side data."""

    def __init__(
        self,
        *,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
        embedding_seed: int = 13201,
        pretrained_weights_path: str | None = None,
        expected_sha256: str | None = None,
        source_id: str | None = None,
        license: str | None = None,
        freeze_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.token_embedding = RuleIRTokenEmbedding(
            embedding_dim=embedding_dim,
            seed=embedding_seed,
            pretrained_weights_path=pretrained_weights_path,
            expected_sha256=expected_sha256,
            source_id=source_id,
            license=license,
            freeze_pretrained=freeze_pretrained,
        )
        self.token_projection = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.scalar_projection = nn.Sequential(
            nn.Linear(SCALAR_DIM, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.position = nn.Embedding(MAX_IR_TOKENS, 64)
        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(POLICY_ACTIONS)),
        )

    @property
    def embedding_provenance(self) -> dict[str, Any]:
        return self.token_embedding.provenance.as_dict()

    def forward(self, token_ids: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] != MAX_IR_TOKENS:
            raise ValueError("PG-132 token ids must have shape [batch, max_tokens]")
        if scalars.ndim != 3 or scalars.shape[1:] != (MAX_IR_TOKENS, SCALAR_DIM):
            raise ValueError("PG-132 scalars must have shape [batch, max_tokens, scalar_dim]")
        encoded = self.token_projection(self.token_embedding(token_ids))
        encoded = encoded + self.scalar_projection(scalars)
        positions = torch.arange(MAX_IR_TOKENS, device=token_ids.device).unsqueeze(0)
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


def policy_index_for_open_source_ir(action: str) -> int:
    return policy_index(action)


__all__ = [
    "HIDDEN_DIM",
    "MAX_IR_TOKENS",
    "OpenSourceIRActionPolicy",
    "SCHEMA_VERSION",
    "policy_index_for_open_source_ir",
    "open_source_ir_token_inputs",
]
