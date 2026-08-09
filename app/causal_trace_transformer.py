"""Small causal Transformer for abstract security-experiment traces."""

from __future__ import annotations

import torch
from torch import nn


class CausalTraceTransformer(nn.Module):
    def __init__(self, vocab_size: int, *, d_model: int = 96, nhead: int = 4, layers: int = 2, max_len: int = 256) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def encode(self, token_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, length = token_ids.shape
        if length > self.max_len:
            raise ValueError(f"trace length {length} exceeds max_len {self.max_len}")
        positions = torch.arange(length, device=token_ids.device).unsqueeze(0).expand(batch, length)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(torch.ones(length, length, device=token_ids.device, dtype=torch.bool), diagonal=1)
        padding_mask = None if attention_mask is None else ~attention_mask.bool()
        hidden = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=padding_mask)
        return self.norm(hidden)

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.lm_head(self.encode(token_ids, attention_mask))
