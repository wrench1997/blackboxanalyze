"""A compact top-1 Mixture-of-Experts causal Transformer for trace tokens."""

from __future__ import annotations

import math

import torch
from torch import nn


class _MoEBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, n_experts: int, expert_ff: int) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.1)
        self.norm_moe = nn.LayerNorm(d_model)
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(d_model, expert_ff), nn.GELU(), nn.Linear(expert_ff, d_model)) for _ in range(n_experts)]
        )
        self.n_experts = n_experts

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor, causal_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.norm_attn(hidden)
        attended, _ = self.attn(normalized, normalized, normalized, attn_mask=causal_mask, key_padding_mask=~attention_mask.bool(), need_weights=False)
        hidden = hidden + attended
        normalized = self.norm_moe(hidden)
        router_logits = self.router(normalized)
        router_probs = torch.softmax(router_logits, dim=-1)
        top_prob, top_index = router_probs.max(dim=-1)
        mixed = torch.zeros_like(normalized)
        for expert_index, expert in enumerate(self.experts):
            selected = top_index.eq(expert_index)
            if bool(selected.any()):
                mixed[selected] = expert(normalized[selected]) * top_prob[selected].unsqueeze(-1)
        active = attention_mask.bool().unsqueeze(-1)
        hidden = hidden + mixed
        importance = (router_probs * active).sum(dim=(0, 1)) / active.sum().clamp_min(1)
        load = torch.stack([(top_index.eq(index) & attention_mask.bool()).sum() for index in range(self.n_experts)]).float()
        load = load / load.sum().clamp_min(1)
        # Switch-style auxiliary loss discourages a single expert absorbing
        # every token while leaving the primary LM objective unchanged.
        balance_loss = self.n_experts * torch.sum(importance * load)
        return hidden, balance_loss, load.detach()


class MoETraceTransformer(nn.Module):
    def __init__(self, vocab_size: int, *, d_model: int = 512, nhead: int = 8, layers: int = 4, n_experts: int = 4, expert_ff: int = 2048, max_len: int = 128) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([_MoEBlock(d_model, nhead, n_experts, expert_ff) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.n_experts = n_experts
        self.max_len = max_len

    def encode(self, token_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length = token_ids.shape
        if length > self.max_len:
            raise ValueError(f"trace length {length} exceeds max_len {self.max_len}")
        if attention_mask is None:
            attention_mask = token_ids.ne(0)
        positions = torch.arange(length, device=token_ids.device).unsqueeze(0).expand(batch, length)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(torch.ones(length, length, device=token_ids.device, dtype=torch.bool), diagonal=1)
        aux = hidden.new_zeros(())
        loads = []
        for block in self.blocks:
            hidden, block_aux, load = block(hidden, attention_mask, causal_mask)
            aux = aux + block_aux
            loads.append(load)
        return self.norm(hidden), aux / max(len(self.blocks), 1), torch.stack(loads)

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden, _, _ = self.encode(token_ids, attention_mask)
        return self.lm_head(hidden)

    def auxiliary_loss_and_load(self, token_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        _, aux, loads = self.encode(token_ids, attention_mask)
        return aux, loads


__all__ = ["MoETraceTransformer"]

