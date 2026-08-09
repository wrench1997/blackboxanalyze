"""PG-134 independent token-hash + GRU policy.

This is a deliberately separate implementation of the PG-133 input contract.
It does not import the PG-133 tokenizer, embedding, or Transformer policy. A
bounded source/Rule-IR atom is hashed into a fixed bucket and pooled per step;
an independently initialized GRU then consumes the ordered prefix. The model
never receives raw HTML, JavaScript, probe strings, response bodies, target
names, evaluator actions, or positive authority.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .pg124_failure_conditioned_policy import POLICY_ACTIONS, policy_index


SCHEMA_VERSION = "pg134-independent-token-hash-gru-v1"
HASH_BUCKETS = 521
TOKEN_DIM = 32
SCALAR_DIM = 5
STEP_DIM = 64
HIDDEN_DIM = 64
MAX_STEPS = 8
MAX_TOKENS_PER_STEP = 64
TOKEN_MODES = ("full", "source_only", "ir_only", "availability_only", "weight_only", "no_weight", "zero")


def _bucket(token: str) -> int:
    digest = hashlib.blake2b(str(token).encode("utf-8"), digest_size=8, person=b"pg134-token").digest()
    # Reserve 0 for padding; all canonical atoms map to 1..HASH_BUCKETS.
    return int.from_bytes(digest, "big") % HASH_BUCKETS + 1


def _bounded_weight(value: Any) -> float:
    try:
        return max(0.0, min(float(value), 2.0))
    except (TypeError, ValueError):
        return 1.0


def _source_atoms(step: Mapping[str, Any]) -> list[tuple[str, float, int]]:
    atoms: list[tuple[str, float, int]] = []
    for layer in list(step.get("source_token_layers") or []):
        modality = str(layer.get("modality", "unknown"))
        if modality not in {"html", "javascript", "transport"}:
            modality = "unknown"
        boundary = f"[SRC_{modality.upper()}]" if modality != "unknown" else "[UNK]"
        atoms.append((boundary, 0.5, 0))
        for token in list(layer.get("tokens") or []):
            if not isinstance(token, Mapping):
                continue
            kind = str(token.get("kind", "unknown"))
            value = "hash_present" if "value_hash" in token else str(token.get("value", "unknown"))
            # The replay bridge already emits bounded categories. Keep a
            # second defensive length limit so an accidental raw field cannot
            # become a model token in this independent implementation.
            value = value[:32] if len(value) <= 32 else "unknown"
            count_bucket = str(token.get("count_bucket", "1-4"))
            weight = {"0": 0.5, "1-4": 1.0, "5-16": 1.25, "17+": 1.5}.get(count_bucket, 1.0)
            atoms.append((f"src.{modality}.{kind}={value}", weight, 0))
    return atoms


def _ir_atoms(step: Mapping[str, Any]) -> list[tuple[str, float, int]]:
    atoms: list[tuple[str, float, int]] = []
    for token in list((step.get("ir_layer") or {}).get("tokens") or []):
        if not isinstance(token, Mapping):
            continue
        slot = str(token.get("slot_id", "unknown"))[:48]
        value = str(token.get("value", "unknown"))[:32]
        atoms.append((f"ir.{slot}={value}", _bounded_weight(token.get("weight", 1.0)), 1))
    return atoms


def _step_atoms(step: Mapping[str, Any], *, mode: str) -> list[tuple[str, float, int]]:
    if mode not in TOKEN_MODES:
        raise ValueError(f"unknown PG-134 token mode: {mode}")
    source = _source_atoms(step)
    ir = _ir_atoms(step)
    if mode == "source_only":
        atoms = [("[STEP]", 0.5, 0), *source]
    elif mode == "ir_only":
        atoms = [("[STEP]", 0.5, 0), ("[IR]", 0.5, 1), *ir]
    elif mode == "availability_only":
        availability = [item for item in ir if item[0].startswith("ir.oracle.availability=")]
        atoms = [("[STEP]", 0.5, 0), ("[IR]", 0.5, 1), *availability]
    else:
        atoms = [("[STEP]", 0.5, 0), *source, ("[IR]", 0.5, 1), *ir]
    if mode == "no_weight":
        atoms = [(token, 1.0, layer) for token, _, layer in atoms]
    if mode == "zero":
        atoms = [(token, 0.0, layer) for token, _, layer in atoms]
    if len(atoms) > MAX_TOKENS_PER_STEP:
        raise ValueError("PG-134 bounded step token limit exceeded")
    return atoms


def encode_prefix(
    prefix_steps: Sequence[Mapping[str, Any]],
    *,
    mode: str = "full",
) -> tuple[list[list[int]], list[list[list[float]]]]:
    """Encode an ordered prefix without importing PG-133 token code."""

    if len(prefix_steps) <= 0 or len(prefix_steps) > MAX_STEPS:
        raise ValueError("PG-134 prefix length is outside the bounded window")
    ids: list[list[int]] = []
    scalars: list[list[list[float]]] = []
    for step_index, step in enumerate(prefix_steps):
        atoms = _step_atoms(step, mode=mode)
        denominator = sum(max(weight, 0.0) for _, weight, _ in atoms) or 1.0
        current = 1.0 if step_index == len(prefix_steps) - 1 else 0.0
        position = float(step_index) / float(max(MAX_STEPS - 1, 1))
        step_ids: list[int] = []
        step_scalars: list[list[float]] = []
        for token, weight, layer_flag in atoms:
            # Weight-only keeps the auditable scalar channel but erases token
            # identity. Zero mode erases both channels for a same-capacity
            # baseline.
            step_ids.append(1 if mode == "weight_only" else (0 if mode == "zero" else _bucket(token)))
            step_scalars.append([weight / 2.0, max(weight, 0.0) / denominator, current, position, float(layer_flag)])
        while len(step_ids) < MAX_TOKENS_PER_STEP:
            step_ids.append(0)
            step_scalars.append([0.0] * SCALAR_DIM)
        if mode == "zero":
            step_scalars = [[0.0] * SCALAR_DIM for _ in step_scalars]
        ids.append(step_ids)
        scalars.append(step_scalars)
    return ids, scalars


class IndependentTokenHashGRUPolicy(nn.Module):
    """Fresh GRU implementation independent from PG-133's Transformer."""

    def __init__(self, *, seed: int = 13401, token_dim: int = TOKEN_DIM, hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        if token_dim <= 0 or hidden_dim <= 0:
            raise ValueError("PG-134 dimensions must be positive")
        self.token_embedding = nn.Embedding(HASH_BUCKETS + 1, token_dim, padding_idx=0)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        with torch.no_grad():
            self.token_embedding.weight.normal_(mean=0.0, std=0.03, generator=generator)
            self.token_embedding.weight[0].zero_()
        self.scalar_projection = nn.Sequential(nn.Linear(SCALAR_DIM, token_dim), nn.LayerNorm(token_dim), nn.GELU())
        self.step_projection = nn.Sequential(nn.Linear(token_dim * 2, STEP_DIM), nn.LayerNorm(STEP_DIM), nn.GELU())
        self.gru = nn.GRU(STEP_DIM, hidden_dim, num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, len(POLICY_ACTIONS)))
        self._provenance = {
            "schema_version": SCHEMA_VERSION,
            "representation": "blake2b_fixed_bucket",
            "hash_buckets": HASH_BUCKETS,
            "token_dim": token_dim,
            "weights_source": "fresh_seeded_torch_embedding",
            "pretrained": False,
            "initialization_seed": int(seed),
        }

    @property
    def embedding_provenance(self) -> dict[str, Any]:
        return dict(self._provenance)

    def forward(self, token_ids: torch.Tensor, scalars: torch.Tensor, step_mask: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 3 or token_ids.shape[2] != MAX_TOKENS_PER_STEP:
            raise ValueError("PG-134 token ids have the wrong shape")
        if scalars.shape[:3] != token_ids.shape or scalars.shape[-1] != SCALAR_DIM:
            raise ValueError("PG-134 scalar channel has the wrong shape")
        if step_mask.ndim != 2 or step_mask.shape[:2] != token_ids.shape[:2]:
            raise ValueError("PG-134 step mask has the wrong shape")
        token_vectors = self.token_embedding(token_ids)
        token_mask = token_ids.ne(0).to(scalars.dtype)
        weights = scalars[..., 1] * token_mask
        denominator = weights.sum(dim=2, keepdim=True).clamp_min(1e-6)
        token_pool = (token_vectors * weights.unsqueeze(-1)).sum(dim=2) / denominator
        scalar_weights = token_mask.unsqueeze(-1)
        scalar_pool = (self.scalar_projection(scalars) * scalar_weights).sum(dim=2) / scalar_weights.sum(dim=2).clamp_min(1.0)
        step_vectors = self.step_projection(torch.cat([token_pool, scalar_pool], dim=-1))
        lengths = step_mask.to(torch.long).sum(dim=1).clamp_min(1)
        packed = nn.utils.rnn.pack_padded_sequence(step_vectors, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=token_ids.shape[1])
        mask = step_mask.to(output.dtype).unsqueeze(-1)
        mean_context = (output * mask).sum(dim=1) / lengths.to(output.dtype).unsqueeze(-1)
        last_context = output[torch.arange(output.shape[0], device=output.device), lengths.to(output.device) - 1]
        return self.classifier(torch.cat([last_context, mean_context], dim=-1))


def policy_index_for_independent_tokens(action: str) -> int:
    return policy_index(action)


__all__ = [
    "HASH_BUCKETS",
    "HIDDEN_DIM",
    "IndependentTokenHashGRUPolicy",
    "MAX_STEPS",
    "MAX_TOKENS_PER_STEP",
    "SCALAR_DIM",
    "SCHEMA_VERSION",
    "TOKEN_DIM",
    "TOKEN_MODES",
    "encode_prefix",
    "policy_index_for_independent_tokens",
]
