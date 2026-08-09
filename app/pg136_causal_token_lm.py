"""PG-136 bounded causal next-token model.

The input is the already-sanitized source/Rule-IR projection used by the
replay experiments.  The causal body is trained with a next-token objective;
the action head is trained afterwards from safe abstract action labels.  No
raw HTML/JavaScript, probe/response body, target identity, evaluator action,
or positive-authority field is tokenized here.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Mapping, Sequence

import torch
from torch import nn

from .pg124_failure_conditioned_policy import POLICY_ACTIONS, policy_index


SCHEMA_VERSION = "pg136-causal-next-token-gru-v1"
PAD_TOKEN = "[PAD]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
UNK_TOKEN = "[UNK]"
SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)
EMBED_DIM = 48
HIDDEN_DIM = 64
MAX_SEQUENCE_LENGTH = 384
MIN_TOKEN_FREQUENCY = 1


def _safe_value(value: object, *, value_hash: bool = False) -> str:
    """Keep only bounded categorical values in the causal vocabulary."""

    if value_hash:
        return "hash_present"
    text = str(value or "unknown")
    if len(text) > 32:
        return "unknown"
    text = re.sub(r"[^A-Za-z0-9_+.-]", "_", text)
    return text or "unknown"


def _weight_bucket(value: object) -> str:
    try:
        weight = max(0.0, min(float(value), 2.0))
    except (TypeError, ValueError):
        weight = 1.0
    if weight <= 0.5:
        return "0.5"
    if weight <= 1.0:
        return "1.0"
    if weight <= 1.25:
        return "1.25"
    if weight <= 1.5:
        return "1.5"
    return "2.0"


def canonical_tokens(layered_steps: Sequence[Mapping[str, object]]) -> list[str]:
    """Flatten bounded source/Rule-IR layers into causal categorical tokens."""

    if not layered_steps:
        raise ValueError("PG-136 requires at least one bounded step")
    tokens: list[str] = [BOS_TOKEN]
    for step in layered_steps:
        tokens.append("[STEP]")
        for layer in list(step.get("source_token_layers") or []):
            modality = _safe_value(layer.get("modality", "unknown")).lower()
            if modality not in {"html", "javascript", "transport"}:
                modality = "unknown"
            tokens.append(f"[SRC_{modality.upper()}]")
            for item in list(layer.get("tokens") or []):
                if not isinstance(item, Mapping):
                    continue
                kind = _safe_value(item.get("kind", "unknown"))
                value = _safe_value(item.get("value", "unknown"), value_hash="value_hash" in item)
                tokens.append(f"src.{modality}.{kind}={value}")
                count_bucket = item.get("count_bucket")
                if count_bucket is not None:
                    tokens.append(f"src_count={_safe_value(count_bucket)}")
        tokens.append("[IR]")
        for item in list((step.get("ir_layer") or {}).get("tokens") or []):
            if not isinstance(item, Mapping):
                continue
            slot = _safe_value(item.get("slot_id", "unknown"))
            value = _safe_value(item.get("value", "unknown"))
            tokens.append(f"ir.{slot}={value}")
            # The bounded weight is represented explicitly as a token so the
            # causal model can learn that a failure reweights later reasoning.
            tokens.append(f"ir_weight={_weight_bucket(item.get('weight', 1.0))}")
    tokens.append(EOS_TOKEN)
    if len(tokens) > MAX_SEQUENCE_LENGTH:
        raise ValueError("PG-136 causal sequence exceeds bounded length")
    return tokens


class CausalVocabulary:
    """Deterministic train-split vocabulary with an explicit unknown bucket."""

    def __init__(self, tokens: Iterable[Sequence[str]], *, min_frequency: int = MIN_TOKEN_FREQUENCY) -> None:
        counts = Counter(token for sequence in tokens for token in sequence)
        learned = sorted(token for token, count in counts.items() if count >= min_frequency and token not in SPECIAL_TOKENS)
        self.itos = list(SPECIAL_TOKENS) + learned
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    def encode(self, tokens: Sequence[str]) -> list[int]:
        return [self.stoi.get(token, self.stoi[UNK_TOKEN]) for token in tokens]

    def decode(self, ids: Sequence[int]) -> list[str]:
        return [self.itos[index] if 0 <= int(index) < len(self.itos) else UNK_TOKEN for index in ids]

    def to_dict(self) -> dict[str, object]:
        return {"itos": list(self.itos), "vocab_size": len(self.itos)}


class CausalTokenGRU(nn.Module):
    """Small causal GRU LM with a separately trained abstract action head."""

    def __init__(self, vocab_size: int, *, seed: int = 13601, embed_dim: int = EMBED_DIM, hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        if vocab_size <= len(SPECIAL_TOKENS):
            raise ValueError("PG-136 vocabulary is too small")
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(MAX_SEQUENCE_LENGTH, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        self.action_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, len(POLICY_ACTIONS)))
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        with torch.no_grad():
            self.token_embedding.weight.normal_(mean=0.0, std=0.03, generator=generator)
            self.token_embedding.weight[0].zero_()
            self.position_embedding.weight.normal_(mean=0.0, std=0.02, generator=generator)
        self._provenance = {
            "schema_version": SCHEMA_VERSION,
            "objective": "causal_next_token_then_safe_action_head",
            "pretrained": False,
            "initialization_seed": int(seed),
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "raw_fields_excluded": True,
        }

    @property
    def provenance(self) -> dict[str, object]:
        return dict(self._provenance)

    def contextualize(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] > MAX_SEQUENCE_LENGTH:
            raise ValueError("PG-136 token ids have the wrong shape")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device).unsqueeze(0)
        embedded = self.token_embedding(token_ids) + self.position_embedding(positions)
        output, _ = self.gru(embedded)
        return output

    def next_token_logits(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.contextualize(token_ids))

    def action_logits(self, token_ids: torch.Tensor) -> torch.Tensor:
        output = self.contextualize(token_ids)
        mask = token_ids.ne(0)
        lengths = mask.to(torch.long).sum(dim=1).clamp_min(1)
        last = output[torch.arange(output.shape[0], device=output.device), lengths - 1]
        return self.action_head(last)


def action_index(action: str) -> int:
    return policy_index(action)


__all__ = [
    "BOS_TOKEN",
    "CausalTokenGRU",
    "CausalVocabulary",
    "EOS_TOKEN",
    "EMBED_DIM",
    "HIDDEN_DIM",
    "MAX_SEQUENCE_LENGTH",
    "PAD_TOKEN",
    "POLICY_ACTIONS",
    "SCHEMA_VERSION",
    "UNK_TOKEN",
    "action_index",
    "canonical_tokens",
]
