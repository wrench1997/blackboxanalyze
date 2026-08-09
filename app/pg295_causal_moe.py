"""PG-295 causal next-token Transformer-MoE for question-driven assembly.

The model is intentionally a language model, not an action classifier.  It
receives abstract context tokens and autoregressively emits the Rule-IR target
tokens.  A small mixture-of-experts feed-forward layer supplies conditional
capacity; no expert receives a route, family, raw payload, or response body.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .pg293_failure_next_action import PAD, TARGET_BOS, TARGET_EOS, UNK, sha256_json


SCHEMA_VERSION = "pg295-causal-moe-v1"


@dataclass(frozen=True)
class CausalMoEConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    experts: int = 4
    expert_hidden: int = 256
    top_k: int = 2
    dropout: float = 0.05
    max_length: int = 128
    initializer_range: float = 0.02


class MoEFeedForward(nn.Module):
    def __init__(self, config: CausalMoEConfig) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.d_model, config.expert_hidden),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.expert_hidden, config.d_model),
                )
                for _ in range(config.experts)
            ]
        )
        self.gate = nn.Linear(config.d_model, config.experts)
        self.top_k = max(1, min(int(config.top_k), int(config.experts)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # This research implementation computes all small experts, then uses
        # top-k routing.  It keeps routing observable and deterministic for the
        # small local experiment; production dispatch can be optimized later.
        probabilities = self.gate(x).softmax(dim=-1)
        values, indices = probabilities.topk(self.top_k, dim=-1)
        values = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        selected = outputs.gather(-2, indices.unsqueeze(-1).expand(*indices.shape, outputs.shape[-1]))
        mixed = (selected * values.unsqueeze(-1)).sum(dim=-2)
        mean_probability = probabilities.mean(dim=(0, 1))
        assignment = F.one_hot(indices, num_classes=len(self.experts)).float().mean(dim=(0, 1, 2))
        balance_loss = len(self.experts) * (mean_probability * assignment).sum()
        return mixed, balance_loss


class CausalMoEBlock(nn.Module):
    def __init__(self, config: CausalMoEConfig) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(config.d_model, config.n_heads, dropout=config.dropout, batch_first=True)
        self.norm_moe = nn.LayerNorm(config.d_model)
        self.moe = MoEFeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, *, causal_mask: torch.Tensor, key_padding_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.norm_attn(x)
        attended, _ = self.attn(normalized, normalized, normalized, attn_mask=causal_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.dropout(attended)
        mixed, balance_loss = self.moe(self.norm_moe(x))
        return x + self.dropout(mixed), balance_loss


class CausalMoELanguageModel(nn.Module):
    """A compact decoder-only Transformer with MoE blocks and an LM head."""

    def __init__(self, *, vocab_size: int, config: CausalMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(int(vocab_size), config.d_model)
        self.position_embedding = nn.Embedding(config.max_length, config.d_model)
        self.blocks = nn.ModuleList([CausalMoEBlock(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, int(vocab_size), bias=False)
        self.lm_head.weight = self.token_embedding.weight
        # ``nn.Embedding`` defaults to unit-variance initialization.  With a
        # tied LM head that makes initial logits scale with the vocabulary and
        # can collapse predictive entropy before learning starts.  Keep the
        # decoder/embedding coordinate system small and explicit instead.
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=float(config.initializer_range))
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=float(config.initializer_range))

    def _mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)

    def forward(self, input_ids: torch.Tensor, *, valid_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, balance = self.forward_hidden(input_ids, valid_mask=valid_mask)
        return self.lm_head(hidden), balance

    def forward_hidden(self, input_ids: torch.Tensor, *, valid_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized causal hidden states for structured heads.

        The language-model path remains the default ``forward`` contract.  A
        separate hidden-state accessor lets an auxiliary Rule-IR target-slot
        decoder read the representation at an explicit boundary token without
        putting target answers back into the context.  It does not remove or
        pool any ontology axis; callers decide which boundary position to use.
        """
        if input_ids.ndim != 2:
            raise ValueError("CausalMoE input_ids must be [batch, length]")
        length = input_ids.shape[1]
        if length > self.config.max_length:
            raise ValueError("CausalMoE sequence exceeds max_length")
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        key_padding_mask = None if valid_mask is None else ~valid_mask.bool()
        causal_mask = self._mask(length, input_ids.device)
        balance = torch.zeros((), device=input_ids.device)
        for block in self.blocks:
            x, block_balance = block(x, causal_mask=causal_mask, key_padding_mask=key_padding_mask)
            balance = balance + block_balance
        return self.norm(x), balance / max(len(self.blocks), 1)


def build_vocabulary(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    tokens = {PAD, UNK}
    for row in records:
        tokens.update(str(token) for token in row.get("context_tokens", []))
        tokens.update(str(token) for token in row.get("target_tokens", []))
    ordered = [PAD, UNK] + sorted(tokens - {PAD, UNK})
    return {token: index for index, token in enumerate(dict.fromkeys(ordered))}


def _sequence(row: Mapping[str, Any], vocabulary: Mapping[str, int]) -> list[int]:
    unknown = int(vocabulary[UNK])
    tokens = list(row.get("context_tokens") or []) + list(row.get("target_tokens") or [])
    return [int(vocabulary.get(str(token), unknown)) for token in tokens]


def _batch(
    records: Sequence[Mapping[str, Any]],
    vocabulary: Mapping[str, int],
    device: torch.device,
    *,
    max_length: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not records:
        raise ValueError("CausalMoE batch cannot be empty")
    sequences = [_sequence(row, vocabulary) for row in records]
    width = min(max(len(item) for item in sequences), max(1, int(max_length)))
    pad = int(vocabulary[PAD])
    ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device)
    valid = torch.zeros((len(sequences), width), dtype=torch.bool, device=device)
    for index, sequence in enumerate(sequences):
        clipped = sequence[:width]
        ids[index, : len(clipped)] = torch.tensor(clipped, dtype=torch.long, device=device)
        valid[index, : len(clipped)] = True
    return ids, valid


def train_causal_moe(records: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], device: torch.device, *, seed: int, config: CausalMoEConfig, epochs: int = 160, learning_rate: float = 0.002, token_weights: Mapping[str, float] | None = None, initial_state: Mapping[str, torch.Tensor] | None = None, normalize_weighted_loss: bool = False, batch_size: int | None = None) -> CausalMoELanguageModel:
    if not records:
        raise ValueError("CausalMoE cannot train on empty records")
    torch.manual_seed(int(seed))
    model = CausalMoELanguageModel(vocab_size=len(vocabulary), config=config).to(device)
    if initial_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in initial_state.items()})
    effective_batch_size = len(records) if batch_size is None else max(1, int(batch_size))
    batches = [
        _batch(records[start : start + effective_batch_size], vocabulary, device, max_length=config.max_length)
        for start in range(0, len(records), effective_batch_size)
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    pad = int(vocabulary[PAD])
    weight_vector = torch.ones((len(vocabulary),), dtype=torch.float32, device=device)
    for token, weight in (token_weights or {}).items():
        if str(token) in vocabulary:
            weight_vector[int(vocabulary[str(token)])] = float(weight)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    for _ in range(int(epochs)):
        for ids, valid in batches:
            model.train()
            logits, balance = model(ids[:, :-1], valid_mask=valid[:, :-1])
            labels = ids[:, 1:]
            label_valid = valid[:, 1:]
            labels = labels.masked_fill(~label_valid, pad)
            per_token = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=pad, reduction="none")
            label_weights = weight_vector[labels.reshape(-1)]
            valid_weights = label_weights[label_valid.reshape(-1)]
            valid_loss = per_token[label_valid.reshape(-1)] * valid_weights
            if normalize_weighted_loss:
                # Token weights are a curriculum/slot objective, not a hidden
                # penalty for long contexts.  Normalize by the effective weight
                # mass so a ten-token Rule-IR target is not diluted by thousands
                # of page tokens when context weight is zero or small.
                loss = valid_loss.sum() / valid_weights.sum().clamp_min(1.0) + 0.01 * balance
            else:
                loss = valid_loss.mean() + 0.01 * balance
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            current = float(loss.detach().cpu())
            if current < best_loss:
                best_loss = current
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def generate_target(model: CausalMoELanguageModel, context_tokens: Sequence[str], target_length: int, vocabulary: Mapping[str, int], device: torch.device) -> list[str]:
    """Autoregressively decode a target sequence after the context.

    Training presents ``context + target_tokens`` to the causal LM, so the
    first token predicted after the context must be ``[TARGET_BOS]``.  Do not
    seed the input with ``[TARGET_BOS]``: doing so shifts every target slot by
    one position and makes the decoder repeat BOS instead of emitting the
    question/action/Rule-IR fields.
    """
    unknown = int(vocabulary[UNK])
    sequence = [int(vocabulary.get(str(token), unknown)) for token in context_tokens]
    eos = int(vocabulary.get(TARGET_EOS, unknown))
    max_new = max(1, int(target_length))
    with torch.inference_mode():
        for _ in range(max_new):
            input_ids = torch.tensor(sequence[-model.config.max_length :], dtype=torch.long, device=device).unsqueeze(0)
            valid = torch.ones_like(input_ids, dtype=torch.bool)
            logits, _ = model(input_ids, valid_mask=valid)
            next_token = int(logits[0, -1].argmax(-1).detach().cpu())
            sequence.append(next_token)
            if next_token == eos:
                break
    reverse = {int(index): str(token) for token, index in vocabulary.items()}
    return [reverse.get(index, UNK) for index in sequence[len(context_tokens) :]]


def _question(tokens: Sequence[str]) -> str:
    return next((token.split("=", 1)[1] for token in tokens if str(token).startswith("question=")), "none")


def evaluate_causal_moe(model: CausalMoELanguageModel, records: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], device: torch.device) -> dict[str, Any]:
    if not records:
        return {"count": 0, "token_accuracy": None, "sequence_exact_accuracy": None, "positive_recall": None, "hard_negative_false_allow": 0, "missing_question_recall": None, "unnecessary_question_rate": None}
    token_correct = 0
    token_total = 0
    exact = 0
    predicted_safe: list[bool] = []
    expected_safe: list[bool] = []
    missing_total = 0
    missing_correct = 0
    unnecessary = 0
    normal_total = 0
    for row in records:
        expected = [str(token) for token in row.get("target_tokens", [])]
        predicted = generate_target(model, row.get("context_tokens", []), len(expected), vocabulary, device)
        expected_body = expected[1:]
        predicted_body = predicted[1:1 + len(expected_body)] if predicted and predicted[0] == TARGET_BOS else predicted[:len(expected_body)]
        token_total += len(expected_body)
        token_correct += sum(int(a == b) for a, b in zip(predicted_body, expected_body))
        exact += int(predicted_body == expected_body)
        predicted_safe.append("safe_to_send=1" in predicted)
        expected_safe.append(bool(row.get("safe_to_send", False)))
        expected_question = _question(expected_body)
        predicted_question = _question(predicted_body)
        if expected_question != "none":
            missing_total += 1
            missing_correct += int(predicted_question == expected_question)
        else:
            normal_total += 1
            unnecessary += int(predicted_question != "none")
    positive_total = sum(int(value) for value in expected_safe)
    negative_total = len(expected_safe) - positive_total
    return {
        "count": len(records),
        "token_accuracy": round(token_correct / max(token_total, 1), 6),
        "sequence_exact_accuracy": round(exact / max(len(records), 1), 6),
        "positive_recall": round(sum(int(pred and expected) for pred, expected in zip(predicted_safe, expected_safe)) / max(positive_total, 1), 6) if positive_total else None,
        "hard_negative_false_allow": sum(int(pred and not expected) for pred, expected in zip(predicted_safe, expected_safe)),
        "safe_reject_rate": round(sum(int(not pred and not expected) for pred, expected in zip(predicted_safe, expected_safe)) / max(negative_total, 1), 6),
        "missing_question_recall": round(missing_correct / max(missing_total, 1), 6) if missing_total else None,
        "unnecessary_question_rate": round(unnecessary / max(normal_total, 1), 6) if normal_total else None,
        "missing_question_count": missing_total,
    }


__all__ = [
    "CausalMoEConfig",
    "CausalMoEBlock",
    "CausalMoELanguageModel",
    "MoEFeedForward",
    "SCHEMA_VERSION",
    "build_vocabulary",
    "evaluate_causal_moe",
    "generate_target",
    "sha256_json",
    "train_causal_moe",
]
