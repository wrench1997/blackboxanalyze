"""PG-291 learned send/abstain gate over observation context tokens."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


SCHEMA_VERSION = "pg291-abstain-gate-v1"
PAD = "[PAD]"
UNK = "[UNK]"


class AbstainGate(nn.Module):
    def __init__(self, vocab_size: int, *, embed_dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(vocab_size), int(embed_dim))
        self.encoder = nn.GRU(int(embed_dim), int(hidden_dim), batch_first=True)
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.output = nn.Linear(int(hidden_dim), 1)

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(values)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, state = self.encoder(packed)
        return self.output(self.norm(state[-1])).squeeze(-1)


def build_vocab(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    tokens = {PAD, UNK}
    for row in rows:
        tokens.update(str(token) for token in list(row.get("context_tokens") or []))
    return {token: index for index, token in enumerate([PAD, UNK] + sorted(tokens - {PAD, UNK}))}


def encode(rows: Sequence[Mapping[str, Any]], vocab: Mapping[str, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences = [[int(vocab.get(str(token), vocab[UNK])) for token in list(row.get("context_tokens") or [])] for row in rows]
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("PG-291 requires non-empty context sequences")
    values = torch.full((len(sequences), max(len(sequence) for sequence in sequences)), int(vocab[PAD]), dtype=torch.long)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    for index, sequence in enumerate(sequences):
        values[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    labels = torch.tensor([1.0 if bool((row.get("target") or {}).get("safe_to_send", False)) else 0.0 for row in rows], dtype=torch.float32)
    return values, lengths, labels


def train_gate(
    rows: Sequence[Mapping[str, Any]],
    vocab: Mapping[str, int],
    device: torch.device,
    seed: int,
    *,
    negative_weight: float = 1.0,
    epochs: int = 180,
    embed_dim: int = 64,
    hidden_dim: int = 128,
) -> AbstainGate:
    if not rows:
        raise ValueError("PG-291 cannot train on empty rows")
    torch.manual_seed(int(seed))
    model = AbstainGate(len(vocab), embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
    values, lengths, labels = encode(rows, vocab)
    values, lengths, labels = values.to(device), lengths.to(device), labels.to(device)
    weights = torch.where(labels < 0.5, torch.tensor(float(negative_weight), device=device), torch.tensor(1.0, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for _ in range(int(epochs)):
        model.train()
        logits = model(values, lengths)
        loss = (F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * weights).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        current = float(loss.detach().cpu())
        if current < best_loss:
            best_loss = current
            best_state = {key: item.detach().cpu() for key, item in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict(model: AbstainGate, rows: Sequence[Mapping[str, Any]], vocab: Mapping[str, int], device: torch.device, *, threshold: float = 0.5) -> dict[str, Any]:
    values, lengths, labels = encode(rows, vocab)
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(values.to(device), lengths.to(device))).detach().cpu().tolist()
    predicted = [float(value) >= float(threshold) for value in probabilities]
    expected = [bool(value >= 0.5) for value in labels.tolist()]
    true_allow = sum(int(actual and target) for actual, target in zip(predicted, expected))
    positive_total = sum(int(target) for target in expected)
    false_allow = sum(int(actual and not target) for actual, target in zip(predicted, expected))
    negative_total = len(expected) - positive_total
    return {
        "count": len(rows),
        "threshold": float(threshold),
        "positive_recall": round(true_allow / max(positive_total, 1), 6),
        "safe_reject_rate": round(sum(int(not actual and not target) for actual, target in zip(predicted, expected)) / max(negative_total, 1), 6),
        "false_allow_count": int(false_allow),
        "expected_positive": int(positive_total),
        "expected_negative": int(negative_total),
        "probability_mean": round(sum(float(value) for value in probabilities) / max(len(probabilities), 1), 6),
        "probability_min": round(min((float(value) for value in probabilities), default=0.0), 6),
        "probability_max": round(max((float(value) for value in probabilities), default=0.0), 6),
    }


__all__ = ["AbstainGate", "SCHEMA_VERSION", "build_vocab", "encode", "predict", "train_gate"]
