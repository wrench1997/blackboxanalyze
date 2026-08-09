"""PG-292 key/value feature gate for OOD evaluator-gap abstention."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


SCHEMA_VERSION = "pg292-key-value-feature-gate-v1"


def _features_from_row(row: Mapping[str, Any]) -> set[str]:
    features: set[str] = set()
    for raw in list(row.get("context_tokens") or []):
        token = str(raw)
        if "=" not in token:
            features.add(f"token:{token}")
            continue
        key, value = token.split("=", 1)
        features.add(f"key:{key}")
        # Keep normalized, low-cardinality values.  Unknown evaluator names
        # may disappear here; their shared keys and other state values remain.
        if len(value) <= 48 and not any(char in value for char in "<>\"'()"):
            features.add(f"value:{value}")
    return features


def build_feature_vocab(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    values = {feature for row in rows for feature in _features_from_row(row)}
    return {feature: index for index, feature in enumerate(sorted(values))}


def encode(rows: Sequence[Mapping[str, Any]], vocab: Mapping[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    if not rows:
        raise ValueError("PG-292 requires non-empty rows")
    values = torch.zeros((len(rows), len(vocab)), dtype=torch.float32)
    labels = torch.tensor([1.0 if bool((row.get("target") or {}).get("safe_to_send", False)) else 0.0 for row in rows], dtype=torch.float32)
    for row_index, row in enumerate(rows):
        for feature in _features_from_row(row):
            if feature in vocab:
                values[row_index, int(vocab[feature])] = 1.0
    return values, labels


class FeatureGate(nn.Module):
    def __init__(self, feature_count: int, *, hidden_dim: int = 96) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(int(feature_count), int(hidden_dim)), nn.GELU(), nn.LayerNorm(int(hidden_dim)), nn.Linear(int(hidden_dim), 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def train_gate(rows: Sequence[Mapping[str, Any]], vocab: Mapping[str, int], device: torch.device, seed: int, *, negative_weight: float = 1.0, epochs: int = 240, hidden_dim: int = 96) -> FeatureGate:
    if not rows:
        raise ValueError("PG-292 cannot train on empty rows")
    torch.manual_seed(int(seed))
    model = FeatureGate(len(vocab), hidden_dim=hidden_dim).to(device)
    values, labels = encode(rows, vocab)
    values, labels = values.to(device), labels.to(device)
    weights = torch.where(labels < 0.5, torch.tensor(float(negative_weight), device=device), torch.tensor(1.0, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    best_state = None
    best_loss = float("inf")
    for _ in range(int(epochs)):
        model.train()
        loss = (F.binary_cross_entropy_with_logits(model(values), labels, reduction="none") * weights).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        value = float(loss.detach().cpu())
        if value < best_loss:
            best_loss = value
            best_state = {key: item.detach().cpu() for key, item in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict(model: FeatureGate, rows: Sequence[Mapping[str, Any]], vocab: Mapping[str, int], device: torch.device, *, threshold: float = 0.5) -> dict[str, Any]:
    values, labels = encode(rows, vocab)
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(values.to(device))).detach().cpu().tolist()
    expected = [bool(value >= 0.5) for value in labels.tolist()]
    predicted = [float(value) >= float(threshold) for value in probabilities]
    positive_total = sum(int(value) for value in expected)
    negative_total = len(expected) - positive_total
    return {"count": len(rows), "threshold": float(threshold), "positive_recall": round(sum(int(actual and target) for actual, target in zip(predicted, expected)) / max(positive_total, 1), 6), "safe_reject_rate": round(sum(int(not actual and not target) for actual, target in zip(predicted, expected)) / max(negative_total, 1), 6), "false_allow_count": int(sum(int(actual and not target) for actual, target in zip(predicted, expected))), "expected_positive": int(positive_total), "expected_negative": int(negative_total), "probability_mean": round(sum(float(value) for value in probabilities) / max(len(probabilities), 1), 6), "probability_max": round(max((float(value) for value in probabilities), default=0.0), 6)}


__all__ = ["FeatureGate", "SCHEMA_VERSION", "build_feature_vocab", "encode", "predict", "train_gate"]
