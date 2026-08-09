"""Risk-aware action decoder for PG-197.

The XXL body remains frozen.  A structured action adapter proposes the next
abstract role, while a separate candidate gate learns whether a typed oracle,
matched negative, reset and budget are sufficient to let that role through.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import torch
from torch import nn

from .pg196_failure_action_decoder import ACTION_NAMES, FEATURE_DIM, encode_features, enumerate_rows, expected_action


GATE_FEATURE_DIM = 6


def encode_gate_features(row: dict[str, Any]) -> list[float]:
    """Project only evaluator evidence into the candidate gate."""

    return [
        float(bool(row["typed_available"])),
        float(bool(row["negative_control"])),
        1.0,  # fresh reset is a protocol fact for every replay row
        1.0,  # evidence hash is present for every replay row
        float(bool(row["candidate_signal"])),
        min(max(int(row["budget_remaining"]), 0), 3) / 3.0,
    ]


class RiskAwareActionDecoder(nn.Module):
    def __init__(self, frozen_base: nn.Module, d_model: int = 1024) -> None:
        super().__init__()
        self.frozen_base = frozen_base
        self.context_projection = nn.Sequential(nn.Linear(d_model, 4), nn.Tanh())
        self.action_adapter = nn.Sequential(nn.Linear(4 + FEATURE_DIM, 64), nn.GELU(), nn.Linear(64, len(ACTION_NAMES)))
        self.candidate_gate = nn.Sequential(nn.Linear(GATE_FEATURE_DIM, 32), nn.GELU(), nn.Linear(32, 2))

    def forward(self, ids: torch.Tensor, mask: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            hidden = self.frozen_base.hidden(ids, mask)
        action = self.action_adapter(torch.cat([0.05 * self.context_projection(hidden), features], dim=-1))
        gate_features = torch.stack([features[:, 6], features[:, 7], torch.ones_like(features[:, 0]), torch.ones_like(features[:, 0]), features[:, 5], features[:, 8]], dim=-1)
        gate = self.candidate_gate(gate_features)
        return action, gate


def _rows_tensor(rows: list[dict[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.tensor([encode_features(**{key: row[key] for key in ("method", "redirect_hops", "status_class", "candidate_signal", "typed_available", "negative_control", "budget_remaining", "failure_kind")}) for row in rows], dtype=torch.float32, device=device)
    labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long, device=device)
    gate_labels = (labels == 2).long()
    return features, labels, gate_labels


def _balance(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(int(row[key]), []).append(row)
    target = max(len(group) for group in groups.values())
    return [groups[label][index % len(groups[label])] for label in sorted(groups) for index in range(target)]


def _metrics_from_logits(action_logits: torch.Tensor, gate_logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    action = action_logits.argmax(-1)
    gate = gate_logits.argmax(-1)
    expected_candidate = labels == 2
    expected_abstain = labels == 3
    gated_action = action.clone()
    gated_action[(action == 2) & (gate != 1)] = 3
    return {
        "count": int(labels.numel()),
        "raw_accuracy": round(float((action == labels).float().mean().item()), 8),
        "raw_safe_candidate_recall": round(float(((action == 2) & expected_candidate).sum().item() / max(int(expected_candidate.sum().item()), 1)), 8),
        "raw_abstain_recall": round(float(((action == 3) & expected_abstain).sum().item() / max(int(expected_abstain.sum().item()), 1)), 8),
        "raw_unsafe_allow_count": int(((action == 2) & (labels != 2)).sum().item()),
        "gate_accuracy": round(float((gate == expected_candidate.long()).float().mean().item()), 8),
        "gate_allow_recall": round(float(((gate == 1) & expected_candidate).sum().item() / max(int(expected_candidate.sum().item()), 1)), 8),
        "gate_unsafe_allow_count": int(((gate == 1) & ~expected_candidate).sum().item()),
        "gated_accuracy": round(float((gated_action == labels).float().mean().item()), 8),
        "gated_safe_candidate_recall": round(float(((gated_action == 2) & expected_candidate).sum().item() / max(int(expected_candidate.sum().item()), 1)), 8),
        "gated_abstain_recall": round(float(((gated_action == 3) & expected_abstain).sum().item() / max(int(expected_abstain.sum().item()), 1)), 8),
        "gated_unsafe_allow_count": int(((gated_action == 2) & (labels != 2)).sum().item()),
    }


def train_risk_decoder(model: RiskAwareActionDecoder, train_rows: list[dict[str, Any]], holdout_rows: list[dict[str, Any]], ids: torch.Tensor, mask: torch.Tensor, *, epochs: int = 30, gate_epochs: int = 200) -> dict[str, Any]:
    action_rows = _balance(train_rows, "label")
    gate_groups = {0: [row for row in train_rows if int(row["label"]) != 2], 1: [row for row in train_rows if int(row["label"]) == 2]}
    gate_target = max(len(group) for group in gate_groups.values())
    gate_rows = [gate_groups[label][index % len(gate_groups[label])] for label in sorted(gate_groups) for index in range(gate_target)]
    action_features, action_labels, _ = _rows_tensor(action_rows, ids.device)
    gate_features = torch.tensor([encode_gate_features(row) for row in gate_rows], dtype=torch.float32, device=ids.device)
    gate_labels = torch.tensor([int(row["label"] == 2) for row in gate_rows], dtype=torch.long, device=ids.device)
    optimizer = torch.optim.AdamW([*model.context_projection.parameters(), *model.action_adapter.parameters()], lr=2e-3, weight_decay=0.01)
    gate_optimizer = torch.optim.AdamW(model.candidate_gate.parameters(), lr=2e-3, weight_decay=0.01)
    history: list[dict[str, Any]] = []
    for epoch in range(1, max(epochs, gate_epochs) + 1):
        if epoch <= epochs:
            model.train()
            logits, _ = model(ids.expand(len(action_rows), -1), mask.expand(len(action_rows), -1), action_features)
            loss = nn.functional.cross_entropy(logits, action_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*model.context_projection.parameters(), *model.action_adapter.parameters()], 1.0)
            optimizer.step()
        if epoch <= gate_epochs:
            model.train()
            gate_logits = model.candidate_gate(gate_features)
            gate_loss = nn.functional.cross_entropy(gate_logits, gate_labels)
            gate_optimizer.zero_grad(set_to_none=True)
            gate_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.candidate_gate.parameters(), 1.0)
            gate_optimizer.step()
        if epoch in {1, max(epochs, gate_epochs)} or epoch % 5 == 0:
            features, labels, _ = _rows_tensor(holdout_rows, ids.device)
            with torch.inference_mode():
                action_logits, gate_logits = model(ids.expand(len(holdout_rows), -1), mask.expand(len(holdout_rows), -1), features)
            history.append({"epoch": epoch, "holdout": _metrics_from_logits(action_logits, gate_logits, labels)})
    train_features, train_labels, _ = _rows_tensor(train_rows, ids.device)
    holdout_features, holdout_labels, _ = _rows_tensor(holdout_rows, ids.device)
    with torch.inference_mode():
        train_action, train_gate = model(ids.expand(len(train_rows), -1), mask.expand(len(train_rows), -1), train_features)
        holdout_action, holdout_gate = model(ids.expand(len(holdout_rows), -1), mask.expand(len(holdout_rows), -1), holdout_features)
    return {"train_rows": len(train_rows), "balanced_action_rows": len(action_rows), "balanced_gate_rows": len(gate_rows), "holdout_rows": len(holdout_rows), "history": history, "train": _metrics_from_logits(train_action, train_gate, train_labels), "holdout": _metrics_from_logits(holdout_action, holdout_gate, holdout_labels)}


def predict(model: RiskAwareActionDecoder, *, ids: torch.Tensor, mask: torch.Tensor, features: list[float]) -> dict[str, Any]:
    feature_tensor = torch.tensor([features], dtype=torch.float32, device=ids.device)
    gate_feature_tensor = torch.tensor([[features[6], features[7], 1.0, 1.0, features[5], features[8]]], dtype=torch.float32, device=ids.device)
    with torch.inference_mode():
        action_logits = model.action_adapter(torch.cat([0.05 * model.context_projection(model.frozen_base.hidden(ids, mask)), feature_tensor], dim=-1))
        gate_logits = model.candidate_gate(gate_feature_tensor)
        action_probs = torch.softmax(action_logits[0], dim=0)
        gate_probs = torch.softmax(gate_logits[0], dim=0)
    action_index = int(action_probs.argmax().item())
    gate_index = int(gate_probs.argmax().item())
    return {"raw_action": ACTION_NAMES[action_index], "raw_confidence": float(action_probs[action_index].cpu()), "gate_action": "allow_candidate" if gate_index == 1 else "abstain", "gate_confidence": float(gate_probs[gate_index].cpu()), "effective_action": ACTION_NAMES[action_index] if not (action_index == 2 and gate_index != 1) else "abstain"}


__all__ = ["RiskAwareActionDecoder", "predict", "train_risk_decoder"]
