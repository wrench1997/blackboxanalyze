"""PG-283 multi-step feedback policy.

This is a small, remote-trained process head.  It predicts the next abstract
action, Rule-IR plan slots and a safe-to-send gate from bounded observation
tokens.  It does not generate literal payloads.  Hard negatives are kept
outside training and are scored for false-allow behavior.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


SCHEMA_VERSION = "pg283-feedback-policy-v1"
PAD = "[PAD]"
UNK = "[UNK]"
ACTIONS = ("send_negative", "send_reference", "ask_typed", "send_candidate", "repair_alternate", "replay_confirmed", "abstain")
PROBE_CLASSES = ("sql", "xss", "redirect", "logic", "file", "other")
CHANNELS = ("query", "form", "unknown")
ENCODINGS = ("plain", "url_percent", "unknown")


def canonical(value: Any) -> bytes:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    import hashlib

    return hashlib.sha256(canonical(value)).hexdigest()


class FeedbackPolicy(nn.Module):
    def __init__(self, vocab_size: int, *, embed_dim: int = 96, hidden_dim: int = 192) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(vocab_size), int(embed_dim))
        self.encoder = nn.GRU(int(embed_dim), int(hidden_dim), batch_first=True)
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.action = nn.Linear(int(hidden_dim), len(ACTIONS))
        self.probe = nn.Linear(int(hidden_dim), len(PROBE_CLASSES))
        self.channel = nn.Linear(int(hidden_dim), len(CHANNELS))
        self.encoding = nn.Linear(int(hidden_dim), len(ENCODINGS))
        self.safe = nn.Linear(int(hidden_dim), 1)

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        packed = nn.utils.rnn.pack_padded_sequence(self.embedding(values), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, state = self.encoder(packed)
        state = self.norm(state[-1])
        return {
            "action": self.action(state),
            "probe": self.probe(state),
            "channel": self.channel(state),
            "encoding": self.encoding(state),
            "safe": self.safe(state).squeeze(-1),
        }


def build_vocab(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    tokens = {PAD, UNK}
    for row in rows:
        tokens.update(str(token) for token in list(row.get("context_tokens") or []))
    return {token: index for index, token in enumerate([PAD, UNK] + sorted(tokens - {PAD, UNK}))}


def encode(rows: Sequence[Mapping[str, Any]], vocab: Mapping[str, int]) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    if not rows:
        raise ValueError("PG-283 requires non-empty rows")
    sequences = [[int(vocab.get(str(token), vocab[UNK])) for token in list(row.get("context_tokens") or [])] for row in rows]
    if any(not sequence for sequence in sequences):
        raise ValueError("PG-283 context tokens cannot be empty")
    values = torch.full((len(sequences), max(len(sequence) for sequence in sequences)), int(vocab[PAD]), dtype=torch.long)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    for index, sequence in enumerate(sequences):
        values[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    targets = {
        "action": torch.tensor([ACTIONS.index(str(row["target"]["next_action"])) for row in rows], dtype=torch.long),
        "probe": torch.tensor([PROBE_CLASSES.index(str(row["target"]["probe_class"])) for row in rows], dtype=torch.long),
        "channel": torch.tensor([CHANNELS.index(str(row["target"]["channel"])) for row in rows], dtype=torch.long),
        "encoding": torch.tensor([ENCODINGS.index(str(row["target"]["encoding"])) for row in rows], dtype=torch.long),
        "safe": torch.tensor([float(bool(row["target"]["safe_to_send"])) for row in rows], dtype=torch.float32),
    }
    return values, lengths, targets


def _has_token(row: Mapping[str, Any], token: str) -> bool:
    return token in {str(value) for value in list(row.get("context_tokens") or [])}


def _hard_gate(row: Mapping[str, Any]) -> bool:
    """Transport/evaluator prerequisites visible to a guarded controller."""

    return all(
        _has_token(row, token)
        for token in (
            "fresh_reset=1",
            "negative_clean=1",
            "reference_agreement=1",
            "typed_available=1",
            "source_attested=1",
        )
    )


def evaluate(model: FeedbackPolicy, rows: Sequence[Mapping[str, Any]], vocab: Mapping[str, int], device: torch.device) -> dict[str, Any]:
    values, lengths, targets = encode(rows, vocab)
    with torch.inference_mode():
        output = model(values.to(device), lengths.to(device))
    action_ids = output["action"].argmax(-1).cpu().tolist()
    probe_ids = output["probe"].argmax(-1).cpu().tolist()
    channel_ids = output["channel"].argmax(-1).cpu().tolist()
    encoding_ids = output["encoding"].argmax(-1).cpu().tolist()
    safe_probs = torch.sigmoid(output["safe"]).cpu().tolist()
    action_correct = sum(ACTIONS[pred] == str(row["target"]["next_action"]) for pred, row in zip(action_ids, rows))
    probe_correct = sum(PROBE_CLASSES[pred] == str(row["target"]["probe_class"]) for pred, row in zip(probe_ids, rows))
    channel_correct = sum(CHANNELS[pred] == str(row["target"]["channel"]) for pred, row in zip(channel_ids, rows))
    encoding_correct = sum(ENCODINGS[pred] == str(row["target"]["encoding"]) for pred, row in zip(encoding_ids, rows))
    false_allow = 0
    guarded_false_allow = 0
    true_allow = 0
    action_safe_exact = 0
    brier = 0.0
    for index, row in enumerate(rows):
        expected_safe = bool(row["target"]["safe_to_send"])
        predicted_safe = safe_probs[index] >= 0.5
        false_allow += int(not expected_safe and predicted_safe)
        true_allow += int(expected_safe and predicted_safe)
        guarded_safe = predicted_safe and _hard_gate(row)
        guarded_false_allow += int(not expected_safe and guarded_safe)
        action_safe_exact += int(ACTIONS[action_ids[index]] == str(row["target"]["next_action"]) and predicted_safe == expected_safe)
        brier += (safe_probs[index] - float(expected_safe)) ** 2
    count = len(rows)
    expected_positive = sum(bool(row["target"]["safe_to_send"]) for row in rows)
    expected_negative = count - expected_positive
    return {
        "count": count,
        "action_accuracy": round(action_correct / max(count, 1), 6),
        "probe_accuracy": round(probe_correct / max(count, 1), 6),
        "channel_accuracy": round(channel_correct / max(count, 1), 6),
        "encoding_accuracy": round(encoding_correct / max(count, 1), 6),
        "action_safe_exact_accuracy": round(action_safe_exact / max(count, 1), 6),
        "safe_accuracy": round((true_allow + expected_negative - false_allow) / max(count, 1), 6),
        "false_allow_count": int(false_allow),
        "guarded_false_allow_count": int(guarded_false_allow),
        "true_allow_count": int(true_allow),
        "positive_recall": round(true_allow / max(expected_positive, 1), 6),
        "safe_reject_rate": round((expected_negative - false_allow) / max(expected_negative, 1), 6),
        "safe_brier": round(brier / max(count, 1), 6),
    }


def train_model(rows: Sequence[Mapping[str, Any]], vocab: Mapping[str, int], device: torch.device, seed: int, *, risk_weight: float, epochs: int = 220) -> FeedbackPolicy:
    torch.manual_seed(int(seed))
    model = FeedbackPolicy(len(vocab)).to(device)
    values, lengths, targets = encode(rows, vocab)
    values, lengths = values.to(device), lengths.to(device)
    targets = {key: value.to(device) for key, value in targets.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    for _ in range(int(epochs)):
        model.train()
        output = model(values, lengths)
        losses = [
            F.cross_entropy(output["action"], targets["action"]),
            0.45 * F.cross_entropy(output["probe"], targets["probe"]),
            0.35 * F.cross_entropy(output["channel"], targets["channel"]),
            0.35 * F.cross_entropy(output["encoding"], targets["encoding"]),
        ]
        safe_loss = F.binary_cross_entropy_with_logits(output["safe"], targets["safe"], reduction="none")
        if float(risk_weight) > 1.0:
            safe_loss = safe_loss * torch.where(targets["safe"] > 0.5, torch.ones_like(safe_loss), torch.full_like(safe_loss, float(risk_weight)))
        losses.append(safe_loss.mean())
        loss = sum(losses)
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


__all__ = [
    "ACTIONS",
    "CHANNELS",
    "ENCODINGS",
    "FeedbackPolicy",
    "PROBE_CLASSES",
    "SCHEMA_VERSION",
    "build_vocab",
    "digest",
    "encode",
    "evaluate",
    "train_model",
]
