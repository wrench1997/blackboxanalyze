"""PG-285 structured payload-grounding next-token decoder.

The decoder emits a bounded sequence of Rule-IR wire slots, for example
``method=GET`` / ``channel=query`` / ``encoding=url_percent`` and an
``<RUNTIME_CANARY>`` field placeholder.  It never receives or emits a literal
exploit string.  A separate authorized adapter may bind the placeholder to a
non-destructive canary after the evaluator gates pass.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


SCHEMA_VERSION = "pg285-payload-grounding-decoder-v1"
PAD = "[PAD]"
UNK = "[UNK]"
TARGET_BOS = "[TARGET_BOS]"
TARGET_EOS = "[TARGET_EOS]"
NEGATIVE_TOKENS = frozenset({"safe_to_send=0", "final_action=abstain", "plan=abstain"})


def canonical(value: Any) -> bytes:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    import hashlib

    return hashlib.sha256(canonical(value)).hexdigest()


def build_vocabs(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    context_tokens = {PAD, UNK}
    target_tokens = {PAD, UNK}
    for row in rows:
        context_tokens.update(str(token) for token in list(row.get("context_tokens") or []))
        target_tokens.update(str(token) for token in list(row.get("target_tokens") or []))
    context_vocab = {token: index for index, token in enumerate([PAD, UNK] + sorted(context_tokens - {PAD, UNK}))}
    target_vocab = {token: index for index, token in enumerate([PAD, UNK] + sorted(target_tokens - {PAD, UNK}))}
    if TARGET_BOS not in target_vocab or TARGET_EOS not in target_vocab:
        raise ValueError("PG-285 target vocabulary must contain decoder boundary tokens")
    return context_vocab, target_vocab


def encode_rows(
    rows: Sequence[Mapping[str, Any]],
    context_vocab: Mapping[str, int],
    target_vocab: Mapping[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[str]]]:
    if not rows:
        raise ValueError("PG-285 requires non-empty rows")
    context_sequences = [[int(context_vocab.get(str(token), context_vocab[UNK])) for token in list(row.get("context_tokens") or [])] for row in rows]
    target_sequences = [[int(target_vocab.get(str(token), target_vocab[UNK])) for token in list(row.get("target_tokens") or [])] for row in rows]
    if any(not sequence for sequence in context_sequences + target_sequences):
        raise ValueError("PG-285 sequences cannot be empty")
    context_values = torch.full((len(rows), max(len(item) for item in context_sequences)), int(context_vocab[PAD]), dtype=torch.long)
    context_lengths = torch.tensor([len(item) for item in context_sequences], dtype=torch.long)
    target_values = torch.full((len(rows), max(len(item) for item in target_sequences)), int(target_vocab[PAD]), dtype=torch.long)
    for index, sequence in enumerate(context_sequences):
        context_values[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    for index, sequence in enumerate(target_sequences):
        target_values[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    target_tokens = [[str(token) for token in list(row.get("target_tokens") or [])] for row in rows]
    return context_values, context_lengths, target_values, target_tokens


class PayloadGroundingDecoder(nn.Module):
    """Small encoder/decoder used for structured wire-plan next tokens."""

    def __init__(self, context_vocab_size: int, target_vocab_size: int, *, embed_dim: int = 96, hidden_dim: int = 192) -> None:
        super().__init__()
        self.context_embedding = nn.Embedding(int(context_vocab_size), int(embed_dim))
        self.encoder = nn.GRU(int(embed_dim), int(hidden_dim), batch_first=True)
        self.target_embedding = nn.Embedding(int(target_vocab_size), int(embed_dim))
        self.decoder = nn.GRU(int(embed_dim), int(hidden_dim), batch_first=True)
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.output = nn.Linear(int(hidden_dim), int(target_vocab_size))

    def _encode(self, context_values: torch.Tensor, context_lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.context_embedding(context_values)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, context_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, state = self.encoder(packed)
        return self.norm(state[-1]).unsqueeze(0)

    def forward(self, context_values: torch.Tensor, context_lengths: torch.Tensor, target_input: torch.Tensor) -> torch.Tensor:
        state = self._encode(context_values, context_lengths)
        decoded, _ = self.decoder(self.target_embedding(target_input), state)
        return self.output(self.norm(decoded))

    def encode_context(self, context_values: torch.Tensor, context_lengths: torch.Tensor) -> torch.Tensor:
        return self._encode(context_values, context_lengths)


def _token_weight(token: str, risk_weight: float) -> float:
    return float(risk_weight) if token in NEGATIVE_TOKENS else 1.0


def train_model(
    rows: Sequence[Mapping[str, Any]],
    context_vocab: Mapping[str, int],
    target_vocab: Mapping[str, int],
    device: torch.device,
    seed: int,
    *,
    risk_weight: float = 1.0,
    epochs: int = 180,
    embed_dim: int = 96,
    hidden_dim: int = 192,
) -> PayloadGroundingDecoder:
    if not rows:
        raise ValueError("PG-285 cannot train on empty rows")
    torch.manual_seed(int(seed))
    model = PayloadGroundingDecoder(len(context_vocab), len(target_vocab), embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
    context_values, context_lengths, target_values, target_tokens = encode_rows(rows, context_vocab, target_vocab)
    context_values = context_values.to(device)
    context_lengths = context_lengths.to(device)
    target_values = target_values.to(device)
    decoder_input = target_values[:, :-1]
    labels = target_values[:, 1:]
    token_weights = torch.tensor(
        [[_token_weight(token, risk_weight) for token in tokens[1:]] for tokens in target_tokens],
        dtype=torch.float32,
        device=device,
    )
    if token_weights.shape != labels.shape:
        raise ValueError("PG-285 target token/weight shape mismatch")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0025, weight_decay=0.01)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    pad_id = int(target_vocab[PAD])
    for _ in range(int(epochs)):
        model.train()
        logits = model(context_values, context_lengths, decoder_input)
        losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none").reshape_as(labels)
        active = labels.ne(pad_id)
        loss = (losses * token_weights * active).sum() / active.sum().clamp_min(1)
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


def greedy_decode(
    model: PayloadGroundingDecoder,
    context_values: torch.Tensor,
    context_lengths: torch.Tensor,
    target_vocab: Mapping[str, int],
    *,
    max_tokens: int = 24,
) -> list[list[str]]:
    reverse = {int(index): str(token) for token, index in target_vocab.items()}
    values = context_values
    lengths = context_lengths
    state = model.encode_context(values, lengths)
    current = torch.full((values.shape[0], 1), int(target_vocab[TARGET_BOS]), dtype=torch.long, device=values.device)
    output: list[list[str]] = [[TARGET_BOS] for _ in range(values.shape[0])]
    finished = [False] * values.shape[0]
    for _ in range(int(max_tokens) - 1):
        decoded, state = model.decoder(model.target_embedding(current[:, -1:]), state)
        logits = model.output(model.norm(decoded[:, -1]))
        next_ids = logits.argmax(-1)
        current = torch.cat([current, next_ids.unsqueeze(1)], dim=1)
        for index, token_id in enumerate(next_ids.detach().cpu().tolist()):
            token = reverse.get(int(token_id), UNK)
            if not finished[index]:
                output[index].append(token)
                if token == TARGET_EOS:
                    finished[index] = True
    return output


def _extract(tokens: Sequence[str], prefix: str, fallback: str = "unknown") -> str:
    for token in tokens:
        if str(token).startswith(prefix):
            return str(token).split("=", 1)[1]
    return fallback


def render_wire_plan(tokens: Sequence[str], *, path: str = "/observed") -> str:
    """Render an operator-readable plan with a runtime canary placeholder."""

    method = _extract(tokens, "method=", "GET")
    channel = _extract(tokens, "channel=", "unknown")
    encoding = _extract(tokens, "encoding=", "unknown")
    wire = _extract(tokens, "wire=", "none")
    action = _extract(tokens, "final_action=", _extract(tokens, "plan=", "abstain"))
    if action == "abstain" or wire == "none":
        return f"ABSTAIN · reason=typed_evaluator_or_context_missing · method={method}"
    placement = {
        "query_param": f"?observed=<RUNTIME_CANARY>",
        "form_field": "form[observed]=<RUNTIME_CANARY>",
        "header_value": "header[observed]=<RUNTIME_CANARY>",
        "path_segment": f"{path.rstrip('/')}/<RUNTIME_CANARY>",
    }.get(wire, "<RUNTIME_CANARY>")
    return f"{method} {path} · {placement} · channel={channel} · encoding={encoding} · action={action}"


def evaluate(
    model: PayloadGroundingDecoder,
    rows: Sequence[Mapping[str, Any]],
    context_vocab: Mapping[str, int],
    target_vocab: Mapping[str, int],
    device: torch.device,
) -> dict[str, Any]:
    context_values, context_lengths, _, target_tokens = encode_rows(rows, context_vocab, target_vocab)
    with torch.inference_mode():
        predicted = greedy_decode(model, context_values.to(device), context_lengths.to(device), target_vocab, max_tokens=max(len(item) for item in target_tokens) + 2)
    active_count = 0
    correct_tokens = 0
    exact = 0
    action_correct = 0
    safe_correct = 0
    false_allow = 0
    true_allow = 0
    repair_total = 0
    repair_correct = 0
    abstain_total = 0
    abstain_correct = 0
    for row, expected, actual in zip(rows, target_tokens, predicted):
        expected_core = [token for token in expected if token != PAD]
        actual_core = actual[: len(expected_core)]
        active_count += len(expected_core)
        correct_tokens += sum(a == b for a, b in zip(actual_core, expected_core))
        exact += int(actual_core == expected_core)
        expected_action = str(row.get("target", {}).get("next_action", _extract(expected_core, "final_action=")))
        predicted_action = _extract(actual, "final_action=", _extract(actual, "plan=", "abstain"))
        action_correct += int(predicted_action == expected_action)
        expected_safe = bool(row.get("target", {}).get("safe_to_send", False))
        predicted_safe = _extract(actual, "safe_to_send=", "0") == "1"
        safe_correct += int(predicted_safe == expected_safe)
        false_allow += int(not expected_safe and predicted_safe)
        true_allow += int(expected_safe and predicted_safe)
        if str(row.get("state")) in {"candidate_failed", "repair_candidate"}:
            repair_total += 1
            repair_correct += int(predicted_action in {"repair_alternate", "candidate_probe"} and predicted_action == expected_action)
        if not expected_safe:
            abstain_total += 1
            abstain_correct += int(not predicted_safe)
    count = len(rows)
    return {
        "count": count,
        "token_accuracy": round(correct_tokens / max(active_count, 1), 6),
        "sequence_exact_accuracy": round(exact / max(count, 1), 6),
        "action_accuracy": round(action_correct / max(count, 1), 6),
        "safe_exact_accuracy": round(safe_correct / max(count, 1), 6),
        "false_allow_count": int(false_allow),
        "positive_recall": round(true_allow / max(sum(bool(row.get("target", {}).get("safe_to_send", False)) for row in rows), 1), 6),
        "safe_reject_rate": round(abstain_correct / max(abstain_total, 1), 6),
        "repair_action_accuracy": round(repair_correct / max(repair_total, 1), 6),
        "hard_negative_false_allow": int(false_allow) if all(bool(row.get("hard_negative")) for row in rows) else None,
        "predicted_plan_hash": digest(predicted),
    }


__all__ = [
    "PAD",
    "TARGET_BOS",
    "TARGET_EOS",
    "PayloadGroundingDecoder",
    "SCHEMA_VERSION",
    "build_vocabs",
    "digest",
    "encode_rows",
    "evaluate",
    "greedy_decode",
    "render_wire_plan",
    "train_model",
]
