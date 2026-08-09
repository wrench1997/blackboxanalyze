"""PG-293 failure-conditioned next-action model.

PG-269/271 provide multi-step replay traces, but their human catalog and
target labels must never become model context.  This module turns only
bounded, observable process tokens into a small sequence-to-sequence task:
predict the next abstract action, the repair class, and a conservative
``safe_to_send`` bit.  It never stores a route, family, payload, response
body, evaluator key, or literal probe.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


SCHEMA_VERSION = "pg293-failure-next-action-v1"
PAD = "[PAD]"
UNK = "[UNK]"
CONTEXT_BOS = "[BOS]"
CONTEXT_EOS = "[EOS]"
CONTEXT_END = "[CTX_END]"
TARGET_BOS = "[TARGET_BOS]"
TARGET_EOS = "[TARGET_EOS]"
SPECIAL_TOKENS = (PAD, UNK, CONTEXT_BOS, CONTEXT_EOS, TARGET_BOS, TARGET_EOS)

NEXT_ACTIONS = (
    "abstain",
    "recheck_oracle",
    "repair_candidate",
    "replay_confirmed",
    "inspect_binding",
    "inspect_environment",
)
REPAIR_ACTIONS = (
    "none",
    "recheck_oracle",
    "retry_candidate",
    "inspect_binding",
    "inspect_environment",
    "gate_correction",
)

_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.=:+/-]{1,80}$")
_FORBIDDEN_MARKERS = (
    "payload",
    "raw_response",
    "response_body",
    "body_text",
    "<script",
    "javascript:",
    "union",
    "sleep(",
    "benchmark(",
)

# These are observations a bounded collector can expose.  Labels such as
# family, lane, typed effect, oracle availability and replay outcome are
# deliberately excluded even when they exist in the source record.
OBSERVABLE_KEYS = frozenset(
    {
        "phase",
        "method",
        "status",
        "channel",
        "field_bucket",
        "history_bucket",
        "fresh_reset",
        "reset_completed",
        "source_attested",
        "reference_sent",
        "negative_sent",
        "candidate_sent",
        "candidate_error_shape",
        "boolean_differential",
        "negative_result_absent",
        "backend_observed",
        "database_health",
        "binding_valid",
        "hard_gate",
        "transport_error",
        "result_mismatch",
        "model_abstained",
        "model_claimed_positive",
        "feedback",
        "failure",
        "failure_observed",
        "repair_attempted",
        "step_budget",
        "step",
        "candidate_present",
        "self_error",
        "replay_expected",
    }
)

SAFE_FAILURES = frozenset(
    {
        "candidate_no_effect",
        "environment_failure",
        "binding_failure",
        "reference_disagreement",
        "result_mismatch",
        "oracle_unavailable",
        "no_surface_delta",
        "method_disagreement",
        "budget_exhausted",
        "model_self_error",
        "negative_clean",
        "reference_observed",
        "candidate_observed",
        "no_typed_repair_available",
    }
)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _safe_token(token: str) -> bool:
    lowered = token.casefold()
    return bool(_TOKEN_RE.fullmatch(token)) and not any(marker in lowered for marker in _FORBIDDEN_MARKERS)


def sanitize_context_tokens(tokens: Sequence[str]) -> list[str]:
    """Project source tokens to observable, family-free model context."""

    if not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes)):
        raise ValueError("PG-293 context tokens must be a sequence")
    result: list[str] = []
    for raw in tokens:
        token = str(raw)
        if token in {CONTEXT_BOS, CONTEXT_EOS, CONTEXT_END}:
            result.append(CONTEXT_EOS if token == CONTEXT_END else token)
            continue
        if not _safe_token(token):
            raise ValueError("PG-293 context contains forbidden or malformed token")
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key not in OBSERVABLE_KEYS:
            continue
        if key == "failure" and value not in SAFE_FAILURES:
            # Do not allow a hidden oracle/outcome name to pass as a failure.
            continue
        result.append(f"{key}={value}")
    if not result or result[0] != CONTEXT_BOS:
        result.insert(0, CONTEXT_BOS)
    if result[-1] != CONTEXT_EOS:
        result.append(CONTEXT_EOS)
    if len(result) > 96:
        raise ValueError("PG-293 context is too long")
    return result


def _repair_from_value(value: Any) -> str:
    value = str(value or "none")
    return value if value in REPAIR_ACTIONS else "none"


def normalize_record(row: Mapping[str, Any], *, source_group: str, split: str) -> dict[str, Any]:
    """Create one abstract training row without retaining source identity."""

    raw_context = row.get("context_tokens") or row.get("tokens") or []
    context_tokens = sanitize_context_tokens(raw_context)
    labels = dict(row.get("labels") or {})
    lane = str(row.get("lane", ""))
    final_belief = str(labels.get("final_belief", ""))
    repair_attempted = bool(labels.get("repair_attempted", False) or row.get("repair_attempted", False))
    repair = _repair_from_value(row.get("repair_action", labels.get("repair_action", "none")))
    if repair == "retry_candidate":
        next_action = "repair_candidate"
    elif repair == "inspect_binding":
        next_action = "inspect_binding"
    elif repair == "inspect_environment":
        next_action = "inspect_environment"
    elif repair == "recheck_oracle":
        next_action = "recheck_oracle"
    elif final_belief == "confirmed_effect" or bool(labels.get("repair_succeeded", False)):
        next_action = "replay_confirmed"
    elif repair_attempted:
        next_action = "repair_candidate"
    else:
        next_action = "abstain"
    grounded_effect = bool(row.get("payload_grounded_eligible", False) or final_belief == "confirmed_effect")
    if lane in {"hard_negative", "reject", "quarantine"}:
        safe_to_send = False
    else:
        safe_to_send = next_action == "replay_confirmed" and grounded_effect
    target_tokens = [
        TARGET_BOS,
        f"next_action={next_action}",
        f"repair_action={repair}",
        f"safe_to_send={int(safe_to_send)}",
        TARGET_EOS,
    ]
    evidence = str(row.get("source_evidence_hash", row.get("evidence_hash", "")))
    if not re.fullmatch(r"[0-9a-f]{64}", evidence):
        evidence = sha256_json({"source_group": source_group, "context": context_tokens, "target": target_tokens})
    record = {
        "schema_version": SCHEMA_VERSION,
        "record_id": f"pg293:{split}:{sha256_json(context_tokens + target_tokens)[:16]}",
        "source_group": source_group,
        "split": split,
        "context_tokens": context_tokens,
        "target_tokens": target_tokens,
        "next_action": next_action,
        "repair_action": repair,
        "safe_to_send": bool(safe_to_send),
        "hard_negative": bool(not safe_to_send),
        "source_evidence_hash": evidence,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "route_identity_stored": False,
        "family_identity_stored": False,
        "oracle_label_in_context": False,
        "training_eligible": True,
        "memory_promotion_allowed": False,
    }
    record["record_sha256"] = sha256_json(record)
    return record


def build_vocabulary(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    tokens = set(SPECIAL_TOKENS)
    for row in records:
        tokens.update(str(token) for token in row.get("context_tokens", []))
        tokens.update(str(token) for token in row.get("target_tokens", []))
    ordered = list(SPECIAL_TOKENS) + sorted(tokens.difference(SPECIAL_TOKENS))
    return {token: index for index, token in enumerate(dict.fromkeys(ordered))}


def encode_batch(records: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not records:
        raise ValueError("PG-293 cannot encode an empty batch")
    unk = int(vocabulary[UNK])

    def ids(tokens: Sequence[str]) -> list[int]:
        return [int(vocabulary.get(str(token), unk)) for token in tokens]

    contexts = [ids(row["context_tokens"]) for row in records]
    targets = [ids(row["target_tokens"]) for row in records]
    context_width = max(len(item) for item in contexts)
    target_width = max(len(item) for item in targets)
    context_tensor = torch.full((len(records), context_width), int(vocabulary[PAD]), dtype=torch.long, device=device)
    target_tensor = torch.full((len(records), target_width), int(vocabulary[PAD]), dtype=torch.long, device=device)
    for index, values in enumerate(contexts):
        context_tensor[index, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
    for index, values in enumerate(targets):
        target_tensor[index, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
    lengths = torch.tensor([len(item) for item in contexts], dtype=torch.long, device=device)
    action_labels = torch.tensor([NEXT_ACTIONS.index(str(row["next_action"])) for row in records], dtype=torch.long, device=device)
    return context_tensor, lengths, target_tensor, action_labels


class FailureNextActionModel(nn.Module):
    """Small context encoder + autoregressive abstract action decoder."""

    def __init__(self, *, vocab_size: int, hidden_dim: int = 256, layers: int = 1) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.context_embedding = nn.Embedding(int(vocab_size), self.hidden_dim)
        self.context_encoder = nn.GRU(self.hidden_dim, self.hidden_dim, num_layers=int(layers), batch_first=True)
        self.target_embedding = nn.Embedding(int(vocab_size), self.hidden_dim)
        self.target_decoder = nn.GRU(self.hidden_dim, self.hidden_dim, num_layers=int(layers), batch_first=True)
        self.token_head = nn.Linear(self.hidden_dim, int(vocab_size))
        self.action_head = nn.Linear(self.hidden_dim, len(NEXT_ACTIONS))

    def encode(self, context_ids: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, state = self.context_encoder(self.context_embedding(context_ids))
        positions = (lengths.to(device=hidden.device) - 1).clamp(min=0, max=hidden.shape[1] - 1)
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]
        return state, pooled

    def forward(self, context_ids: torch.Tensor, lengths: torch.Tensor, target_input: torch.Tensor) -> dict[str, torch.Tensor]:
        state, pooled = self.encode(context_ids, lengths)
        decoded, _ = self.target_decoder(self.target_embedding(target_input), state)
        return {"token": self.token_head(decoded), "action": self.action_head(pooled)}


def greedy_decode(model: FailureNextActionModel, context_ids: torch.Tensor, lengths: torch.Tensor, vocabulary: Mapping[str, int], *, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode target tokens without feeding any target-side ground truth."""

    state, pooled = model.encode(context_ids, lengths)
    current = torch.full(
        (context_ids.shape[0], 1),
        int(vocabulary[TARGET_BOS]),
        dtype=torch.long,
        device=context_ids.device,
    )
    decoded_tokens: list[torch.Tensor] = []
    with torch.inference_mode():
        for _ in range(max(1, int(max_length) - 1)):
            decoded, state = model.target_decoder(model.target_embedding(current[:, -1:]), state)
            logits = model.token_head(decoded[:, -1])
            next_token = logits.argmax(-1)
            decoded_tokens.append(next_token)
            current = torch.cat([current, next_token.unsqueeze(1)], dim=1)
    return torch.stack(decoded_tokens, dim=1), model.action_head(pooled)


def train_model(records: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], device: torch.device, *, seed: int = 29301, epochs: int = 160, hidden_dim: int = 256) -> FailureNextActionModel:
    if not records:
        raise ValueError("PG-293 cannot train on empty records")
    torch.manual_seed(int(seed))
    model = FailureNextActionModel(vocab_size=len(vocabulary), hidden_dim=hidden_dim).to(device)
    context, lengths, targets, actions = encode_batch(records, vocabulary, device)
    target_input, target_expected = targets[:, :-1], targets[:, 1:]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    pad = int(vocabulary[PAD])
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    for _ in range(int(epochs)):
        model.train()
        outputs = model(context, lengths, target_input)
        token_loss = F.cross_entropy(outputs["token"].reshape(-1, outputs["token"].shape[-1]), target_expected.reshape(-1), ignore_index=pad)
        action_loss = F.cross_entropy(outputs["action"], actions)
        loss = token_loss + 0.35 * action_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        value = float(loss.detach().cpu())
        if value < best_loss:
            best_loss = value
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def evaluate_model(model: FailureNextActionModel, records: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], device: torch.device) -> dict[str, Any]:
    context, lengths, targets, actions = encode_batch(records, vocabulary, device)
    with torch.inference_mode():
        token_pred, action_logits = greedy_decode(model, context, lengths, vocabulary, max_length=targets.shape[1])
        action_pred = action_logits.argmax(-1)
    pad = int(vocabulary[PAD])
    valid = targets[:, 1:].ne(pad)
    token_count = int(valid.sum().item())
    token_correct = int(((token_pred == targets[:, 1:]) & valid).sum().item())
    action_correct = int((action_pred == actions).sum().item())
    reverse = {int(index): str(token) for token, index in vocabulary.items()}
    exact = 0
    predicted_safe: list[bool] = []
    for row_index, row in enumerate(records):
        width = len(row["target_tokens"]) - 1
        predicted = [reverse.get(int(item), UNK) for item in token_pred[row_index, :width].detach().cpu().tolist()]
        expected = [str(item) for item in row["target_tokens"][1:]]
        exact += int(predicted == expected)
        predicted_safe.append("safe_to_send=1" in predicted)
    expected_safe = [bool(row.get("safe_to_send", False)) for row in records]
    negative_total = sum(int(not value) for value in expected_safe)
    positive_total = sum(int(value) for value in expected_safe)
    false_allow = sum(int(pred and not expected) for pred, expected in zip(predicted_safe, expected_safe))
    positive_recall = sum(int(pred and expected) for pred, expected in zip(predicted_safe, expected_safe)) / max(positive_total, 1)
    return {
        "count": len(records),
        "token_accuracy": round(token_correct / max(token_count, 1), 6),
        "token_count": token_count,
        "sequence_exact_accuracy": round(exact / max(len(records), 1), 6),
        "action_accuracy": round(action_correct / max(len(records), 1), 6),
        "positive_recall": round(positive_recall, 6) if positive_total else None,
        "hard_negative_false_allow": int(false_allow),
        "safe_reject_rate": round(sum(int(not pred and not expected) for pred, expected in zip(predicted_safe, expected_safe)) / max(negative_total, 1), 6),
    }


__all__ = [
    "FailureNextActionModel",
    "NEXT_ACTIONS",
    "OBSERVABLE_KEYS",
    "REPAIR_ACTIONS",
    "SCHEMA_VERSION",
    "build_vocabulary",
    "encode_batch",
    "evaluate_model",
    "greedy_decode",
    "normalize_record",
    "sanitize_context_tokens",
    "sha256_json",
    "train_model",
]
