"""PG-363 full-context structured Rule-IR decoder.

This is an auxiliary decoder over the same decoder-only Transformer-MoE used
by the causal next-token model.  It keeps the complete abstract page context,
but pools every valid context position (mean + learned attention + boundary)
before predicting each Rule-IR slot independently.  The evaluator target is
never appended to the context during slot inference.

The module accepts only abstract tokens.  Raw payloads, response bodies,
routes, family labels and evaluator answers are deliberately absent from the
API and are rejected by the dataset/runner firewall.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .pg293_failure_next_action import PAD, TARGET_BOS, TARGET_EOS, UNK
from .pg295_causal_moe import CausalMoEConfig, CausalMoELanguageModel, _batch


SCHEMA_VERSION = "pg363-pooled-rule-ir-v1"
SLOT_PREFIXES: tuple[tuple[str, str], ...] = (
    ("question", "question="),
    ("ask_reason", "ask_reason="),
    ("next_action", "next_action="),
    ("repair_action", "repair_action="),
    ("transport_ref", "transport_ref="),
    ("field_role_ref", "field_role_ref="),
    ("encoding_ref", "encoding_ref="),
    ("syntax_category_ref", "syntax_category_ref="),
    ("probe_variant_ref", "probe_variant_ref="),
    ("safe_to_send", "safe_to_send="),
    ("payload_shape_ref", "payload_shape_ref="),
    ("oracle_ref", "oracle_ref="),
    ("negative_control_presence_ref", "negative_control_presence_ref="),
)


@dataclass(frozen=True)
class PooledSlotConfig:
    language_model_weight: float = 0.15
    slot_weight: float = 1.0
    balance_weight: float = 0.01
    label_smoothing: float = 0.0


def _slot_values(target_tokens: Sequence[str]) -> dict[str, str]:
    body = [str(token) for token in target_tokens]
    if body[:1] != [TARGET_BOS] or body[-1:] != [TARGET_EOS]:
        raise ValueError("target stream must have Rule-IR BOS/EOS")
    values: dict[str, str] = {}
    prefixes = tuple(SLOT_PREFIXES)
    for token in body[1:-1]:
        for name, prefix in prefixes:
            if token.startswith(prefix):
                if name in values:
                    raise ValueError(f"duplicate Rule-IR slot: {name}")
                values[name] = token
                break
        else:
            raise ValueError("target contains a non-abstract Rule-IR token")
    missing = [name for name, _ in SLOT_PREFIXES if name not in values]
    if missing:
        raise ValueError("target missing Rule-IR slots: " + ",".join(missing))
    return values


def build_slot_candidates(vocabulary: Mapping[str, int]) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    for name, prefix in SLOT_PREFIXES:
        ids = tuple(sorted(int(index) for token, index in vocabulary.items() if str(token).startswith(prefix)))
        if not ids:
            raise ValueError(f"locked vocabulary has no candidates for {name}")
        result[name] = ids
    return result


def _context_batch(
    records: Sequence[Mapping[str, Any]],
    vocabulary: Mapping[str, int],
    device: torch.device,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not records:
        raise ValueError("pooled slot batch cannot be empty")
    unknown = int(vocabulary[UNK])
    sequences = [[int(vocabulary.get(str(token), unknown)) for token in row.get("context_tokens", [])] for row in records]
    if any(not sequence for sequence in sequences):
        raise ValueError("pooled slot context cannot be empty")
    width = min(max(len(sequence) for sequence in sequences), max(1, int(max_length)))
    pad = int(vocabulary[PAD])
    ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device)
    valid = torch.zeros((len(sequences), width), dtype=torch.bool, device=device)
    for index, sequence in enumerate(sequences):
        clipped = sequence[:width]
        ids[index, : len(clipped)] = torch.tensor(clipped, dtype=torch.long, device=device)
        valid[index, : len(clipped)] = True
    return ids, valid


class PooledRuleIRDecoder(nn.Module):
    """Causal backbone plus full-context, slot-conditioned heads."""

    def __init__(self, *, vocab_size: int, config: CausalMoEConfig, slot_candidates: Mapping[str, Sequence[int]]) -> None:
        super().__init__()
        self.config = config
        self.backbone = CausalMoELanguageModel(vocab_size=int(vocab_size), config=config)
        self.slot_candidates = {str(name): tuple(int(value) for value in values) for name, values in slot_candidates.items()}
        expected = {name for name, _ in SLOT_PREFIXES}
        if set(self.slot_candidates) != expected:
            raise ValueError("slot candidates must cover exactly the Rule-IR slots")
        self.pool_score = nn.Linear(config.d_model, 1)
        self.pool_projection = nn.Linear(config.d_model * 3, config.d_model)
        self.slot_queries = nn.Parameter(torch.zeros(len(SLOT_PREFIXES), config.d_model))
        nn.init.normal_(self.slot_queries, mean=0.0, std=float(config.initializer_range))
        self.slot_heads = nn.ModuleDict({name: nn.Linear(config.d_model, len(values)) for name, values in self.slot_candidates.items()})

    def _pool(self, hidden: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        mask = valid_mask.bool()
        scores = self.pool_score(hidden).squeeze(-1).masked_fill(~mask, float("-inf"))
        attention = scores.softmax(dim=-1)
        attention = attention.masked_fill(~mask, 0.0)
        weighted = (hidden * attention.unsqueeze(-1)).sum(dim=1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
        mean = (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom
        last_index = mask.sum(dim=1).clamp_min(1) - 1
        boundary = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_index]
        return torch.tanh(self.pool_projection(torch.cat((weighted, mean, boundary), dim=-1)))

    def forward(self, context_ids: torch.Tensor, *, valid_mask: torch.Tensor | None = None) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if valid_mask is None:
            valid_mask = torch.ones_like(context_ids, dtype=torch.bool)
        hidden, balance = self.backbone.forward_hidden(context_ids, valid_mask=valid_mask)
        pooled = self._pool(hidden, valid_mask)
        logits: dict[str, torch.Tensor] = {}
        for index, (name, _) in enumerate(SLOT_PREFIXES):
            query = pooled + self.slot_queries[index].unsqueeze(0)
            logits[name] = self.slot_heads[name](query)
        return logits, balance

    def predict_slot_ids(self, context_ids: torch.Tensor, *, valid_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        logits, _ = self(context_ids, valid_mask=valid_mask)
        return {name: values.argmax(dim=-1) for name, values in logits.items()}

    def decode_predictions(self, slot_indices: Mapping[str, torch.Tensor], *, vocabulary: Mapping[str, int]) -> list[list[str]]:
        reverse = {int(index): str(token) for token, index in vocabulary.items()}
        first = next(iter(slot_indices.values()))
        rows: list[list[str]] = []
        for row_index in range(first.shape[0]):
            tokens = [TARGET_BOS]
            for name, _ in SLOT_PREFIXES:
                local_index = int(slot_indices[name][row_index].detach().cpu())
                tokens.append(reverse.get(self.slot_candidates[name][local_index], UNK))
            tokens.append(TARGET_EOS)
            rows.append(tokens)
        return rows


def _slot_targets(records: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], candidates: Mapping[str, Sequence[int]], device: torch.device) -> dict[str, torch.Tensor]:
    values = [_slot_values(row.get("target_tokens") or []) for row in records]
    targets: dict[str, torch.Tensor] = {}
    for name, _ in SLOT_PREFIXES:
        candidate_ids = tuple(int(value) for value in candidates[name])
        lookup = {token_id: offset for offset, token_id in enumerate(candidate_ids)}
        ids: list[int] = []
        for row in values:
            token_id = int(vocabulary[row[name]])
            if token_id not in lookup:
                raise ValueError(f"target token outside candidate set for {name}")
            ids.append(lookup[token_id])
        targets[name] = torch.tensor(ids, dtype=torch.long, device=device)
    return targets


def train_pooled_rule_ir(
    records: Sequence[Mapping[str, Any]],
    vocabulary: Mapping[str, int],
    device: torch.device,
    *,
    seed: int,
    config: CausalMoEConfig,
    slot_config: PooledSlotConfig = PooledSlotConfig(),
    epochs: int = 8,
    learning_rate: float = 2e-4,
    batch_size: int = 16,
) -> PooledRuleIRDecoder:
    if not records:
        raise ValueError("pooled Rule-IR decoder cannot train on empty records")
    if not 0.0 < float(slot_config.language_model_weight) <= 10.0 or not 0.0 < float(slot_config.slot_weight) <= 10.0:
        raise ValueError("objective weights outside bounds")
    if not 0.0 <= float(slot_config.label_smoothing) <= 0.5:
        raise ValueError("label smoothing outside bounds")
    torch.manual_seed(int(seed))
    candidates = build_slot_candidates(vocabulary)
    model = PooledRuleIRDecoder(vocab_size=len(vocabulary), config=config, slot_candidates=candidates).to(device)
    effective_batch_size = max(1, int(batch_size))
    pad = int(vocabulary[PAD])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=0.01)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    # Do not materialize the full 1k–10k row corpus as one attention batch.
    # Attention memory scales with batch × sequence²; the previous full-batch
    # implementation could consume an entire 80 GiB A800 before the first
    # optimizer step, which is an engineering failure rather than evidence.
    import random

    for epoch in range(int(epochs)):
        order = list(range(len(records)))
        random.Random(int(seed) + epoch).shuffle(order)
        epoch_loss = 0.0
        epoch_batches = 0
        for start in range(0, len(order), effective_batch_size):
            batch_records = [records[index] for index in order[start : start + effective_batch_size]]
            full_ids, full_valid = _batch(batch_records, vocabulary, device, max_length=config.max_length)
            context_ids, context_valid = _context_batch(batch_records, vocabulary, device, max_length=config.max_length)
            slot_targets = _slot_targets(batch_records, vocabulary, candidates, device)
            model.train()
            lm_logits, lm_balance = model.backbone(full_ids[:, :-1], valid_mask=full_valid[:, :-1])
            labels = full_ids[:, 1:]
            label_valid = full_valid[:, 1:]
            labels = labels.masked_fill(~label_valid, pad)
            lm_loss = F.cross_entropy(
                lm_logits.reshape(-1, lm_logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=pad,
                label_smoothing=float(slot_config.label_smoothing),
            )
            slot_logits, slot_balance = model(context_ids, valid_mask=context_valid)
            slot_losses = [
                F.cross_entropy(slot_logits[name], slot_targets[name], label_smoothing=float(slot_config.label_smoothing))
                for name, _ in SLOT_PREFIXES
            ]
            slot_loss = torch.stack(slot_losses).mean()
            loss = float(slot_config.language_model_weight) * lm_loss + float(slot_config.slot_weight) * slot_loss + float(slot_config.balance_weight) * (lm_balance + slot_balance)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            epoch_batches += 1
        current = epoch_loss / max(epoch_batches, 1)
        if current < best_loss:
            best_loss = current
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_pooled_rule_ir(
    model: PooledRuleIRDecoder,
    records: Sequence[Mapping[str, Any]],
    vocabulary: Mapping[str, int],
    device: torch.device,
    *,
    batch_size: int = 32,
) -> list[list[str]]:
    if not records:
        return []
    rows: list[list[str]] = []
    effective_batch_size = max(1, int(batch_size))
    with torch.inference_mode():
        for start in range(0, len(records), effective_batch_size):
            batch = records[start : start + effective_batch_size]
            ids, valid = _context_batch(batch, vocabulary, device, max_length=model.config.max_length)
            predictions = model.predict_slot_ids(ids, valid_mask=valid)
            rows.extend(model.decode_predictions(predictions, vocabulary=vocabulary))
    return rows


def evaluate_pooled_rule_ir(
    model: PooledRuleIRDecoder,
    records: Sequence[Mapping[str, Any]],
    vocabulary: Mapping[str, int],
    device: torch.device,
    *,
    batch_size: int = 32,
) -> dict[str, Any]:
    if not records:
        return {"rows": 0, "sequence_exact_accuracy": None, "slot_accuracy": {}, "negative_false_allow": 0}
    predicted = predict_pooled_rule_ir(model, records, vocabulary, device, batch_size=batch_size)
    slot_correct = {name: 0 for name, _ in SLOT_PREFIXES}
    exact = ask_total = ask_correct = repair_total = repair_correct = abstain_total = abstain_correct = positive_total = positive_correct = positive_action_total = positive_action_correct = false_allow = 0
    for row, guess in zip(records, predicted):
        expected = [str(token) for token in row.get("target_tokens") or []]
        expected_slots = _slot_values(expected)
        guessed_slots = _slot_values(guess)
        exact += int(guess == expected)
        for name, _ in SLOT_PREFIXES:
            slot_correct[name] += int(expected_slots[name] == guessed_slots[name])
        expected_question = expected_slots["question"].split("=", 1)[1]
        guessed_question = guessed_slots["question"].split("=", 1)[1]
        expected_action = expected_slots["next_action"].split("=", 1)[1]
        guessed_action = guessed_slots["next_action"].split("=", 1)[1]
        expected_safe = expected_slots["safe_to_send"].endswith("=1")
        guessed_safe = guessed_slots["safe_to_send"].endswith("=1")
        ask_total += int(expected_question.startswith("ask_"))
        ask_correct += int(expected_question.startswith("ask_") and guessed_question == expected_question)
        repair_total += int(expected_action == "repair")
        repair_correct += int(expected_action == "repair" and guessed_action == "repair")
        abstain_total += int(expected_action == "abstain")
        abstain_correct += int(expected_action == "abstain" and guessed_action == "abstain")
        positive_action_total += int(expected_action in {"select_probe_variant", "replay"})
        positive_action_correct += int(expected_action in {"select_probe_variant", "replay"} and guessed_action == expected_action)
        positive_total += int(expected_safe)
        positive_correct += int(expected_safe and guessed_safe)
        false_allow += int((not expected_safe) and guessed_safe)
    return {
        "rows": len(records),
        "sequence_exact_accuracy": round(exact / max(len(records), 1), 6),
        "slot_accuracy": {name: round(slot_correct[name] / max(len(records), 1), 6) for name, _ in SLOT_PREFIXES},
        "ask_recall": round(ask_correct / max(ask_total, 1), 6) if ask_total else None,
        "repair_recall": round(repair_correct / max(repair_total, 1), 6) if repair_total else None,
        "abstain_recall": round(abstain_correct / max(abstain_total, 1), 6) if abstain_total else None,
        "positive_action_recall": round(positive_action_correct / max(positive_action_total, 1), 6) if positive_action_total else None,
        "positive_recall": round(positive_correct / max(positive_total, 1), 6) if positive_total else None,
        "negative_false_allow": false_allow,
    }


__all__ = [
    "PooledRuleIRDecoder",
    "PooledSlotConfig",
    "SCHEMA_VERSION",
    "SLOT_PREFIXES",
    "build_slot_candidates",
    "evaluate_pooled_rule_ir",
    "predict_pooled_rule_ir",
    "train_pooled_rule_ir",
]
