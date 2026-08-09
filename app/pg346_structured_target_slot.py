"""PG-346 structured Rule-IR target-slot decoder.

This is an auxiliary decoder head over the existing causal Transformer-MoE.
The backbone still trains a causal next-token objective over the complete
abstract context/target stream.  The slot heads read only the final abstract
boundary token in the context and predict the fixed Rule-IR fields separately.
This preserves the seven-axis context information while avoiding an
autoregressive target stream being treated as one long undifferentiated class
sequence during inference.

Only abstract tokens are accepted.  Raw payloads, response bodies, evaluator
answers, routes and family literals are rejected by the caller/collector and
are not represented here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .pg293_failure_next_action import PAD, TARGET_BOS, TARGET_EOS, UNK
from .pg295_causal_moe import CausalMoEConfig, CausalMoELanguageModel, _batch, _sequence


SCHEMA_VERSION = "pg346-structured-target-slot-v1"

# Keep the Rule-IR order explicit.  The names are ontology slots, not
# vulnerability/family labels and do not expose wire literals.
SLOT_PREFIXES: tuple[tuple[str, str], ...] = (
    ("question", "question="),
    ("next_action", "next_action="),
    ("repair_action", "repair_action="),
    ("transport_ref", "transport_ref="),
    ("field_role_ref", "field_role_ref="),
    ("encoding_ref", "encoding_ref="),
    ("probe_variant_ref", "probe_variant_ref="),
    ("safe_to_send", "safe_to_send="),
)


@dataclass(frozen=True)
class StructuredSlotConfig:
    """Training weights for the joint LM + slot objective."""

    language_model_weight: float = 0.25
    slot_weight: float = 1.0
    balance_weight: float = 0.01


def _slot_values(target_tokens: Sequence[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    body = [str(token) for token in target_tokens]
    if body[:1] != [TARGET_BOS] or body[-1:] != [TARGET_EOS]:
        raise ValueError("target stream must have Rule-IR BOS/EOS")
    for token in body[1:-1]:
        for name, prefix in SLOT_PREFIXES:
            if token.startswith(prefix):
                if name in values:
                    raise ValueError(f"duplicate Rule-IR slot: {name}")
                values[name] = token
                break
        else:
            raise ValueError("target contains non-abstract Rule-IR token")
    missing = [name for name, _ in SLOT_PREFIXES if name not in values]
    if missing:
        raise ValueError("target missing Rule-IR slots: " + ",".join(missing))
    return values


def build_slot_candidates(vocabulary: Mapping[str, int]) -> dict[str, tuple[int, ...]]:
    """Build fixed per-slot candidate IDs from the locked abstract vocabulary."""

    result: dict[str, tuple[int, ...]] = {}
    for name, prefix in SLOT_PREFIXES:
        ids = tuple(sorted(int(index) for token, index in vocabulary.items() if str(token).startswith(prefix)))
        if not ids:
            raise ValueError(f"locked vocabulary has no candidates for {name}")
        result[name] = ids
    return result


def _context_batch(records: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], device: torch.device, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not records:
        raise ValueError("structured slot batch cannot be empty")
    sequences = [[int(vocabulary.get(str(token), vocabulary[UNK])) for token in row.get("context_tokens", [])] for row in records]
    if any(not sequence for sequence in sequences):
        raise ValueError("structured slot context cannot be empty")
    width = min(max(len(sequence) for sequence in sequences), max(1, int(max_length)))
    pad = int(vocabulary[PAD])
    ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device)
    valid = torch.zeros((len(sequences), width), dtype=torch.bool, device=device)
    for index, sequence in enumerate(sequences):
        clipped = sequence[:width]
        ids[index, : len(clipped)] = torch.tensor(clipped, dtype=torch.long, device=device)
        valid[index, : len(clipped)] = True
    return ids, valid


class StructuredTargetSlotDecoder(nn.Module):
    """Causal backbone plus fixed Rule-IR slot heads."""

    def __init__(self, *, vocab_size: int, config: CausalMoEConfig, slot_candidates: Mapping[str, Sequence[int]]) -> None:
        super().__init__()
        self.config = config
        self.backbone = CausalMoELanguageModel(vocab_size=int(vocab_size), config=config)
        self.slot_candidates = {str(name): tuple(int(value) for value in values) for name, values in slot_candidates.items()}
        expected = {name for name, _ in SLOT_PREFIXES}
        if set(self.slot_candidates) != expected:
            raise ValueError("slot candidates must cover exactly the Rule-IR slots")
        self.slot_heads = nn.ModuleDict({name: nn.Linear(config.d_model, len(values)) for name, values in self.slot_candidates.items()})

    def forward(self, context_ids: torch.Tensor, *, valid_mask: torch.Tensor | None = None) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        hidden, balance = self.backbone.forward_hidden(context_ids, valid_mask=valid_mask)
        if valid_mask is None:
            last = torch.full((context_ids.shape[0],), context_ids.shape[1] - 1, dtype=torch.long, device=context_ids.device)
        else:
            last = valid_mask.bool().sum(dim=1).clamp_min(1) - 1
        query = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        return {name: head(query) for name, head in self.slot_heads.items()}, balance

    def predict_slot_ids(self, context_ids: torch.Tensor, *, valid_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        logits, _ = self(context_ids, valid_mask=valid_mask)
        return {name: values.argmax(dim=-1) for name, values in logits.items()}

    def decode_predictions(self, slot_indices: Mapping[str, torch.Tensor], *, vocabulary: Mapping[str, int]) -> list[list[str]]:
        reverse = {int(index): str(token) for token, index in vocabulary.items()}
        rows: list[list[str]] = []
        batch = next(iter(slot_indices.values())).shape[0]
        for row_index in range(batch):
            tokens = [TARGET_BOS]
            for name, _ in SLOT_PREFIXES:
                local_index = int(slot_indices[name][row_index].detach().cpu())
                token_id = self.slot_candidates[name][local_index]
                tokens.append(reverse.get(token_id, UNK))
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


def train_structured_slot_decoder(
    records: Sequence[Mapping[str, Any]],
    vocabulary: Mapping[str, int],
    device: torch.device,
    *,
    seed: int,
    config: CausalMoEConfig,
    slot_config: StructuredSlotConfig = StructuredSlotConfig(),
    epochs: int = 16,
    learning_rate: float = 2e-4,
) -> StructuredTargetSlotDecoder:
    if not records:
        raise ValueError("structured slot decoder cannot train on empty records")
    if not 0.0 < float(slot_config.language_model_weight) <= 10.0 or not 0.0 < float(slot_config.slot_weight) <= 10.0:
        raise ValueError("structured objective weights outside bounds")
    torch.manual_seed(int(seed))
    candidates = build_slot_candidates(vocabulary)
    model = StructuredTargetSlotDecoder(vocab_size=len(vocabulary), config=config, slot_candidates=candidates).to(device)
    max_length = int(config.max_length)
    full_ids, full_valid = _batch(records, vocabulary, device, max_length=max_length)
    context_ids, context_valid = _context_batch(records, vocabulary, device, max_length=max_length)
    slot_targets = _slot_targets(records, vocabulary, candidates, device)
    pad = int(vocabulary[PAD])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=0.01)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    for _ in range(int(epochs)):
        model.train()
        lm_logits, lm_balance = model.backbone(full_ids[:, :-1], valid_mask=full_valid[:, :-1])
        labels = full_ids[:, 1:]
        label_valid = full_valid[:, 1:]
        labels = labels.masked_fill(~label_valid, pad)
        lm_loss = F.cross_entropy(lm_logits.reshape(-1, lm_logits.shape[-1]), labels.reshape(-1), ignore_index=pad)
        slot_logits, slot_balance = model(context_ids, valid_mask=context_valid)
        slot_losses = [F.cross_entropy(slot_logits[name], slot_targets[name]) for name, _ in SLOT_PREFIXES]
        slot_loss = torch.stack(slot_losses).mean()
        loss = float(slot_config.language_model_weight) * lm_loss + float(slot_config.slot_weight) * slot_loss + float(slot_config.balance_weight) * (lm_balance + slot_balance)
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


def predict_structured_slots(model: StructuredTargetSlotDecoder, records: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], device: torch.device) -> list[list[str]]:
    if not records:
        return []
    context_ids, context_valid = _context_batch(records, vocabulary, device, max_length=model.config.max_length)
    with torch.inference_mode():
        predictions = model.predict_slot_ids(context_ids, valid_mask=context_valid)
        return model.decode_predictions(predictions, vocabulary=vocabulary)


def evaluate_structured_slots(model: StructuredTargetSlotDecoder, records: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], device: torch.device) -> dict[str, Any]:
    if not records:
        return {"rows": 0, "sequence_exact_accuracy": None, "slot_accuracy": {}, "negative_false_allow": 0}
    predicted = predict_structured_slots(model, records, vocabulary, device)
    slot_correct = {name: 0 for name, _ in SLOT_PREFIXES}
    slot_total = {name: 0 for name, _ in SLOT_PREFIXES}
    exact = 0
    ask_total = ask_correct = repair_total = repair_correct = variant_total = variant_correct = positive_total = positive_correct = false_allow = 0
    variant_actions = {"select_probe_variant", "assemble_rule_ir", "send_probe"}
    for row, guess in zip(records, predicted):
        expected = [str(token) for token in row.get("target_tokens") or []]
        exact += int(guess == expected)
        expected_slots = _slot_values(expected)
        guessed_slots = _slot_values(guess)
        for name, _ in SLOT_PREFIXES:
            slot_total[name] += 1
            slot_correct[name] += int(expected_slots[name] == guessed_slots[name])
        expected_question = expected_slots["question"].split("=", 1)[1]
        guessed_question = guessed_slots["question"].split("=", 1)[1]
        expected_action = expected_slots["next_action"].split("=", 1)[1]
        guessed_action = guessed_slots["next_action"].split("=", 1)[1]
        expected_variant = expected_slots["probe_variant_ref"].split("=", 1)[1]
        guessed_variant = guessed_slots["probe_variant_ref"].split("=", 1)[1]
        expected_safe = expected_slots["safe_to_send"].endswith("=1")
        guessed_safe = guessed_slots["safe_to_send"].endswith("=1")
        ask_total += int(expected_question.startswith("ask_"))
        ask_correct += int(expected_question.startswith("ask_") and guessed_question == expected_question)
        repair_total += int(expected_action == "repair")
        repair_correct += int(expected_action == "repair" and guessed_action == "repair")
        variant_total += int(expected_action in variant_actions)
        variant_correct += int(expected_action in variant_actions and guessed_action in variant_actions)
        positive_total += int(expected_safe)
        positive_correct += int(expected_safe and guessed_safe)
        false_allow += int((not expected_safe) and guessed_safe)
    return {
        "rows": len(records),
        "sequence_exact_accuracy": round(exact / max(len(records), 1), 6),
        "slot_accuracy": {name: round(slot_correct[name] / max(slot_total[name], 1), 6) for name, _ in SLOT_PREFIXES},
        "ask_recall": round(ask_correct / max(ask_total, 1), 6) if ask_total else None,
        "repair_recall": round(repair_correct / max(repair_total, 1), 6) if repair_total else None,
        "variant_recall": round(variant_correct / max(variant_total, 1), 6) if variant_total else None,
        "positive_recall": round(positive_correct / max(positive_total, 1), 6) if positive_total else None,
        "negative_false_allow": false_allow,
    }


__all__ = [
    "SCHEMA_VERSION",
    "SLOT_PREFIXES",
    "StructuredSlotConfig",
    "StructuredTargetSlotDecoder",
    "build_slot_candidates",
    "evaluate_structured_slots",
    "predict_structured_slots",
    "train_structured_slot_decoder",
]
