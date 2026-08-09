"""PG-375 composed Rule-IR candidate trainer.

This runner is deliberately candidate-only.  It consumes the abstract
PG-364 implementation-holdout rows, validates the split before constructing
an optimizer, and keeps the evaluator/raw-wire sidecars out of the model.
The model has two related Rule-IR paths:

* a target-only next-token lane on the shared causal MoE backbone; and
* an autoregressive 13-slot composition decoder.  The decoder is conditioned
  on the abstract page boundary and the previously emitted slot value.  The
  independent slot heads are retained as an auxiliary signal, while ASK,
  repair and negative heads read the composition summary.

The default command is plan-only.  ``--cpu-smoke`` is a bounded local tensor
  check; ``--remote-candidate`` is the only CUDA path and requires the
  explicit BLACKBOX_REMOTE_A800_TRAIN=1/CUDA_VISIBLE_DEVICES=0 gate.  This
  module never starts Docker, contacts a target, or loads raw wire material.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg293_failure_next_action import PAD, TARGET_BOS, TARGET_EOS, UNK
from app.pg295_causal_moe import CausalMoEConfig, CausalMoELanguageModel
from scripts.build_pg362_full_rule_ir_dataset import SLOTS
from scripts.run_pg370_multitask_moe_candidate import (
    PROMOTION_KEYS,
    _safe_abstract_row,
    _sha_file,
    _sha_json,
    _target_values,
)

SCHEMA_VERSION = "pg375-composed-rule-ir-candidate-v1"
DEFAULT_DATASET = ROOT / "research" / "pg364_compositional_rule_ir_dataset_v1.json"
DEFAULT_AUDIT = ROOT / "research" / "pg364_compositional_rule_ir_audit_v1.json"
SEEDS = (37501, 37502, 37503)

# These fragments identify data that is forbidden in a model-visible row.  A
# token such as ``oracle_ref=typed_effect`` is an abstract Rule-IR slot and is
# intentionally allowed; only literal/raw sidecar shapes are rejected.
RAW_FRAGMENT_MARKERS = (
    "raw_payload=",
    "payload=",
    "response_body=",
    "response_body_text=",
    "raw_response=",
    "wire=",
    "evaluator=",
    "oracle=",
    "route_literal=",
    "family=",
    "http://",
    "https://",
)


def _normalise(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _slot_values_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    values: dict[str, set[str]] = {slot: set() for slot in SLOTS}
    for row in rows:
        parsed = row.get("_target_values") or _target_values(row["target_tokens"])
        for slot in SLOTS:
            values[slot].add(str(parsed[slot]))
    missing = [slot for slot in SLOTS if not values[slot]]
    if missing:
        raise ValueError("train rows are missing Rule-IR slots: " + ",".join(missing))
    return {slot: {value: index for index, value in enumerate(sorted(items))} for slot, items in values.items()}


def build_train_vocabulary(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Build the coordinate system from *train rows only*."""

    tokens = {PAD, UNK}
    for row in rows:
        tokens.update(str(token) for token in row.get("context_tokens", []))
        tokens.update(str(token) for token in row.get("target_tokens", []))
    return {token: index for index, token in enumerate([PAD, UNK, *sorted(tokens - {PAD, UNK})])}


def _vocabulary_gaps(
    train_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
    vocabulary: Mapping[str, int],
    slot_classes: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    unknown_tokens = sorted(
        {
            str(token)
            for row in holdout_rows
            for token in [*row.get("context_tokens", []), *row.get("target_tokens", [])]
            if str(token) not in vocabulary
        }
    )
    unknown_slots: dict[str, list[str]] = {}
    for slot in SLOTS:
        values = sorted(
            {
                str((row.get("_target_values") or _target_values(row["target_tokens"]))[slot])
                for row in holdout_rows
                if str((row.get("_target_values") or _target_values(row["target_tokens"]))[slot]) not in slot_classes[slot]
            }
        )
        if values:
            unknown_slots[slot] = values
    return {
        "unknown_token_count": len(unknown_tokens),
        "unknown_tokens_sha256": _sha_json(unknown_tokens),
        "unknown_slot_value_count": sum(len(values) for values in unknown_slots.values()),
        "unknown_slot_values_sha256": {slot: _sha_json(values) for slot, values in sorted(unknown_slots.items())},
        "blocked": bool(unknown_tokens or unknown_slots),
    }


def _target_shape(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(token) for token in row.get("target_tokens", []))


def _context_shape(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(token) for token in row.get("context_tokens", []))


def audit_pg375_contract(
    dataset: Mapping[str, Any],
    train_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a strict, read-only data gate evaluated before any optimizer."""

    failures: list[str] = []
    dataset_status = str(dataset.get("status", ""))
    if dataset_status == "blocked_incomplete":
        failures.append("dataset_status_blocked_incomplete")
    # A structurally clean derived split is not a capability-training grant.
    # PG-375 strict-filtered artifacts deliberately carry an explicit closed
    # capability gate while they wait for reviewed typed/fresh source rows.
    # Do not infer permission from row count or split cleanliness.
    if "capability_training_allowed" in dataset and dataset.get("capability_training_allowed") is not True:
        failures.append("capability_training_not_authorized")
    if isinstance(dataset.get("source_contract"), Mapping):
        source_contract = dataset["source_contract"]
        if any(source_contract.get(key) is not True for key in ("operator_reviewed", "typed_evaluator_complete", "fresh_reset_role_attested", "capability_training_eligible")):
            failures.append("source_capability_gate_incomplete")
    if not train_rows:
        failures.append("empty_train_split")
    if not holdout_rows:
        failures.append("empty_holdout_split")

    # Validate each row's complete 13-slot target and context firewall.  Rows
    # have already been reduced by _safe_abstract_row, so this checks only
    # model-visible fields.
    malformed = 0
    for row in [*train_rows, *holdout_rows]:
        try:
            values = row.get("_target_values") or _target_values(row["target_tokens"])
            if set(values) != set(SLOTS):
                malformed += 1
            if row.get("context_firewall") not in (None, {"forbidden_token_count": 0, "sidecars_off_context": True}):
                malformed += 1
        except (KeyError, TypeError, ValueError):
            malformed += 1
    if malformed:
        failures.append("malformed_rule_ir_rows")

    # Strict train-only coordinate system.  No target value or token may be
    # silently introduced from the implementation holdout.
    vocabulary = build_train_vocabulary(train_rows)
    try:
        slot_classes = _slot_values_from_rows(train_rows)
    except ValueError:
        slot_classes = {slot: {} for slot in SLOTS}
        failures.append("train_slot_coverage_missing")
    gaps = _vocabulary_gaps(train_rows, holdout_rows, vocabulary, slot_classes)
    if gaps["blocked"]:
        failures.append("train_only_vocabulary_gap")

    # Exact context/target reuse is a leakage path even when implementation
    # group hashes are disjoint.  Context reuse is reported separately and is
    # also blocked: an implementation holdout must require a new page
    # observation, not a memorised context-to-Rule-IR mapping.
    train_full = {_context_shape(row) + ("[TARGET_SEPARATOR]",) + _target_shape(row) for row in train_rows}
    holdout_full = {_context_shape(row) + ("[TARGET_SEPARATOR]",) + _target_shape(row) for row in holdout_rows}
    train_context = {_context_shape(row) for row in train_rows}
    holdout_context = {_context_shape(row) for row in holdout_rows}
    train_target = {_target_shape(row) for row in train_rows}
    holdout_target = {_target_shape(row) for row in holdout_rows}
    exact_overlap = len(train_full & holdout_full)
    context_overlap = len(train_context & holdout_context)
    target_overlap = len(train_target & holdout_target)
    if exact_overlap:
        failures.append("cross_split_exact_sequence_overlap")
    if context_overlap:
        failures.append("cross_split_context_overlap")

    # The manifest's hashed implementation groups remain an independent
    # contract.  Literal implementation IDs never enter model rows.
    split_contract = dataset.get("split_contract") if isinstance(dataset.get("split_contract"), Mapping) else {}
    train_groups = {str(value) for value in split_contract.get("train_group_hashes", [])}
    holdout_groups = {str(value) for value in split_contract.get("holdout_group_hashes", [])}
    group_disjoint = bool(train_groups and holdout_groups and not (train_groups & holdout_groups))
    if not group_disjoint:
        failures.append("implementation_group_overlap_or_missing")

    declared = dataset.get("vocabulary") if isinstance(dataset.get("vocabulary"), Mapping) else {}
    declared_context = set(str(value) for value in declared.get("context_tokens", []))
    declared_target = set(str(value) for value in declared.get("target_tokens", []))
    declared_missing_from_rows = sorted((set(vocabulary) - {PAD, UNK}) - (declared_context | declared_target))
    # Hand-built unit fixtures may omit the optional ontology inventory.  A
    # real PG-364 manifest has a non-empty inventory; only then can a missing
    # declared token be an integrity failure.
    if (declared_context or declared_target) and declared_missing_from_rows:
        failures.append("declared_manifest_missing_train_token")

    failures = sorted(set(failures))
    return {
        "status": "passed" if not failures else "blocked",
        "failures": failures,
        "dataset_status": dataset_status,
        "counts": {"train_rows": len(train_rows), "holdout_rows": len(holdout_rows)},
        "vocabulary_scope": "train_only",
        "vocabulary_size": len(vocabulary),
        "vocabulary_gaps": gaps,
        "slot_order": list(SLOTS),
        "slot_class_counts": {slot: len(slot_classes.get(slot, {})) for slot in SLOTS},
        "cross_split": {
            "exact_sequence_overlap_count": exact_overlap,
            "context_overlap_count": context_overlap,
            "target_overlap_count": target_overlap,
            "implementation_group_disjoint": group_disjoint,
        },
        "declared_manifest": {
            "context_count": len(declared_context),
            "target_count": len(declared_target),
            "missing_train_tokens": len(declared_missing_from_rows),
        },
        "training_eligible": not bool(failures),
    }


def load_pg364_dataset(
    dataset_path: str | Path = DEFAULT_DATASET,
    audit_path: str | Path | None = DEFAULT_AUDIT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Load only sanitized PG-364 rows and return the strict contract."""

    path = Path(dataset_path)
    dataset = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(dataset, Mapping):
        raise ValueError("PG-364 dataset must be an object")
    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for index, raw in enumerate(dataset.get("records") or []):
        if not isinstance(raw, Mapping):
            raise ValueError(f"PG-364 row {index} is not an object")
        # The helper copies context/target only and checks the raw firewall.
        item = _safe_abstract_row(raw, source="pg364")
        if item["split"] == "train":
            train.append(item)
        elif item["split"] == "implementation_holdout":
            holdout.append(item)
        else:
            raise ValueError(f"PG-364 row {index} has unsupported split {item['split']!r}")
    audit: dict[str, Any] = {}
    if audit_path is not None and Path(audit_path).exists():
        loaded_audit = json.loads(Path(audit_path).read_text(encoding="utf-8-sig"))
        if isinstance(loaded_audit, Mapping):
            audit = dict(loaded_audit)
            if str(audit.get("status", "")).startswith("blocked"):
                # A caller may not replace a blocked audit with a hand-built
                # row list.  Keep the status in the in-memory manifest so the
                # strict contract fails before vocabulary/model construction.
                dataset = dict(dataset)
                dataset["status"] = "blocked_incomplete"
    locks = {
        "dataset_path": _normalise(path),
        "dataset_sha256": _sha_file(path),
        "audit_path": _normalise(Path(audit_path)) if audit_path is not None else None,
        "audit_sha256": _sha_file(Path(audit_path)) if audit_path is not None and Path(audit_path).exists() else None,
        "dataset_schema_version": str(dataset.get("schema_version", "")),
    }
    contract = audit_pg375_contract(dataset, train, holdout)
    return train, holdout, {"dataset": dataset, "audit": audit, "locks": locks}, contract


def _pad_context(rows: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = [[int(vocabulary.get(str(token), vocabulary[UNK])) for token in row["context_tokens"]] for row in rows]
    width = max((len(sequence) for sequence in sequences), default=1)
    ids = torch.full((len(sequences), width), int(vocabulary[PAD]), dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), width), dtype=torch.bool, device=device)
    for index, sequence in enumerate(sequences):
        ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[index, : len(sequence)] = True
    return ids, mask


def _pad_lm(rows: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad full context+target while masking labels to target positions only."""

    sequences = [
        [int(vocabulary.get(str(token), vocabulary[UNK])) for token in [*row["context_tokens"], *row["target_tokens"]]]
        for row in rows
    ]
    width = max((len(sequence) for sequence in sequences), default=2)
    ids = torch.full((len(sequences), width), int(vocabulary[PAD]), dtype=torch.long, device=device)
    valid = torch.zeros((len(sequences), width), dtype=torch.bool, device=device)
    target_mask = torch.zeros((len(sequences), max(width - 1, 1)), dtype=torch.bool, device=device)
    for index, sequence in enumerate(sequences):
        ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        valid[index, : len(sequence)] = True
        context_len = len(rows[index]["context_tokens"])
        # labels[j] is predicted from logits[j].  The first target label is at
        # original position context_len and therefore starts at context_len-1;
        # end is exclusive of the final input position and includes EOS.
        if len(sequence) > context_len:
            target_mask[index, max(context_len - 1, 0) : len(sequence) - 1] = True
    return ids, valid, target_mask[:, : max(width - 1, 1)]


def _batch_slot_labels(rows: Sequence[Mapping[str, Any]], slot_classes: Mapping[str, Mapping[str, int]], device: torch.device) -> torch.Tensor:
    labels = []
    for row in rows:
        values = row.get("_target_values") or _target_values(row["target_tokens"])
        labels.append([slot_classes[slot][str(values[slot])] for slot in SLOTS])
    return torch.tensor(labels, dtype=torch.long, device=device)


class ComposedRuleIRModel(nn.Module):
    """Shared causal MoE plus an autoregressive Rule-IR slot decoder."""

    def __init__(
        self,
        *,
        vocab_size: int,
        config: CausalMoEConfig,
        slot_classes: Mapping[str, Mapping[str, int]],
        slot_decoder_layers: int = 2,
        slot_decoder_heads: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.slot_classes = {slot: dict(classes) for slot, classes in slot_classes.items()}
        self.backbone = CausalMoELanguageModel(vocab_size=int(vocab_size), config=config)
        heads = int(slot_decoder_heads or config.n_heads)
        if config.d_model % heads:
            raise ValueError("slot_decoder_heads must divide d_model")
        self.slot_queries = nn.Parameter(torch.zeros(len(SLOTS), config.d_model))
        nn.init.normal_(self.slot_queries, mean=0.0, std=float(config.initializer_range))
        # Index zero is BOS/unknown previous value; classes are shifted by
        # one.  A shared coordinate is required because the previous slot and
        # the current slot can have different class cardinalities.  Reusing a
        # current-slot embedding here would make a valid previous class index
        # out of range (and silently destroy composition learning).
        self.previous_value_embedding = nn.Embedding(max((len(classes) for classes in slot_classes.values()), default=1) + 1, config.d_model)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=heads,
            dim_feedforward=max(config.d_model * 2, 64),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.composition_decoder = nn.TransformerEncoder(decoder_layer, num_layers=max(1, int(slot_decoder_layers)))
        self.slot_heads = nn.ModuleDict({slot: nn.Linear(config.d_model, len(classes)) for slot, classes in slot_classes.items()})
        self.slot_aux_heads = nn.ModuleDict({slot: nn.Linear(config.d_model, len(classes)) for slot, classes in slot_classes.items()})
        summary_dim = config.d_model * 2
        self.ask_head = nn.Linear(summary_dim, 2)
        self.repair_head = nn.Linear(summary_dim, 4)
        self.negative_head = nn.Linear(summary_dim, 2)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)

    def _composition_teacher(self, boundary: torch.Tensor, labels: torch.Tensor | None) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        batch = boundary.shape[0]
        previous = torch.zeros((batch,), dtype=torch.long, device=boundary.device)
        inputs: list[torch.Tensor] = []
        for index, slot in enumerate(SLOTS):
            inputs.append(boundary + self.slot_queries[index].unsqueeze(0) + self.previous_value_embedding(previous))
            if labels is not None:
                previous = labels[:, index] + 1
        sequence = torch.stack(inputs, dim=1)
        hidden = self.composition_decoder(sequence, mask=self._causal_mask(len(SLOTS), sequence.device))
        logits = {slot: self.slot_heads[slot](hidden[:, index]) for index, slot in enumerate(SLOTS)}
        return logits, hidden

    def _composition_greedy(self, boundary: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        batch = boundary.shape[0]
        previous_values = [torch.zeros((batch,), dtype=torch.long, device=boundary.device) for _ in range(len(SLOTS) + 1)]
        logits: dict[str, torch.Tensor] = {}
        hidden_steps: list[torch.Tensor] = []
        # Recompute the causal decoder after each prediction.  This keeps the
        # inference path genuinely autoregressive and makes the feedback path
        # explicit; training uses one teacher-forced pass above.
        for index, slot in enumerate(SLOTS):
            inputs: list[torch.Tensor] = []
            for prior_index, prior_slot in enumerate(SLOTS[: index + 1]):
                inputs.append(boundary + self.slot_queries[prior_index].unsqueeze(0) + self.previous_value_embedding(previous_values[prior_index]))
            sequence = torch.stack(inputs, dim=1)
            hidden = self.composition_decoder(sequence, mask=self._causal_mask(sequence.shape[1], sequence.device))
            current_hidden = hidden[:, -1]
            current_logits = self.slot_heads[slot](current_hidden)
            logits[slot] = current_logits
            hidden_steps.append(current_hidden)
            previous_values[index + 1] = current_logits.argmax(dim=-1).detach() + 1
        return logits, torch.stack(hidden_steps, dim=1)

    def forward(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        *,
        lm_ids: torch.Tensor | None = None,
        lm_mask: torch.Tensor | None = None,
        teacher_slot_labels: torch.Tensor | None = None,
        decode_composition: bool = True,
    ) -> dict[str, Any]:
        context_hidden, context_balance = self.backbone.forward_hidden(context_ids, valid_mask=context_mask)
        lengths = context_mask.long().sum(dim=1).clamp_min(1) - 1
        boundary = context_hidden[torch.arange(context_hidden.shape[0], device=context_hidden.device), lengths]
        if teacher_slot_labels is not None:
            composition, slot_hidden = self._composition_teacher(boundary, teacher_slot_labels)
        elif decode_composition:
            composition, slot_hidden = self._composition_greedy(boundary)
        else:
            composition, slot_hidden = {}, boundary.unsqueeze(1).expand(-1, len(SLOTS), -1)
        summary = slot_hidden.mean(dim=1)
        head_input = torch.cat([boundary, summary], dim=-1)
        output: dict[str, Any] = {
            "composition": composition,
            "slot_aux": {slot: self.slot_aux_heads[slot](boundary) for slot in SLOTS},
            "ask": self.ask_head(head_input),
            "repair": self.repair_head(head_input),
            "negative": self.negative_head(head_input),
            "balance": context_balance,
        }
        if lm_ids is not None:
            lm_hidden, lm_balance = self.backbone.forward_hidden(lm_ids, valid_mask=lm_mask)
            output["lm"] = self.backbone.lm_head(lm_hidden)
            output["balance"] = output["balance"] + lm_balance
        return output


def _entropy(logits: torch.Tensor) -> float:
    probabilities = logits.softmax(dim=-1)
    return float((-(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)).mean().detach().cpu())


def _loss_components(
    model: ComposedRuleIRModel,
    rows: Sequence[Mapping[str, Any]],
    vocabulary: Mapping[str, int],
    slot_classes: Mapping[str, Mapping[str, int]],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    context_ids, context_mask = _pad_context(rows, vocabulary, device)
    lm_ids, lm_mask, target_mask = _pad_lm(rows, vocabulary, device)
    labels = _batch_slot_labels(rows, slot_classes, device)
    output = model(
        context_ids,
        context_mask,
        lm_ids=lm_ids[:, :-1],
        lm_mask=lm_mask[:, :-1],
        teacher_slot_labels=labels,
    )
    lm_labels = lm_ids[:, 1:]
    lm_loss_all = F.cross_entropy(
        output["lm"].reshape(-1, output["lm"].shape[-1]),
        lm_labels.reshape(-1),
        ignore_index=int(vocabulary[PAD]),
        reduction="none",
    ).reshape(lm_labels.shape)
    lm_loss = lm_loss_all[target_mask].mean() if bool(target_mask.any()) else lm_loss_all.mean() * 0.0
    composition_loss = torch.stack([F.cross_entropy(output["composition"][slot], labels[:, index]) for index, slot in enumerate(SLOTS)]).mean()
    aux_loss = torch.stack([F.cross_entropy(output["slot_aux"][slot], labels[:, index]) for index, slot in enumerate(SLOTS)]).mean()
    ask = torch.tensor(
        [int((row.get("_target_values") or _target_values(row["target_tokens"]))["question"].startswith("ask_")) for row in rows],
        dtype=torch.long,
        device=device,
    )
    repair = torch.tensor(
        [{"select_probe_variant": 0, "replay": 1, "repair": 2, "abstain": 3}.get((row.get("_target_values") or _target_values(row["target_tokens"]))["next_action"], 0) for row in rows],
        dtype=torch.long,
        device=device,
    )
    negative = torch.tensor(
        [int((row.get("_target_values") or _target_values(row["target_tokens"]))["safe_to_send"] == "1") for row in rows],
        dtype=torch.long,
        device=device,
    )
    components = {
        "next_token": lm_loss,
        "composition": composition_loss,
        "slot_aux": aux_loss,
        "ask": F.cross_entropy(output["ask"], ask),
        "repair": F.cross_entropy(output["repair"], repair),
        "negative": F.cross_entropy(output["negative"], negative),
        "balance": output["balance"],
    }
    return components, output["lm"]


def _target_kl(student: torch.Tensor, teacher: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    if not bool(target_mask.any()):
        return student.sum() * 0.0
    student_log = F.log_softmax(student[target_mask], dim=-1)
    teacher_prob = F.softmax(teacher[target_mask].detach(), dim=-1)
    return F.kl_div(student_log, teacher_prob, reduction="batchmean")


def _train_stage_a(
    model: ComposedRuleIRModel,
    rows: Sequence[Mapping[str, Any]],
    vocabulary: Mapping[str, int],
    slot_classes: Mapping[str, Mapping[str, int]],
    *,
    epochs: int,
    microbatch: int,
    grad_accum: int,
    lr: float,
    device: torch.device,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=0.01)
    order = list(range(len(rows)))
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(max(1, int(epochs))):
        random.Random(1000 + epoch).shuffle(order)
        for batch_index, start in enumerate(range(0, len(order), max(1, int(microbatch)))):
            batch = [rows[index] for index in order[start : start + max(1, int(microbatch))]]
            components, _ = _loss_components(model, batch, vocabulary, slot_classes, device)
            loss = components["next_token"] / max(1, int(grad_accum))
            loss.backward()
            if (batch_index + 1) % max(1, int(grad_accum)) == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if len(order) and (len(order) // max(1, int(microbatch))) % max(1, int(grad_accum)):
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


def _train_stage_b(
    model: ComposedRuleIRModel,
    teacher: ComposedRuleIRModel,
    rows: Sequence[Mapping[str, Any]],
    vocabulary: Mapping[str, int],
    slot_classes: Mapping[str, Mapping[str, int]],
    *,
    epochs: int,
    microbatch: int,
    grad_accum: int,
    lr: float,
    kl_weight: float,
    weights: Mapping[str, float],
    device: torch.device,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=0.01)
    teacher.eval()
    order = list(range(len(rows)))
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(max(1, int(epochs))):
        random.Random(2000 + epoch).shuffle(order)
        for batch_index, start in enumerate(range(0, len(order), max(1, int(microbatch)))):
            batch = [rows[index] for index in order[start : start + max(1, int(microbatch))]]
            components, student_lm = _loss_components(model, batch, vocabulary, slot_classes, device)
            context_ids, context_mask = _pad_context(batch, vocabulary, device)
            lm_ids, lm_mask, target_mask = _pad_lm(batch, vocabulary, device)
            with torch.no_grad():
                teacher_output = teacher(
                    context_ids,
                    context_mask,
                    lm_ids=lm_ids[:, :-1],
                    lm_mask=lm_mask[:, :-1],
                    teacher_slot_labels=_batch_slot_labels(batch, slot_classes, device),
                    decode_composition=False,
                )
            kl = _target_kl(student_lm, teacher_output["lm"], target_mask)
            total = (
                float(weights.get("next_token", 0.4)) * components["next_token"]
                + float(weights.get("composition", 1.0)) * components["composition"]
                + float(weights.get("slot_aux", 0.25)) * components["slot_aux"]
                + float(weights.get("ask", 1.0)) * components["ask"]
                + float(weights.get("repair", 1.5)) * components["repair"]
                + float(weights.get("negative", 2.0)) * components["negative"]
                + float(kl_weight) * kl
                + float(weights.get("balance", 0.01)) * components["balance"]
            ) / max(1, int(grad_accum))
            total.backward()
            if (batch_index + 1) % max(1, int(grad_accum)) == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if len(order) and (len(order) // max(1, int(microbatch))) % max(1, int(grad_accum)):
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


def _evaluate(
    model: ComposedRuleIRModel,
    rows: Sequence[Mapping[str, Any]],
    vocabulary: Mapping[str, int],
    slot_classes: Mapping[str, Mapping[str, int]],
    device: torch.device,
) -> dict[str, Any]:
    if not rows:
        return {
            "rows": 0,
            "next_token_sequence_exact": None,
            "sequence_exact": None,
            "slot_composition_exact": None,
            "slot_accuracy": None,
            "per_slot": {slot: {"correct": 0, "total": 0, "accuracy": None} for slot in SLOTS},
            "ask_recall": None,
            "repair_recall": None,
            "positive_recall": None,
            "negative_false_allow": 0,
            "predictive_entropy": None,
        }
    reverse = {slot: {index: value for value, index in classes.items()} for slot, classes in slot_classes.items()}
    totals = Counter()
    per_slot = {slot: Counter() for slot in SLOTS}
    entropies: list[float] = []
    model.eval()
    with torch.inference_mode():
        for row in rows:
            context_ids, context_mask = _pad_context([row], vocabulary, device)
            lm_ids, lm_mask, target_mask = _pad_lm([row], vocabulary, device)
            output = model(context_ids, context_mask, lm_ids=lm_ids[:, :-1], lm_mask=lm_mask[:, :-1], decode_composition=True)
            values = row.get("_target_values") or _target_values(row["target_tokens"])
            all_correct = True
            for slot in SLOTS:
                predicted_index = int(output["composition"][slot].argmax(-1).item())
                predicted = reverse[slot][predicted_index]
                correct = int(predicted == str(values[slot]))
                per_slot[slot]["correct"] += correct
                per_slot[slot]["total"] += 1
                all_correct = all_correct and bool(correct)
            totals["composition_exact"] += int(all_correct)
            labels = _batch_slot_labels([row], slot_classes, device)[0]
            lm_positions = torch.nonzero(target_mask[0], as_tuple=False).flatten()
            if len(lm_positions):
                lm_logits = output["lm"][0, lm_positions]
                lm_labels = lm_ids[:, 1:][0, lm_positions]
                totals["next_token_exact"] += int(torch.all(lm_logits.argmax(-1) == lm_labels).item())
                entropies.append(_entropy(lm_logits))
            expected_question = str(values["question"]).startswith("ask_")
            predicted_question = int(output["ask"].argmax(-1).item()) == 1
            totals["ask_total"] += int(expected_question)
            totals["ask_correct"] += int(expected_question and predicted_question)
            expected_repair = str(values["next_action"]) == "repair"
            predicted_repair = int(output["repair"].argmax(-1).item()) == 2
            totals["repair_total"] += int(expected_repair)
            totals["repair_correct"] += int(expected_repair and predicted_repair)
            expected_safe = str(values["safe_to_send"]) == "1"
            predicted_safe = int(output["negative"].argmax(-1).item()) == 1
            totals["positive_total"] += int(expected_safe)
            totals["positive_correct"] += int(expected_safe and predicted_safe)
            totals["negative_total"] += int(not expected_safe)
            totals["negative_false_allow"] += int(not expected_safe and predicted_safe)
    slot_total = sum(per_slot[slot]["total"] for slot in SLOTS)
    return {
        "rows": len(rows),
        "next_token_sequence_exact": round(totals["next_token_exact"] / max(len(rows), 1), 6),
        "sequence_exact": round(totals["composition_exact"] / max(len(rows), 1), 6),
        "slot_composition_exact": round(totals["composition_exact"] / max(len(rows), 1), 6),
        "slot_accuracy": round(sum(per_slot[slot]["correct"] for slot in SLOTS) / max(slot_total, 1), 6),
        "per_slot": {
            slot: {
                "correct": int(per_slot[slot]["correct"]),
                "total": int(per_slot[slot]["total"]),
                "accuracy": round(per_slot[slot]["correct"] / max(per_slot[slot]["total"], 1), 6),
            }
            for slot in SLOTS
        },
        "ask_recall": round(totals["ask_correct"] / max(totals["ask_total"], 1), 6) if totals["ask_total"] else None,
        "repair_recall": round(totals["repair_correct"] / max(totals["repair_total"], 1), 6) if totals["repair_total"] else None,
        "positive_recall": round(totals["positive_correct"] / max(totals["positive_total"], 1), 6) if totals["positive_total"] else None,
        "negative_false_allow": int(totals["negative_false_allow"]),
        "negative_total": int(totals["negative_total"]),
        "predictive_entropy": round(sum(entropies) / max(len(entropies), 1), 6),
    }


def _device_gate(device: str) -> torch.device:
    if str(device) == "cpu":
        return torch.device("cpu")
    if os.environ.get("BLACKBOX_REMOTE_A800_TRAIN") != "1":
        raise RuntimeError("PG-375 CUDA requires BLACKBOX_REMOTE_A800_TRAIN=1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("PG-375 CUDA requires CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("PG-375 requires exactly one visible CUDA device")
    torch.cuda.set_device(0)
    if "A800" not in torch.cuda.get_device_name(0):
        raise RuntimeError("PG-375 requires NVIDIA A800 GPU0")
    return torch.device("cuda:0")


DEFAULT_WEIGHTS = {"next_token": 0.4, "composition": 1.0, "slot_aux": 0.25, "ask": 1.0, "repair": 1.5, "negative": 2.0, "balance": 0.01}


def _hard_gate(worst: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "data_contract": contract.get("status") == "passed",
        "sequence_exact": float(worst.get("sequence_exact_min", 0.0)) >= 0.90,
        "slot_composition_exact": float(worst.get("slot_composition_exact_min", 0.0)) >= 0.90,
        "ask": float(worst.get("ask_recall_min", 0.0)) >= 0.95,
        "repair": float(worst.get("repair_recall_min", 0.0)) >= 0.95,
        "positive": float(worst.get("positive_recall_min", 0.0)) >= 0.95,
        "negative": int(worst.get("negative_false_allow_max", 1)) == 0,
        "entropy": float(worst.get("entropy_relative_drop_max", 1.0)) <= 0.25,
    }
    return {"checks": checks, "passed": all(checks.values())}


def _run_seed(
    train_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
    vocabulary: Mapping[str, int],
    slot_classes: Mapping[str, Mapping[str, int]],
    *,
    seed: int,
    config: CausalMoEConfig,
    slot_decoder_layers: int,
    slot_decoder_heads: int,
    pretrain_epochs: int,
    posttrain_epochs: int,
    microbatch: int,
    grad_accum: int,
    lr_pretrain: float,
    lr_posttrain: float,
    kl_weight: float,
    weights: Mapping[str, float],
    device: torch.device,
    checkpoint_dir: Path | None,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    model = ComposedRuleIRModel(vocab_size=len(vocabulary), config=config, slot_classes=slot_classes, slot_decoder_layers=slot_decoder_layers, slot_decoder_heads=slot_decoder_heads).to(device)
    _train_stage_a(model, train_rows, vocabulary, slot_classes, epochs=pretrain_epochs, microbatch=microbatch, grad_accum=grad_accum, lr=lr_pretrain, device=device)
    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    baseline = _evaluate(teacher, holdout_rows, vocabulary, slot_classes, device)
    _train_stage_b(model, teacher, train_rows, vocabulary, slot_classes, epochs=posttrain_epochs, microbatch=microbatch, grad_accum=grad_accum, lr=lr_posttrain, kl_weight=kl_weight, weights=weights, device=device)
    post = _evaluate(model, holdout_rows, vocabulary, slot_classes, device)
    drop = (float(baseline["predictive_entropy"] or 0.0) - float(post["predictive_entropy"] or 0.0)) / max(abs(float(baseline["predictive_entropy"] or 0.0)), 1e-12)
    checkpoint = {"path": None, "sha256": None}
    if checkpoint_dir is not None:
        checkpoint_path = checkpoint_dir / f"pg375_seed_{int(seed)}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema_version": SCHEMA_VERSION, "seed": int(seed), "config": dict(config.__dict__), "slot_decoder_layers": slot_decoder_layers, "slot_decoder_heads": slot_decoder_heads, "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "vocabulary": dict(vocabulary), "slot_classes": {slot: dict(values) for slot, values in slot_classes.items()}}, checkpoint_path)
        checkpoint = {"path": _normalise(checkpoint_path), "sha256": _sha_file(checkpoint_path)}
    return {"seed": int(seed), "baseline": baseline, "post": post, "entropy_relative_drop": round(drop, 6), "checkpoint": checkpoint}


def run_candidate(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
    dataset: Mapping[str, Any] | None = None,
    seeds: Sequence[int] = SEEDS,
    device: str = "cpu",
    config: CausalMoEConfig | None = None,
    slot_decoder_layers: int = 2,
    slot_decoder_heads: int | None = None,
    pretrain_epochs: int = 1,
    posttrain_epochs: int = 1,
    microbatch: int = 2,
    grad_accum: int = 1,
    lr_pretrain: float = 1e-4,
    lr_posttrain: float = 2.5e-5,
    kl_weight: float = 0.25,
    weights: Mapping[str, float] | None = None,
    checkpoint_dir: Path | None = None,
    locks: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    data = dataset or {"status": "fixture", "split_contract": {"train_group_hashes": ["train"], "holdout_group_hashes": ["holdout"]}, "vocabulary": {}}
    contract = audit_pg375_contract(data, train_rows, holdout_rows)
    base_result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked_data_contract" if contract["status"] != "passed" else ("cpu_smoke_candidate_only" if device == "cpu" else "remote_candidate_only"),
        "data_contract": contract,
        "training": {"device": device, "seeds": [int(seed) for seed in seeds], "baseline_kind": "train_only_next_token_pretrain", "vocabulary_scope": "train_only", "pretrain_epochs": int(pretrain_epochs), "posttrain_epochs": int(posttrain_epochs), "microbatch": int(microbatch), "grad_accum": int(grad_accum), "kl_weight": float(kl_weight), "loss_weights": dict(weights or DEFAULT_WEIGHTS), "optimizer_started": False, "checkpoint_written": False},
        "promotion": {key: False for key in PROMOTION_KEYS},
        "scientific_gate": {"claim_allowed": False, "typed_live_replay_with_model_selected_wire": False, "independent_implementation": False, "trained_baseline_entropy_comparison": False},
    }
    if contract["status"] != "passed":
        # Critical fail-closed behavior: no device initialization, optimizer,
        # checkpoint or GPU touch occurs when the data contract is incomplete.
        base_result["execution"] = {"optimizer_started": False, "gpu_touched": False, "docker_started": False, "network_used": False, "checkpoint_written": False}
        if locks is not None:
            base_result["locks"] = dict(locks)
        return base_result
    vocabulary = build_train_vocabulary(train_rows)
    slot_classes = _slot_values_from_rows(train_rows)
    torch_device = _device_gate(device)
    effective_config = config or CausalMoEConfig(d_model=32 if device == "cpu" else 384, n_heads=2 if device == "cpu" else 4, n_layers=1 if device == "cpu" else 6, experts=2 if device == "cpu" else 4, expert_hidden=64 if device == "cpu" else 1024, max_length=768)
    required_window = max((len(row["context_tokens"]) + len(row["target_tokens"]) for row in [*train_rows, *holdout_rows]), default=1)
    if int(effective_config.max_length) < required_window:
        raise ValueError(f"PG-375 max_length={effective_config.max_length} below required context window {required_window}")
    effective_heads = int(slot_decoder_heads or effective_config.n_heads)
    results = [_run_seed(train_rows, holdout_rows, vocabulary, slot_classes, seed=int(seed), config=effective_config, slot_decoder_layers=slot_decoder_layers, slot_decoder_heads=effective_heads, pretrain_epochs=pretrain_epochs, posttrain_epochs=posttrain_epochs, microbatch=microbatch, grad_accum=grad_accum, lr_pretrain=lr_pretrain, lr_posttrain=lr_posttrain, kl_weight=kl_weight, weights=weights or DEFAULT_WEIGHTS, device=torch_device, checkpoint_dir=checkpoint_dir) for seed in seeds]
    worst = {
        "sequence_exact_min": min(float(item["post"]["sequence_exact"]) for item in results),
        "slot_composition_exact_min": min(float(item["post"]["slot_composition_exact"]) for item in results),
        "slot_accuracy_min": min(float(item["post"]["slot_accuracy"]) for item in results),
        "ask_recall_min": min(float(item["post"]["ask_recall"] or 0.0) for item in results),
        "repair_recall_min": min(float(item["post"]["repair_recall"] or 0.0) for item in results),
        "positive_recall_min": min(float(item["post"]["positive_recall"] or 0.0) for item in results),
        "negative_false_allow_max": max(int(item["post"]["negative_false_allow"]) for item in results),
        "entropy_relative_drop_max": max(float(item["entropy_relative_drop"]) for item in results),
    }
    base_result.update({"candidates": results, "worst_seed": worst, "hard_gate": _hard_gate(worst, contract), "training": {**base_result["training"], "optimizer_started": True, "vocabulary_size": len(vocabulary), "slot_class_counts": {slot: len(values) for slot, values in slot_classes.items()}, "required_context_window": required_window}, "scientific_gate": {"claim_allowed": False, "typed_live_replay_with_model_selected_wire": False, "independent_implementation": False, "trained_baseline_entropy_comparison": True}, "execution": {"optimizer_started": True, "gpu_touched": device != "cpu", "docker_started": False, "network_used": False, "checkpoint_written": bool(checkpoint_dir)}})
    return base_result


def build_plan_report(dataset: Mapping[str, Any], contract: Mapping[str, Any], locks: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "plan_only" if contract.get("status") == "passed" else "blocked_data_contract",
        "data_contract": contract,
        "training": {"device": "not_run", "baseline_kind": "train_only_next_token_pretrain", "vocabulary_scope": "train_only", "optimizer_started": False},
        "locks": dict(locks),
        "execution": {"optimizer_started": False, "gpu_touched": False, "docker_started": False, "network_used": False, "checkpoint_written": False},
        "promotion": {key: False for key in PROMOTION_KEYS},
        "scientific_gate": {"claim_allowed": False, "trained_baseline_entropy_comparison": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PG-375 composed Rule-IR candidate")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "pg375_composed_rule_ir_candidate_v1.json")
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--remote-candidate", action="store_true")
    parser.add_argument("--pretrain-epochs", type=int, default=1)
    parser.add_argument("--posttrain-epochs", type=int, default=1)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--expert-hidden", type=int, default=1024)
    parser.add_argument("--slot-decoder-layers", type=int, default=2)
    parser.add_argument("--slot-decoder-heads", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--lr-pretrain", type=float, default=1e-4)
    parser.add_argument("--lr-posttrain", type=float, default=2.5e-5)
    parser.add_argument("--kl-weight", type=float, default=0.25)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.cpu_smoke and args.remote_candidate:
        parser.error("--cpu-smoke and --remote-candidate are mutually exclusive")
    train, holdout, info, contract = load_pg364_dataset(args.dataset, args.audit)
    locks = {**info["locks"], "runner_sha256": _sha_file(Path(__file__)), "base_model_sha256": _sha_file(ROOT / "app" / "pg295_causal_moe.py")}
    if args.cpu_smoke or args.remote_candidate:
        seeds = (SEEDS[0],) if args.cpu_smoke else SEEDS
        config = CausalMoEConfig(d_model=16 if args.cpu_smoke else int(args.d_model), n_heads=2 if args.cpu_smoke else max(1, int(args.slot_decoder_heads)), n_layers=1 if args.cpu_smoke else int(args.n_layers), experts=2 if args.cpu_smoke else int(args.experts), expert_hidden=32 if args.cpu_smoke else int(args.expert_hidden), max_length=int(args.max_length))
        result = run_candidate(train_rows=train, holdout_rows=holdout, dataset=info["dataset"], seeds=seeds, device="cpu" if args.cpu_smoke else "cuda:0", config=config, slot_decoder_layers=1 if args.cpu_smoke else int(args.slot_decoder_layers), slot_decoder_heads=2 if args.cpu_smoke else int(args.slot_decoder_heads), pretrain_epochs=int(args.pretrain_epochs), posttrain_epochs=int(args.posttrain_epochs), microbatch=int(args.microbatch), grad_accum=int(args.grad_accum), lr_pretrain=float(args.lr_pretrain), lr_posttrain=float(args.lr_posttrain), kl_weight=float(args.kl_weight), checkpoint_dir=args.checkpoint_dir, locks=locks)
    else:
        result = build_plan_report(info["dataset"], contract, locks)
    result["locks"] = {**result.get("locks", {}), **locks}
    result["report_sha256"] = _sha_json(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result if args.json else {"status": result["status"], "report_sha256": result["report_sha256"]}, ensure_ascii=False, indent=2 if args.json else None))
    return 0 if result["status"] not in {"blocked_data_contract"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
