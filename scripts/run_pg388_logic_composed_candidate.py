"""PG-388 abstract Rule-IR composition candidate.

This is a bounded, CPU-first research runner for the local logic trajectory
dataset.  It learns an ordered 11-slot Rule-IR composition from abstract
context tokens only; evaluator summaries, source metadata, raw values, wire
and payload strings are never copied into a model batch.  The default command
is plan-only.  ``--cpu-smoke`` is an explicitly candidate-only wiring run and
never grants training, memory, payload-catalog or vulnerability promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime
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

from app.pg293_failure_next_action import PAD, UNK
from app.pg295_causal_moe import CausalMoEConfig, CausalMoELanguageModel


SCHEMA_VERSION = "pg388-logic-composed-candidate-v1"
DEFAULT_DATASET = ROOT / "research" / "pg388_logic_rule_ir_composition_dataset_v1.json"
DEFAULT_AUDIT = ROOT / "research" / "pg388_logic_rule_ir_composition_audit_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg388_logic_composed_candidate_cpu_smoke_v1.json"
SEEDS = (38841, 38842, 38843)
SLOT_ORDER = (
    "question",
    "ask_reason",
    "logic_invariant_ref",
    "state_transition_ref",
    "precondition_ref",
    "counterfactual_ref",
    "probe_variant_ref",
    "next_action",
    "repair_action",
    "oracle_ref",
    "safe_to_send",
)
RAW_MARKERS = (
    "raw_payload=",
    "payload=",
    "response_body=",
    "response_body_text=",
    "raw_response=",
    "wire=",
    "evaluator=",
    "oracle_answer=",
    "route_literal=",
    "http://",
    "https://",
)
PROMOTION_KEYS = (
    "training_allowed",
    "memory_promotion_allowed",
    "payload_catalog_promotion_allowed",
    "vulnerability_claim_allowed",
)


def _local_cuda_gate(*, now: datetime | None = None, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Return a fail-closed gate for the explicitly authorized local weekday lane.

    This lane is a candidate diagnostic only.  The formal source-row/capability
    contract remains independent and is never bypassed by a CUDA run.
    """
    env = dict(environ or {}) if environ is not None else dict(os.environ)
    current = now or datetime.now().astimezone()
    explicit = env.get("BLACKBOX_LOCAL_MORNING_TRAIN") == "1"
    weekday = current.weekday() < 5
    in_window = 8 <= current.hour < 18
    visible = env.get("CUDA_VISIBLE_DEVICES", "")
    failures: list[str] = []
    if not explicit:
        failures.append("missing_BLACKBOX_LOCAL_MORNING_TRAIN")
    if not weekday:
        failures.append("outside_weekday_lane")
    if not in_window:
        failures.append("outside_local_morning_window")
    if visible != "0":
        failures.append("CUDA_VISIBLE_DEVICES_must_equal_0")
    return {
        "status": "passed" if not failures else "blocked",
        "explicit_flag": explicit,
        "weekday": weekday,
        "in_window": in_window,
        "timezone": str(current.tzinfo or "local"),
        "cuda_visible_devices": visible,
        "failures": failures,
    }


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_promotion() -> dict[str, bool]:
    return {key: False for key in PROMOTION_KEYS}


def _parse_slots(target_tokens: Sequence[Any]) -> dict[str, str]:
    values: dict[str, str] = {}
    for token in target_tokens:
        text = str(token)
        if "=" not in text:
            raise ValueError("target_slot_token_malformed")
        key, value = text.split("=", 1)
        if key not in SLOT_ORDER or key in values or not value:
            raise ValueError("target_slot_contract_mismatch")
        values[key] = value
    if tuple(values) != SLOT_ORDER and set(values) != set(SLOT_ORDER):
        raise ValueError("target_slot_order_mismatch")
    return {slot: values[slot] for slot in SLOT_ORDER}


def _safe_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    context = raw.get("context_tokens")
    target = raw.get("target_tokens")
    if not isinstance(context, list) or not isinstance(target, list):
        raise ValueError("model_visible_tokens_must_be_lists")
    tokens = [str(token) for token in [*context, *target]]
    if any(marker in token.casefold() for token in tokens for marker in RAW_MARKERS):
        raise ValueError("raw_or_evaluator_marker_in_model_tokens")
    for key in ("raw_source_stored", "raw_payload_stored", "raw_response_body_stored", "oracle_answer_in_context"):
        if raw.get(key) is not False:
            raise ValueError("context_firewall_not_closed")
    return {
        "context_tokens": [str(token) for token in context],
        "target_values": _parse_slots(target),
        "split": str(raw.get("split", "")),
    }


def load_rows(dataset_path: Path, audit_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict) or not isinstance(audit, dict):
        raise ValueError("composition_inputs_must_be_objects")
    declared = tuple(str(item) for item in dataset.get("slot_order", []))
    if declared != SLOT_ORDER:
        raise ValueError("pg388_slot_order_mismatch")
    raw_rows = dataset.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError("pg388_rows_missing")
    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("pg388_row_not_object")
        row = _safe_row(raw)
        if row["split"] == "train":
            train.append(row)
        elif row["split"] == "implementation_holdout":
            holdout.append(row)
        else:
            raise ValueError("pg388_unknown_split")
    return train, holdout, {
        "dataset_status": str(dataset.get("status", "unknown")),
        "audit_status": str(audit.get("status", "unknown")),
        "dataset_sha256": _sha_file(dataset_path),
        "audit_sha256": _sha_file(audit_path),
        "source_contract": dataset.get("source_contract") if isinstance(dataset.get("source_contract"), dict) else {},
    }


def build_context_vocabulary(train: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    tokens = {PAD, UNK}
    for row in train:
        tokens.update(str(token) for token in row["context_tokens"])
    return {token: index for index, token in enumerate([PAD, UNK, *sorted(tokens - {PAD, UNK})])}


def build_slot_classes(train: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    classes: dict[str, set[str]] = {slot: set() for slot in SLOT_ORDER}
    for row in train:
        values = row["target_values"]
        for slot in SLOT_ORDER:
            classes[slot].add(str(values[slot]))
    if any(not values for values in classes.values()):
        raise ValueError("train_slot_coverage_missing")
    return {slot: {value: index for index, value in enumerate(sorted(values))} for slot, values in classes.items()}


def contract_audit(train: Sequence[Mapping[str, Any]], holdout: Sequence[Mapping[str, Any]], info: Mapping[str, Any], vocabulary: Mapping[str, int], slot_classes: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    unknown_context = sorted({str(token) for row in holdout for token in row["context_tokens"] if str(token) not in vocabulary})
    unknown_slots = {
        slot: sorted({str(row["target_values"][slot]) for row in holdout if str(row["target_values"][slot]) not in slot_classes[slot]})
        for slot in SLOT_ORDER
    }
    unknown_slots = {slot: values for slot, values in unknown_slots.items() if values}
    train_context = {tuple(row["context_tokens"]) for row in train}
    holdout_context = {tuple(row["context_tokens"]) for row in holdout}
    failures: list[str] = []
    source = info.get("source_contract") if isinstance(info.get("source_contract"), Mapping) else {}
    if info.get("dataset_status") != "abstract_rule_ir_composition_candidate_only":
        failures.append("dataset_status_not_candidate")
    if info.get("audit_status") != "passed_candidate_rule_ir_audit":
        failures.append("audit_not_passed")
    if source.get("row_bound_typed_evidence") is not True:
        failures.append("row_bound_typed_evidence_missing")
    if source.get("fresh_role_reset_attested") is not True:
        failures.append("fresh_role_reset_attestation_missing")
    if source.get("operator_reviewed") is not True:
        failures.append("operator_review_missing")
    if not train:
        failures.append("train_split_empty")
    if not holdout:
        failures.append("holdout_split_empty")
    if unknown_context or unknown_slots:
        failures.append("train_only_vocabulary_gap")
    if train_context & holdout_context:
        failures.append("cross_split_context_overlap")
    return {
        "status": "passed_candidate_input_contract" if not failures else "blocked_candidate_contract",
        "failures": failures,
        "train_count": len(train),
        "holdout_count": len(holdout),
        "unknown_context_count": len(unknown_context),
        "unknown_slot_value_count": sum(len(values) for values in unknown_slots.values()),
        "unknown_slot_values": {slot: len(values) for slot, values in sorted(unknown_slots.items())},
        "context_overlap": len(train_context & holdout_context),
        "source_contract_complete": all(source.get(key) is True for key in ("row_bound_typed_evidence", "fresh_role_reset_attested", "operator_reviewed")),
    }


def _pad_context(rows: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = [[int(vocabulary.get(str(token), vocabulary[UNK])) for token in row["context_tokens"]] for row in rows]
    width = max((len(sequence) for sequence in sequences), default=1)
    ids = torch.full((len(sequences), width), int(vocabulary[PAD]), dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), width), dtype=torch.bool, device=device)
    for index, sequence in enumerate(sequences):
        ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[index, : len(sequence)] = True
    return ids, mask


def _labels(rows: Sequence[Mapping[str, Any]], classes: Mapping[str, Mapping[str, int]], device: torch.device) -> torch.Tensor:
    return torch.tensor([[classes[slot][str(row["target_values"][slot])] for slot in SLOT_ORDER] for row in rows], dtype=torch.long, device=device)


class LogicCompositionModel(nn.Module):
    def __init__(self, *, vocabulary_size: int, config: CausalMoEConfig, slot_classes: Mapping[str, Mapping[str, int]], decoder_layers: int = 1) -> None:
        super().__init__()
        self.config = config
        self.slot_classes = {slot: dict(values) for slot, values in slot_classes.items()}
        self.backbone = CausalMoELanguageModel(vocab_size=vocabulary_size, config=config)
        self.slot_queries = nn.Parameter(torch.zeros(len(SLOT_ORDER), config.d_model))
        nn.init.normal_(self.slot_queries, mean=0.0, std=float(config.initializer_range))
        max_classes = max((len(values) for values in slot_classes.values()), default=1)
        self.previous_value = nn.Embedding(max_classes + 1, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=max(config.d_model * 2, 64),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=max(1, int(decoder_layers)))
        self.slot_heads = nn.ModuleDict({slot: nn.Linear(config.d_model, len(values)) for slot, values in slot_classes.items()})
        self.ask_head = nn.Linear(config.d_model, 2)
        self.repair_head = nn.Linear(config.d_model, 4)
        self.safe_head = nn.Linear(config.d_model, 2)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)

    def _composition(self, boundary: torch.Tensor, labels: torch.Tensor | None) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        previous = torch.zeros((boundary.shape[0],), dtype=torch.long, device=boundary.device)
        inputs: list[torch.Tensor] = []
        for index, slot in enumerate(SLOT_ORDER):
            inputs.append(boundary + self.slot_queries[index].unsqueeze(0) + self.previous_value(previous))
            if labels is not None:
                previous = labels[:, index] + 1
        sequence = torch.stack(inputs, dim=1)
        hidden = self.decoder(sequence, mask=self._causal_mask(len(SLOT_ORDER), sequence.device))
        return {slot: self.slot_heads[slot](hidden[:, index]) for index, slot in enumerate(SLOT_ORDER)}, hidden

    def _greedy(self, boundary: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        previous = torch.zeros((boundary.shape[0],), dtype=torch.long, device=boundary.device)
        inputs: list[torch.Tensor] = []
        hidden_steps: list[torch.Tensor] = []
        logits: dict[str, torch.Tensor] = {}
        for index, slot in enumerate(SLOT_ORDER):
            inputs.append(boundary + self.slot_queries[index].unsqueeze(0) + self.previous_value(previous))
            hidden = self.decoder(torch.stack(inputs, dim=1), mask=self._causal_mask(len(inputs), boundary.device))
            current = hidden[:, -1]
            current_logits = self.slot_heads[slot](current)
            logits[slot] = current_logits
            hidden_steps.append(current)
            previous = current_logits.argmax(-1).detach() + 1
        return logits, torch.stack(hidden_steps, dim=1)

    def forward(self, context_ids: torch.Tensor, context_mask: torch.Tensor, *, labels: torch.Tensor | None = None, greedy: bool = False) -> dict[str, Any]:
        hidden, balance = self.backbone.forward_hidden(context_ids, valid_mask=context_mask)
        lengths = context_mask.long().sum(dim=1).clamp_min(1) - 1
        boundary = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
        if labels is not None and not greedy:
            composition, slot_hidden = self._composition(boundary, labels)
        else:
            composition, slot_hidden = self._greedy(boundary)
        summary = slot_hidden.mean(dim=1)
        return {
            "composition": composition,
            "ask": self.ask_head(summary),
            "repair": self.repair_head(summary),
            "safe": self.safe_head(summary),
            "balance": balance,
        }


def _loss(model: LogicCompositionModel, rows: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], classes: Mapping[str, Mapping[str, int]], device: torch.device) -> torch.Tensor:
    ids, mask = _pad_context(rows, vocabulary, device)
    labels = _labels(rows, classes, device)
    output = model(ids, mask, labels=labels)
    composition = torch.stack([F.cross_entropy(output["composition"][slot], labels[:, index]) for index, slot in enumerate(SLOT_ORDER)]).mean()
    ask = torch.tensor([int(str(row["target_values"]["question"]).startswith("ask")) for row in rows], dtype=torch.long, device=device)
    repair = torch.tensor([{"select_probe_variant": 0, "replay": 1, "repair": 2, "abstain": 3}.get(str(row["target_values"]["next_action"]), 0) for row in rows], dtype=torch.long, device=device)
    safe = torch.tensor([int(str(row["target_values"]["safe_to_send"]) == "true") for row in rows], dtype=torch.long, device=device)
    return composition + F.cross_entropy(output["ask"], ask) + F.cross_entropy(output["repair"], repair) + F.cross_entropy(output["safe"], safe) + 0.01 * output["balance"]


def _evaluate(model: LogicCompositionModel, rows: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], classes: Mapping[str, Mapping[str, int]], device: torch.device) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "composition_exact": 0.0, "slot_accuracy": 0.0, "per_slot": {}, "ask_recall": 0.0, "repair_recall": 0.0, "positive_recall": 0.0, "negative_false_allow": 0, "negative_count": 0, "composition_entropy": 0.0}
    reverse = {slot: {index: value for value, index in values.items()} for slot, values in classes.items()}
    per_slot = {slot: [0, 0] for slot in SLOT_ORDER}
    composition_exact = 0
    ask_total = ask_correct = repair_total = repair_correct = positive_total = positive_correct = negative_count = false_allow = 0
    entropies: list[float] = []
    model.eval()
    with torch.inference_mode():
        for row in rows:
            ids, mask = _pad_context([row], vocabulary, device)
            output = model(ids, mask, greedy=True)
            all_correct = True
            for slot in SLOT_ORDER:
                prediction = reverse[slot][int(output["composition"][slot].argmax(-1).item())]
                correct = int(prediction == str(row["target_values"][slot]))
                per_slot[slot][0] += correct
                per_slot[slot][1] += 1
                all_correct = all_correct and bool(correct)
                entropies.append(float((-(output["composition"][slot].softmax(-1) * output["composition"][slot].softmax(-1).clamp_min(1e-12).log()).sum()).cpu()))
            composition_exact += int(all_correct)
            expected_ask = str(row["target_values"]["question"]).startswith("ask")
            predicted_ask = int(output["ask"].argmax(-1).item()) == 1
            ask_total += int(expected_ask)
            ask_correct += int(expected_ask and predicted_ask)
            expected_repair = str(row["target_values"]["next_action"]) == "repair"
            predicted_repair = int(output["repair"].argmax(-1).item()) == 2
            repair_total += int(expected_repair)
            repair_correct += int(expected_repair and predicted_repair)
            expected_safe = str(row["target_values"]["safe_to_send"]) == "true"
            predicted_safe = int(output["safe"].argmax(-1).item()) == 1
            positive_total += int(expected_safe)
            positive_correct += int(expected_safe and predicted_safe)
            negative_count += int(not expected_safe)
            false_allow += int(not expected_safe and predicted_safe)
    total_slots = sum(item[1] for item in per_slot.values())
    return {
        "count": len(rows),
        "composition_exact": round(composition_exact / len(rows), 6),
        "slot_accuracy": round(sum(item[0] for item in per_slot.values()) / max(total_slots, 1), 6),
        "per_slot": {slot: {"accuracy": round(values[0] / max(values[1], 1), 6), "count": values[1]} for slot, values in per_slot.items()},
        "ask_recall": round(ask_correct / max(ask_total, 1), 6) if ask_total else None,
        "repair_recall": round(repair_correct / max(repair_total, 1), 6) if repair_total else None,
        "positive_recall": round(positive_correct / max(positive_total, 1), 6) if positive_total else None,
        "negative_false_allow": false_allow,
        "negative_count": negative_count,
        "composition_entropy": round(sum(entropies) / max(len(entropies), 1), 6),
    }


def _train_seed(train: Sequence[Mapping[str, Any]], holdout: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], classes: Mapping[str, Mapping[str, int]], *, seed: int, config: CausalMoEConfig, epochs: int, microbatch: int, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = LogicCompositionModel(vocabulary_size=len(vocabulary), config=config, slot_classes=classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    order = list(range(len(train)))
    model.train()
    for epoch in range(max(1, epochs)):
        random.Random(seed + epoch).shuffle(order)
        for start in range(0, len(order), max(1, microbatch)):
            batch = [train[index] for index in order[start : start + max(1, microbatch)]]
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, batch, vocabulary, classes, device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return {"seed": seed, "train": _evaluate(model, train, vocabulary, classes, device), "holdout": _evaluate(model, holdout, vocabulary, classes, device)}


def run_candidate(*, dataset_path: Path = DEFAULT_DATASET, audit_path: Path = DEFAULT_AUDIT, cpu_smoke: bool = False, local_cuda: bool = False, row_limit: int | None = 128, epochs: int = 1, d_model: int = 64, layers: int = 2, experts: int = 2, expert_hidden: int = 128, microbatch: int = 16) -> dict[str, Any]:
    if cpu_smoke and local_cuda:
        raise ValueError("cpu_smoke_and_local_cuda_are_mutually_exclusive")
    train, holdout, info = load_rows(dataset_path, audit_path)
    vocabulary = build_context_vocabulary(train)
    classes = build_slot_classes(train)
    contract = contract_audit(train, holdout, info, vocabulary, classes)
    if cpu_smoke and row_limit is not None:
        limit = max(1, int(row_limit))
        train = train[:limit]
        holdout = holdout[:limit]
    config = CausalMoEConfig(d_model=int(d_model), n_heads=4 if int(d_model) % 4 == 0 else 2, n_layers=int(layers), experts=int(experts), expert_hidden=int(expert_hidden), max_length=128)
    local_gate = _local_cuda_gate() if local_cuda else {"status": "not_requested", "failures": []}
    cuda_available = bool(torch.cuda.is_available()) if local_cuda and local_gate["status"] == "passed" else False
    if local_cuda and local_gate["status"] == "passed" and not cuda_available:
        local_gate = {**local_gate, "status": "blocked", "failures": ["cuda_unavailable"]}
    if local_cuda and local_gate["status"] == "passed" and torch.cuda.device_count() != 1:
        local_gate = {**local_gate, "status": "blocked", "failures": ["visible_device_count_must_equal_1"]}
    device = torch.device("cuda:0" if local_cuda and local_gate["status"] == "passed" else "cpu")
    local_device_name = torch.cuda.get_device_name(0) if local_cuda and local_gate["status"] == "passed" else None
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": ("local_cuda_composed_candidate_only" if local_cuda and local_gate["status"] == "passed" else "blocked_local_cuda_gate" if local_cuda else "cpu_composed_candidate_only" if cpu_smoke else "plan_only_blocked"),
        "dataset_file": dataset_path.name,
        "audit_file": audit_path.name,
        "dataset_sha256": info["dataset_sha256"],
        "audit_sha256": info["audit_sha256"],
        "slot_order": list(SLOT_ORDER),
        "contract": contract,
        "train_count": len(train),
        "holdout_count": len(holdout),
        "train_context_vocabulary": {"size": len(vocabulary), "scope": "train_context_only"},
        "model": {"backbone": "decoder_only_causal_moe", "composition_decoder": "causal_previous_slot_conditioned", "d_model": int(d_model), "layers": int(layers), "experts": int(experts)},
        "execution": {"optimizer_started": False, "device": "cuda:0" if local_cuda and local_gate["status"] == "passed" else "cpu", "gpu_touched": False, "local_cuda_candidate": bool(local_cuda), "docker_started": False, "network_contacted": False, "wire_created": False},
        "local_training_gate": local_gate,
        "local_device_name": local_device_name,
        "training_eligible": 0,
        "capability_training_allowed": False,
        "logic_composition_candidate_only": True,
        "promotion": _safe_promotion(),
    }
    if cpu_smoke or (local_cuda and local_gate["status"] == "passed"):
        report["execution"]["optimizer_started"] = True
        report["execution"]["gpu_touched"] = bool(local_cuda)
        report["seeds"] = [_train_seed(train, holdout, vocabulary, classes, seed=seed, config=config, epochs=epochs, microbatch=microbatch, device=device) for seed in SEEDS]
    else:
        report["seeds"] = []
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--local-cuda", action="store_true", help="run an explicitly gated local CUDA candidate; never grants promotion")
    parser.add_argument("--row-limit", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--experts", type=int, default=2)
    parser.add_argument("--expert-hidden", type=int, default=128)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run_candidate(dataset_path=args.dataset, audit_path=args.audit, cpu_smoke=args.cpu_smoke, local_cuda=args.local_cuda, row_limit=args.row_limit, epochs=args.epochs, d_model=args.d_model, layers=args.layers, experts=args.experts, expert_hidden=args.expert_hidden, microbatch=args.microbatch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": report["status"], "train_count": report["train_count"], "holdout_count": report["holdout_count"], "optimizer_started": report["execution"]["optimizer_started"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
