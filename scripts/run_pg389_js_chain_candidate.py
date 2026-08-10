"""PG-389 abstract JavaScript decode/filter chain candidate.

The runner learns ordered abstract chain decisions (action, repair, fixture
shape, safety and observation goal).  It never materializes a concrete probe,
source text, response body or evaluator answer.  CPU smoke is the default
research path; the explicitly gated local CUDA path remains candidate-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
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
from scripts.run_pg388_logic_composed_candidate import _local_cuda_gate


SCHEMA_VERSION = "pg389-js-chain-candidate-v1"
DEFAULT_DATASET = ROOT / "research" / "pg389_js_decode_filter_chain_dataset_v1.json"
DEFAULT_AUDIT = ROOT / "research" / "pg389_js_decode_filter_chain_audit_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg389_js_chain_candidate_cpu_smoke_v1.json"
SEEDS = (38941, 38942, 38943)
SLOT_ORDER = (
    "next_action",
    "repair_action",
    "probe_variant_ref",
    "safe_to_send",
    "ask_reason",
    "observation_goal",
)
RAW_MARKERS = (
    "raw_payload=",
    "payload=",
    "response_body=",
    "response_body_text=",
    "raw_response=",
    "raw_value=",
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


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_promotion() -> dict[str, bool]:
    return {key: False for key in PROMOTION_KEYS}


def _parse_target(target_tokens: Sequence[Any]) -> dict[str, str]:
    values: dict[str, str] = {}
    for token in target_tokens:
        text = str(token)
        if text in {"[TARGET_BEGIN]", "[TARGET_END]"}:
            continue
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
    visible = [str(token) for token in [*context, *target]]
    if any(marker in token.casefold() for token in visible for marker in RAW_MARKERS):
        raise ValueError("raw_or_evaluator_marker_in_model_tokens")
    for key in ("source_text_stored", "raw_value_stored", "raw_wire_stored", "oracle_answer_in_context"):
        if raw.get(key, False) is not False:
            raise ValueError("context_firewall_not_closed")
    return {
        "context_tokens": [str(token) for token in context],
        "target_values": _parse_target(target),
        "split": str(raw.get("split", "")),
    }


def load_rows(dataset_path: Path = DEFAULT_DATASET, audit_path: Path = DEFAULT_AUDIT) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict) or not isinstance(audit, dict):
        raise ValueError("pg389_inputs_must_be_objects")
    rows = dataset.get("rows")
    if not isinstance(rows, list):
        raise ValueError("pg389_rows_missing")
    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("pg389_row_not_object")
        row = _safe_row(raw)
        if row["split"] == "train":
            train.append(row)
        elif row["split"] == "implementation_holdout":
            holdout.append(row)
        else:
            raise ValueError("pg389_unknown_split")
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
        for slot in SLOT_ORDER:
            classes[slot].add(str(row["target_values"][slot]))
    if any(not values for values in classes.values()):
        raise ValueError("pg389_train_slot_coverage_missing")
    return {slot: {value: index for index, value in enumerate(sorted(values))} for slot, values in classes.items()}


def contract_audit(train: Sequence[Mapping[str, Any]], holdout: Sequence[Mapping[str, Any]], info: Mapping[str, Any], vocabulary: Mapping[str, int], classes: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    unknown_context = sorted({str(token) for row in holdout for token in row["context_tokens"] if str(token) not in vocabulary})
    unknown_slots = {
        slot: sorted({str(row["target_values"][slot]) for row in holdout if str(row["target_values"][slot]) not in classes[slot]})
        for slot in SLOT_ORDER
    }
    unknown_slots = {slot: values for slot, values in unknown_slots.items() if values}
    train_context = {tuple(row["context_tokens"]) for row in train}
    holdout_context = {tuple(row["context_tokens"]) for row in holdout}
    source = info.get("source_contract") if isinstance(info.get("source_contract"), Mapping) else {}
    failures: list[str] = []
    if info.get("dataset_status") != "abstract_js_chain_candidate_only":
        failures.append("dataset_status_not_candidate")
    if info.get("audit_status") != "passed_candidate_audit":
        failures.append("audit_not_passed")
    for key in ("fresh_role_reset", "candidate_reference_negative_replay", "typed_evidence", "operator_reviewed"):
        if source.get(key) is not True:
            failures.append(f"source_contract_{key}_missing")
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
        "source_contract_complete": not any(source.get(key) is not True for key in ("fresh_role_reset", "candidate_reference_negative_replay", "typed_evidence", "operator_reviewed")),
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


class JSChainCompositionModel(nn.Module):
    def __init__(self, *, vocabulary_size: int, config: CausalMoEConfig, slot_classes: Mapping[str, Mapping[str, int]]) -> None:
        super().__init__()
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
        self.decoder = nn.TransformerEncoder(layer, num_layers=1)
        self.slot_heads = nn.ModuleDict({slot: nn.Linear(config.d_model, len(values)) for slot, values in slot_classes.items()})

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)

    def forward(self, context_ids: torch.Tensor, context_mask: torch.Tensor, *, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        hidden, balance = self.backbone.forward_hidden(context_ids, valid_mask=context_mask)
        lengths = context_mask.long().sum(dim=1).clamp_min(1) - 1
        boundary = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
        previous = torch.zeros((boundary.shape[0],), dtype=torch.long, device=boundary.device)
        inputs: list[torch.Tensor] = []
        outputs: dict[str, torch.Tensor] = {}
        for index, slot in enumerate(SLOT_ORDER):
            inputs.append(boundary + self.slot_queries[index].unsqueeze(0) + self.previous_value(previous))
            hidden_steps = self.decoder(torch.stack(inputs, dim=1), mask=self._causal_mask(len(inputs), boundary.device))
            logits = self.slot_heads[slot](hidden_steps[:, -1])
            outputs[slot] = logits
            if labels is not None:
                previous = labels[:, index] + 1
            else:
                previous = logits.argmax(-1).detach() + 1
        outputs["balance"] = balance
        return outputs


def _loss(model: JSChainCompositionModel, rows: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], classes: Mapping[str, Mapping[str, int]], device: torch.device) -> torch.Tensor:
    ids, mask = _pad_context(rows, vocabulary, device)
    labels = _labels(rows, classes, device)
    output = model(ids, mask, labels=labels)
    slot_loss = torch.stack([F.cross_entropy(output[slot], labels[:, index]) for index, slot in enumerate(SLOT_ORDER)]).mean()
    return slot_loss + 0.01 * output["balance"]


def _evaluate(model: JSChainCompositionModel, rows: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], classes: Mapping[str, Mapping[str, int]], device: torch.device) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "composition_exact": 0.0, "slot_accuracy": 0.0, "per_slot": {}, "ask_recall": 0.0, "repair_recall": 0.0, "negative_false_allow": 0, "negative_count": 0, "composition_entropy": 0.0}
    reverse = {slot: {index: value for value, index in values.items()} for slot, values in classes.items()}
    per_slot = {slot: [0, 0] for slot in SLOT_ORDER}
    exact = 0
    ask_total = ask_correct = repair_total = repair_correct = negative_count = false_allow = 0
    entropies: list[float] = []
    model.eval()
    with torch.inference_mode():
        for row in rows:
            ids, mask = _pad_context([row], vocabulary, device)
            output = model(ids, mask)
            all_correct = True
            predictions: dict[str, str] = {}
            for slot in SLOT_ORDER:
                logits = output[slot]
                prediction = reverse[slot][int(logits.argmax(-1).item())]
                predictions[slot] = prediction
                correct = int(prediction == str(row["target_values"][slot]))
                per_slot[slot][0] += correct
                per_slot[slot][1] += 1
                all_correct = all_correct and bool(correct)
                probabilities = logits.softmax(-1)
                entropies.append(float((-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).cpu()))
            exact += int(all_correct)
            expected_ask = str(row["target_values"]["ask_reason"]) != "none"
            predicted_ask = predictions["ask_reason"] != "none"
            ask_total += int(expected_ask)
            ask_correct += int(expected_ask and predicted_ask)
            expected_repair = str(row["target_values"]["next_action"]) == "repair"
            predicted_repair = predictions["next_action"] == "repair"
            repair_total += int(expected_repair)
            repair_correct += int(expected_repair and predicted_repair)
            expected_safe = str(row["target_values"]["safe_to_send"]) in {"1", "true"}
            predicted_safe = predictions["safe_to_send"] in {"1", "true"}
            negative_count += int(not expected_safe)
            false_allow += int(not expected_safe and predicted_safe)
    total_slots = sum(value[1] for value in per_slot.values())
    return {
        "count": len(rows),
        "composition_exact": round(exact / len(rows), 6),
        "slot_accuracy": round(sum(value[0] for value in per_slot.values()) / max(total_slots, 1), 6),
        "per_slot": {slot: {"accuracy": round(value[0] / max(value[1], 1), 6), "count": value[1]} for slot, value in per_slot.items()},
        "ask_recall": round(ask_correct / max(ask_total, 1), 6) if ask_total else None,
        "repair_recall": round(repair_correct / max(repair_total, 1), 6) if repair_total else None,
        "negative_false_allow": false_allow,
        "negative_count": negative_count,
        "composition_entropy": round(sum(entropies) / max(len(entropies), 1), 6),
    }


def _train_seed(train: Sequence[Mapping[str, Any]], holdout: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], classes: Mapping[str, Mapping[str, int]], *, seed: int, config: CausalMoEConfig, epochs: int, microbatch: int, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = JSChainCompositionModel(vocabulary_size=len(vocabulary), config=config, slot_classes=classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    order = list(range(len(train)))
    for epoch in range(max(1, epochs)):
        model.train()
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
    if (cpu_smoke or local_cuda) and row_limit is not None:
        limit = max(1, int(row_limit))
        train = train[:limit]
        holdout = holdout[:limit]
    local_gate = _local_cuda_gate() if local_cuda else {"status": "not_requested", "failures": []}
    if local_cuda and local_gate["status"] == "passed" and not torch.cuda.is_available():
        local_gate = {**local_gate, "status": "blocked", "failures": ["cuda_unavailable"]}
    if local_cuda and local_gate["status"] == "passed" and torch.cuda.device_count() != 1:
        local_gate = {**local_gate, "status": "blocked", "failures": ["visible_device_count_must_equal_1"]}
    active_cuda = local_cuda and local_gate["status"] == "passed"
    device = torch.device("cuda:0" if active_cuda else "cpu")
    config = CausalMoEConfig(d_model=int(d_model), n_heads=4 if int(d_model) % 4 == 0 else 2, n_layers=int(layers), experts=int(experts), expert_hidden=int(expert_hidden), max_length=128)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "local_cuda_js_chain_candidate_only" if active_cuda else "blocked_local_cuda_gate" if local_cuda else "cpu_js_chain_candidate_only" if cpu_smoke else "plan_only_blocked",
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
        "execution": {"optimizer_started": False, "device": "cuda:0" if active_cuda else "cpu", "gpu_touched": bool(active_cuda), "local_cuda_candidate": bool(local_cuda), "docker_started": False, "network_contacted": False, "wire_created": False},
        "local_training_gate": local_gate,
        "local_device_name": torch.cuda.get_device_name(0) if active_cuda else None,
        "training_eligible": 0,
        "capability_training_allowed": False,
        "js_chain_reasoning_candidate_only": True,
        "promotion": _safe_promotion(),
    }
    if cpu_smoke or active_cuda:
        report["execution"]["optimizer_started"] = True
        report["seeds"] = [_train_seed(train, holdout, vocabulary, classes, seed=seed, config=config, epochs=epochs, microbatch=microbatch, device=device) for seed in SEEDS]
    else:
        report["seeds"] = []
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--local-cuda", action="store_true")
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
