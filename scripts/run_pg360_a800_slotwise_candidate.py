"""A800 candidate for slot-wise causal Rule-IR decoding (PG-360).

The model remains a decoder-only next-token Transformer-MoE.  Instead of
free-running a 12-field target, each query asks for one abstract slot and the
model predicts exactly one value token.  The evaluator later assembles those
values into Rule-IR; no raw payload, response, oracle answer, or route literal
is read by this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg293_failure_next_action import PAD, TARGET_BOS, TARGET_EOS, UNK  # noqa: E402
from app.pg295_causal_moe import CausalMoEConfig, CausalMoELanguageModel, train_causal_moe  # noqa: E402
from app.pg349_constrained_rule_ir_decoder import constrain_rule_ir  # noqa: E402
from scripts.build_pg360_slotwise_dataset import SLOTS  # noqa: E402


SCHEMA_VERSION = "pg360-a800-slotwise-candidate-v2"
TZ = ZoneInfo("Asia/Shanghai")
SEEDS = (36001, 36002, 36003)
RAW_FRAGMENTS = ("raw_payload=", "payload=", "response_body=", "raw_response=", "wire=", "evaluator=", "oracle=", "route_literal=", "family=", "implementation=", "image=")
SLOT_QUERY_BOS = "[SLOT_QUERY_BOS]"
SLOT_QUERY_EOS = "[SLOT_QUERY_EOS]"


def _slot_order(dataset: Mapping[str, Any]) -> tuple[str, ...]:
    declared = dataset.get("slot_order")
    if isinstance(declared, (list, tuple)) and declared:
        return tuple(str(slot) for slot in declared)
    return tuple(SLOTS)


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _weekend(now: datetime) -> bool:
    local = now.astimezone(TZ) if now.tzinfo else now.replace(tzinfo=TZ)
    return local.weekday() >= 5


def _rows(dataset: Mapping[str, Any], split: str, slot_order: Sequence[str] | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    slots = tuple(slot_order or _slot_order(dataset))
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for index, raw in enumerate(dataset.get("records") or []):
        if not isinstance(raw, Mapping) or str(raw.get("split")) != split:
            continue
        context = raw.get("context_tokens")
        target = raw.get("target_tokens")
        slot = str(raw.get("slot", ""))
        if slot not in slots or not isinstance(context, list) or len(context) < 3 or not isinstance(target, list) or len(target) != 3 or target[0] != "[TARGET_BOS]" or target[2] != "[TARGET_EOS]" or not str(target[1]).startswith(slot + "="):
            failures.append(f"row_{index}:stream")
            continue
        if raw.get("context_firewall") != {"forbidden_token_count": 0, "sidecars_off_context": True}:
            failures.append(f"row_{index}:firewall")
            continue
        if any(raw.get(flag) is not False for flag in ("raw_payload_stored", "raw_response_body_stored", "oracle_answer_in_context")):
            failures.append(f"row_{index}:raw_flag")
            continue
        tokens = [str(token) for token in [*context, *target]]
        if any(any(fragment in token.casefold() for fragment in RAW_FRAGMENTS) for token in tokens):
            failures.append(f"row_{index}:raw_token")
            continue
        rows.append({"context_tokens": [str(token) for token in context], "target_tokens": [str(token) for token in target], "slot": slot, "source_record_digest": str(raw.get("source_record_digest", "")), "split": split})
    return rows, sorted(set(failures))


def _vocabulary(dataset: Mapping[str, Any]) -> dict[str, int]:
    vocab = dataset.get("vocabulary") if isinstance(dataset.get("vocabulary"), Mapping) else {}
    ordered = [PAD, UNK, *(str(token) for token in vocab.get("context_tokens") or []), *(str(token) for token in vocab.get("target_tokens") or [])]
    return {str(token): index for index, token in enumerate(dict.fromkeys(ordered))}


def _slot_value(target_tokens: Sequence[str], slot: str) -> str:
    prefix = slot + "="
    return next((str(token) for token in target_tokens if str(token).startswith(prefix)), f"{slot}=unknown")


def _balanced_slot_rows(rows: Sequence[Mapping[str, Any]], seed: int) -> tuple[list[Mapping[str, Any]], dict[str, dict[str, int]]]:
    """Balance values within each slot without changing source/holdout rows."""

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        slot = str(row["slot"])
        groups[(slot, _slot_value(row["target_tokens"], slot))].append(row)
    by_slot: dict[str, list[tuple[str, list[Mapping[str, Any]]]]] = defaultdict(list)
    for (slot, value), group in groups.items():
        by_slot[slot].append((value, group))
    rng = random.Random(int(seed))
    balanced: list[Mapping[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for slot in sorted(by_slot):
        target = max(len(group) for _, group in by_slot[slot])
        counts[slot] = {}
        for value, group in sorted(by_slot[slot]):
            repeated = [group[index % len(group)] for index in range(target)]
            rng.shuffle(repeated)
            balanced.extend(repeated)
            counts[slot][value] = len(repeated)
    rng.shuffle(balanced)
    return balanced, counts


def _sqrt_balanced_slot_rows(rows: Sequence[Mapping[str, Any]], seed: int) -> tuple[list[Mapping[str, Any]], dict[str, dict[str, int]]]:
    """Partially rebalance rare values without flattening the empirical prior.

    A full equalization made the previous candidate lose predictive entropy.
    The geometric target ``sqrt(count * max_count)`` increases exposure to
    rare repair/abstain/positive values while retaining prevalence information.
    It only duplicates existing abstract rows; it never reads evaluator data or
    changes split membership.
    """

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        slot = str(row["slot"])
        groups[(slot, _slot_value(row["target_tokens"], slot))].append(row)
    by_slot: dict[str, list[tuple[str, list[Mapping[str, Any]]]]] = defaultdict(list)
    for (slot, value), group in groups.items():
        by_slot[slot].append((value, group))
    rng = random.Random(int(seed))
    sampled: list[Mapping[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for slot in sorted(by_slot):
        maximum = max(len(group) for _, group in by_slot[slot])
        counts[slot] = {}
        for value, group in sorted(by_slot[slot]):
            target = max(len(group), int(math.ceil(math.sqrt(len(group) * maximum))))
            repeated = [group[index % len(group)] for index in range(target)]
            rng.shuffle(repeated)
            sampled.extend(repeated)
            counts[slot][value] = len(repeated)
    rng.shuffle(sampled)
    return sampled, counts


def _slot_prediction(model: CausalMoELanguageModel, row: Mapping[str, Any], vocabulary: Mapping[str, int], device: Any) -> str:
    """Predict one field token after the exact schema-query used in training.

    The slotwise dataset trains on ``context + SLOT_QUERY_BOS + slot_query +
    SLOT_QUERY_EOS + TARGET_BOS``.  Evaluation must provide that same prefix;
    omitting it changes the conditional task and can make the negative-control
    metric look like a model failure when it is actually a protocol mismatch.
    """

    import torch

    unknown = int(vocabulary[UNK])
    reverse = {int(index): str(token) for token, index in vocabulary.items()}
    slot = str(row["slot"])
    sequence = [int(vocabulary.get(str(token), unknown)) for token in row["context_tokens"]]
    query = [SLOT_QUERY_BOS, f"slot_query={slot}", SLOT_QUERY_EOS, TARGET_BOS]
    sequence.extend(int(vocabulary.get(token, unknown)) for token in query)
    allowed = [int(index) for token, index in vocabulary.items() if str(token).startswith(str(row["slot"]) + "=")]
    if not allowed:
        return f"{row['slot']}=unknown"
    with torch.inference_mode():
        input_ids = torch.tensor(sequence[-model.config.max_length :], dtype=torch.long, device=device).unsqueeze(0)
        logits, _ = model(input_ids, valid_mask=torch.ones_like(input_ids, dtype=torch.bool))
        scores = logits[0, -1].clone()
        allowed_ids = torch.tensor(allowed, dtype=torch.long, device=device)
        mask = torch.full_like(scores, float("-inf"))
        mask[allowed_ids] = scores[allowed_ids]
        return reverse.get(int(mask.argmax(-1).detach().cpu()), f"{row['slot']}=unknown")


def _guard_prediction(row: Mapping[str, Any], proposal: Mapping[str, str]) -> dict[str, str]:
    """Apply the abstract fail-closed Rule-IR guard without reading targets."""

    normalized: dict[str, str] = {}
    for key, value in proposal.items():
        text = str(value)
        normalized[str(key)] = text.split("=", 1)[1] if "=" in text else text
    constrained = constrain_rule_ir(row["context_tokens"], normalized)
    output = dict(proposal)
    for key, value in dict(constrained["target"]).items():
        output[key] = f"{key}={value}"
    return output


def _predictive_entropy(model: CausalMoELanguageModel, rows: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], device: Any, *, limit: int = 512) -> float:
    import torch
    import torch.nn.functional as F

    sample = list(rows[:limit])
    values: list[float] = []
    model.eval()
    with torch.inference_mode():
        for row in sample:
            ids = [int(vocabulary.get(str(token), vocabulary[UNK])) for token in [*row["context_tokens"], *row["target_tokens"]]]
            input_ids = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(input_ids, valid_mask=torch.ones_like(input_ids, dtype=torch.bool))
            probs = F.softmax(logits[0], dim=-1)
            values.extend(float(value) for value in (-(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)).detach().cpu())
    return round(sum(values) / max(len(values), 1), 6)


def _evaluate(model: CausalMoELanguageModel, rows: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], device: Any, *, apply_guard: bool = False, slot_order: Sequence[str] | None = None) -> dict[str, Any]:
    slots = tuple(slot_order or SLOTS)
    grouped: dict[str, dict[str, str]] = defaultdict(dict)
    expected: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        source = str(row["source_record_digest"])
        grouped[source][str(row["slot"])] = _slot_prediction(model, row, vocabulary, device)
        expected[source][str(row["slot"])] = _slot_value(row["target_tokens"], str(row["slot"]))
    if apply_guard:
        guarded: dict[str, dict[str, str]] = {}
        for source, predictions in grouped.items():
            # The guard only sees the context.  All slots for a source share it.
            source_row = next(row for row in rows if str(row["source_record_digest"]) == source)
            guarded[source] = _guard_prediction(source_row, predictions)
        grouped = guarded
    slot_correct = Counter()
    slot_total = Counter()
    for source, values in expected.items():
        for slot, value in values.items():
            slot_total[slot] += 1
            slot_correct[slot] += int(grouped[source].get(slot) == value)
    def rate(slot: str) -> float:
        return round(slot_correct[slot] / max(slot_total[slot], 1), 6)
    ask_total = ask_correct = repair_total = repair_correct = abstain_total = abstain_correct = positive_action_total = positive_action_correct = positive_total = positive_correct = negative_total = false_allow = 0
    assembled_exact = 0
    for source, exp in expected.items():
        pred = grouped[source]
        if exp.get("question", "").split("=", 1)[-1].startswith("ask_"):
            ask_total += 1
            ask_correct += int(pred.get("question") == exp.get("question"))
        if exp.get("next_action") == "next_action=repair":
            repair_total += 1
            repair_correct += int(pred.get("next_action") == exp.get("next_action"))
        if exp.get("next_action") == "next_action=abstain":
            abstain_total += 1
            abstain_correct += int(pred.get("next_action") == exp.get("next_action"))
        if exp.get("next_action") in {"next_action=select_probe_variant", "next_action=replay"}:
            positive_action_total += 1
            positive_action_correct += int(pred.get("next_action") == exp.get("next_action"))
        safe = exp.get("safe_to_send") == "safe_to_send=1"
        predicted_safe = pred.get("safe_to_send") == "safe_to_send=1"
        if safe:
            positive_total += 1
            positive_correct += int(predicted_safe)
        else:
            negative_total += 1
            false_allow += int(predicted_safe)
        assembled_exact += int(all(pred.get(slot) == value for slot, value in exp.items()))
    return {
        "source_rows": len(expected),
        "slot_accuracy": {slot: rate(slot) for slot in slots},
        "rule_ir_assembly_exact": round(assembled_exact / max(len(expected), 1), 6),
        "ask_recall": round(ask_correct / max(ask_total, 1), 6),
        "repair_recall": round(repair_correct / max(repair_total, 1), 6),
        "abstain_recall": round(abstain_correct / max(abstain_total, 1), 6),
        "positive_action_recall": round(positive_action_correct / max(positive_action_total, 1), 6),
        "positive_recall": round(positive_correct / max(positive_total, 1), 6),
        "negative_false_allow": false_allow,
        "negative_total": negative_total,
    }


def _gate(dataset: Mapping[str, Any], audit: Mapping[str, Any], env: Mapping[str, str], device: Mapping[str, Any], locks: Mapping[str, str], train_rows: Sequence[Mapping[str, Any]], holdout_rows: Sequence[Mapping[str, Any]], train_failures: Sequence[str], holdout_failures: Sequence[str], now: datetime, slot_order: Sequence[str] | None = None) -> dict[str, Any]:
    slots = tuple(slot_order or _slot_order(dataset))
    checks = {
        "weekend_remote_lane": _weekend(now),
        "explicit_training_flag": env.get("BLACKBOX_REMOTE_A800_TRAIN") == "1",
        "cuda_visible_devices_zero": env.get("CUDA_VISIBLE_DEVICES") == "0",
        "single_visible_a800_gpu0": bool(device.get("cuda_available") is True and device.get("visible_device_count") == 1 and device.get("current_device") == 0 and "A800" in str(device.get("name", ""))),
        "dataset_candidate_only": dataset.get("status") == "diagnostic_candidate_only",
        "audit_candidate_only": audit.get("status") == "diagnostic_candidate_only" and not audit.get("failures"),
        "all_slots_present": set(str(row.get("slot")) for row in [*train_rows, *holdout_rows]) == set(slots),
        "vocabulary_locked": bool(dataset.get("vocabulary")),
        "promotion_closed": all(value is False for value in dict(audit.get("promotion") or {}).values()),
        "hashes_locked": all(len(str(value)) == 64 for value in locks.values()),
    }
    failures = [key for key, passed in checks.items() if not passed] + list(train_failures) + list(holdout_failures)
    return {"schema_version": f"{SCHEMA_VERSION}-gate", "status": "ready" if not failures else "blocked", "checks": checks, "failures": sorted(set(failures)), "training_allowed": not failures, "candidate_only": True, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-weight", type=float, default=8.0)
    parser.add_argument("--context-weight", type=float, default=0.25)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--experts", type=int, default=2)
    parser.add_argument("--expert-hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--balance-slot-values", action="store_true")
    parser.add_argument("--sqrt-balance-slot-values", action="store_true")
    parser.add_argument("--apply-rule-ir-guard", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text(encoding="utf-8-sig"))
    dataset, audit_report = load(args.dataset), load(args.audit)
    slot_order = _slot_order(dataset)
    train_rows, train_failures = _rows(dataset, "train", slot_order)
    holdout_rows, holdout_failures = _rows(dataset, "implementation_holdout", slot_order)
    locks = {"dataset": _sha_file(args.dataset), "audit": _sha_file(args.audit), "rules": _sha_file(args.rules), "script": _sha_file(Path(__file__)), "model": _sha_file(ROOT / "app" / "pg295_causal_moe.py")}
    import torch

    device_info = {"cuda_available": bool(torch.cuda.is_available()), "visible_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0, "current_device": int(torch.cuda.current_device()) if torch.cuda.is_available() else -1, "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    gate = _gate(dataset, audit_report, os.environ, device_info, locks, train_rows, holdout_rows, train_failures, holdout_failures, datetime.now(TZ), slot_order)
    if not gate["training_allowed"]:
        raise RuntimeError("PG-360 gate blocked: " + ",".join(gate["failures"]))
    if not 1 <= args.epochs <= 16 or not 1 <= args.batch_size <= 512 or not 0 < args.learning_rate <= 0.01:
        raise ValueError("training bounds invalid")
    if args.balance_slot_values and args.sqrt_balance_slot_values:
        raise ValueError("choose at most one slot balancing strategy")
    if not 32 <= args.d_model <= 1024 or not 1 <= args.n_heads <= 32 or args.d_model % args.n_heads != 0 or not 1 <= args.n_layers <= 24 or not 1 <= args.experts <= 16 or not 32 <= args.expert_hidden <= 4096 or not 0 <= args.dropout < 1:
        raise ValueError("model capacity bounds invalid")
    vocabulary_map = _vocabulary(dataset)
    all_rows = [*train_rows, *holdout_rows]
    max_length = max(len(row["context_tokens"]) + len(row["target_tokens"]) for row in all_rows)
    config = CausalMoEConfig(d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, experts=args.experts, expert_hidden=args.expert_hidden, dropout=args.dropout, max_length=max_length)
    target_weights = {str(token): float(args.context_weight) for token in dataset["vocabulary"].get("context_tokens") or []}
    target_weights.update({str(token): float(args.target_weight) for token in dataset["vocabulary"].get("target_tokens") or []})
    device = torch.device("cuda:0")
    candidates: list[dict[str, Any]] = []
    states: dict[str, Mapping[str, Any]] = {}
    fit_counts: dict[str, dict[str, dict[str, int]]] = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        baseline = CausalMoELanguageModel(vocab_size=len(vocabulary_map), config=config).to(device)
        baseline_entropy = _predictive_entropy(baseline, holdout_rows, vocabulary_map, device)
        if args.sqrt_balance_slot_values:
            fit_rows, value_counts = _sqrt_balanced_slot_rows(train_rows, seed)
        elif args.balance_slot_values:
            fit_rows, value_counts = _balanced_slot_rows(train_rows, seed)
        else:
            fit_rows, value_counts = list(train_rows), {}
        fit_counts[str(seed)] = value_counts
        model = train_causal_moe(fit_rows, vocabulary_map, device, seed=seed, config=config, epochs=args.epochs, learning_rate=args.learning_rate, token_weights=target_weights, normalize_weighted_loss=True, batch_size=args.batch_size)
        post_entropy = _predictive_entropy(model, holdout_rows, vocabulary_map, device)
        relative_drop = round((baseline_entropy - post_entropy) / max(abs(baseline_entropy), 1e-12), 6)
        metrics = _evaluate(model, holdout_rows, vocabulary_map, device, apply_guard=args.apply_rule_ir_guard, slot_order=slot_order)
        raw_metrics = _evaluate(model, holdout_rows, vocabulary_map, device, apply_guard=False, slot_order=slot_order) if args.apply_rule_ir_guard else None
        metrics["baseline_predictive_entropy"] = baseline_entropy
        metrics["post_predictive_entropy"] = post_entropy
        metrics["relative_entropy_drop"] = relative_drop
        if raw_metrics is not None:
            metrics["raw_model_metrics"] = raw_metrics
        candidates.append({"seed": seed, "implementation_holdout": metrics})
        states[str(seed)] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    worst = {
        "max_relative_entropy_drop": max(float(item["implementation_holdout"]["relative_entropy_drop"]) for item in candidates),
        "entropy_gate_passed": max(float(item["implementation_holdout"]["relative_entropy_drop"]) for item in candidates) <= 0.25,
        "ask_recall_min": min(float(item["implementation_holdout"]["ask_recall"]) for item in candidates),
        "repair_recall_min": min(float(item["implementation_holdout"]["repair_recall"]) for item in candidates),
        "abstain_recall_min": min(float(item["implementation_holdout"]["abstain_recall"]) for item in candidates),
        "positive_action_recall_min": min(float(item["implementation_holdout"]["positive_action_recall"]) for item in candidates),
        "positive_recall_min": min(float(item["implementation_holdout"]["positive_recall"]) for item in candidates),
        "negative_false_allow_max": max(int(item["implementation_holdout"]["negative_false_allow"]) for item in candidates),
        "rule_ir_assembly_exact_min": min(float(item["implementation_holdout"]["rule_ir_assembly_exact"]) for item in candidates),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "slotwise_candidate_only",
        "gate": gate,
        "locks": locks,
        "training": {"device": "cuda:0", "gpu": device_info.get("name"), "seeds": list(SEEDS), "epochs": args.epochs, "learning_rate": args.learning_rate, "batch_size": args.batch_size, "target_weight": args.target_weight, "context_weight": args.context_weight, "balance_slot_values": args.balance_slot_values, "sqrt_balance_slot_values": args.sqrt_balance_slot_values, "balance_strategy": "sqrt_geometric" if args.sqrt_balance_slot_values else ("full_equal" if args.balance_slot_values else "empirical"), "apply_rule_ir_guard": args.apply_rule_ir_guard, "d_model": args.d_model, "n_heads": args.n_heads, "n_layers": args.n_layers, "experts": args.experts, "expert_hidden": args.expert_hidden, "dropout": args.dropout, "fit_value_counts": fit_counts, "max_length": max_length, "source_rows": len({str(row.get('source_record_digest')) for row in [*train_rows, *holdout_rows]}), "slot_records": len(train_rows) + len(holdout_rows), "target_information_in_context": False, "candidate_only": True},
        "slot_order": list(slot_order),
        "candidates": candidates,
        "worst_seed": worst,
        "scientific_gate": {"status": "blocked_candidate_only", "raw_payload_in_context": False, "typed_live_replay_with_model_selected_wire": False, "independent_implementation": False, "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    report["report_sha256"] = _sha_json(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": SCHEMA_VERSION, "config": config.__dict__, "vocabulary": vocabulary_map, "states": states, "promotion": report["promotion"]}, args.checkpoint)
    print(json.dumps(report if args.json else {"status": report["status"], "worst_seed": report["worst_seed"]}, ensure_ascii=False, indent=2 if args.json else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
