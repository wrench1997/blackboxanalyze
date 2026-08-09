"""Remote A800 candidate for PG-351 ASK + Rule-IR composition.

The runner reads only abstract context/target tokens from the PG-351
candidate dataset.  It is explicitly *not* training-eligible capability
data: no payload, response, route, family, evaluator answer, or wire is
read, and all promotion flags remain false.  Its purpose is to measure
whether a causal next-token model can jointly preserve ASK, repair, abstain,
replay, and abstract Rule-IR targets before any live model-selected probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg293_failure_next_action import PAD, UNK  # noqa: E402
from app.pg295_causal_moe import CausalMoEConfig, CausalMoELanguageModel, generate_target, train_causal_moe  # noqa: E402


SCHEMA_VERSION = "pg351-a800-ask-oracle-composition-candidate-v1"
TZ = ZoneInfo("Asia/Shanghai")
SEEDS = (35101, 35102, 35103)
TARGET_PREFIXES = ("[TARGET_BOS]", "[TARGET_EOS]", "question=", "ask_reason=", "next_action=", "repair_action=", "transport_ref=", "field_role_ref=", "encoding_ref=", "probe_variant_ref=", "safe_to_send=", "payload_shape_ref=", "oracle_ref=", "negative_control_presence_ref=")
# The field order is an ontology contract, not a row label.  Constrained
# decoding uses it to prevent free-running syntax drift while the model still
# chooses every field value.
TARGET_KEY_ORDER = ("question", "ask_reason", "next_action", "repair_action", "safe_to_send", "transport_ref", "field_role_ref", "encoding_ref", "probe_variant_ref", "payload_shape_ref", "oracle_ref", "negative_control_presence_ref")
RAW_FRAGMENTS = ("raw_payload=", "payload=", "response_body=", "response_body_text=", "raw_response=", "wire=", "evaluator=", "oracle=", "route_literal=", "family=", "implementation=", "image=", "source=")


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _weekend(now: datetime) -> bool:
    local = now.astimezone(TZ) if now.tzinfo else now.replace(tzinfo=TZ)
    return local.weekday() >= 5


def _target_value(tokens: Sequence[str], key: str, default: str = "") -> str:
    prefix = key + "="
    return next((str(token)[len(prefix) :] for token in tokens if str(token).startswith(prefix)), default)


def _rows(dataset: Mapping[str, Any], split: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for index, raw in enumerate(dataset.get("records") or []):
        if not isinstance(raw, Mapping) or str(raw.get("split")) != split:
            continue
        context = raw.get("context_tokens")
        target = raw.get("target_tokens")
        firewall = raw.get("context_firewall")
        if not isinstance(context, list) or len(context) < 2 or not isinstance(target, list) or target[:1] != ["[TARGET_BOS]"] or target[-1:] != ["[TARGET_EOS]"]:
            failures.append(f"row_{index}_stream")
            continue
        if firewall != {"forbidden_token_count": 0, "sidecars_off_context": True}:
            failures.append(f"row_{index}_firewall")
            continue
        if any(raw.get(flag) is not False for flag in ("raw_payload_stored", "raw_response_body_stored", "oracle_answer_in_context")):
            failures.append(f"row_{index}_raw_flag")
            continue
        context_normalized = [str(token) for token in context]
        target_normalized = [str(token) for token in target]
        if any(any(fragment in token.casefold() for fragment in RAW_FRAGMENTS) for token in [*context_normalized, *target_normalized]):
            failures.append(f"row_{index}_raw_token")
            continue
        if any(not token.startswith(TARGET_PREFIXES) for token in target_normalized):
            failures.append(f"row_{index}_nonabstract_target")
            continue
        rows.append({"context_tokens": context_normalized, "target_tokens": target_normalized, "safe_to_send": "safe_to_send=1" in target_normalized, "supervision_lane": str(raw.get("supervision_lane", ""))})
    return rows, sorted(set(failures))


def _coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    questions = Counter(_target_value(row["target_tokens"], "question", "none") for row in rows)
    actions = Counter(_target_value(row["target_tokens"], "next_action", "none") for row in rows)
    return {
        "rows": len(rows),
        "questions": dict(sorted(questions.items())),
        "next_actions": dict(sorted(actions.items())),
        "ask_present": questions.get("ask_typed", 0) > 0 or questions.get("ask_failure", 0) > 0,
        "repair_present": actions.get("repair", 0) > 0,
        "abstain_present": actions.get("abstain", 0) > 0,
        "positive_present": actions.get("select_probe_variant", 0) > 0 or actions.get("replay", 0) > 0,
    }


def _balanced_action_rows(rows: Sequence[Mapping[str, Any]], seed: int) -> tuple[list[Mapping[str, Any]], dict[str, int]]:
    """Oversample only the training view so every abstract action is seen equally.

    This changes optimization frequency, not the source/holdout data.  Rows
    remain immutable abstract records; the duplicate count is reported so it
    cannot be mistaken for additional information.
    """
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        action = _target_value(row["target_tokens"], "next_action", "none")
        groups.setdefault(action, []).append(row)
    if not groups:
        return [], {}
    target = max(len(group) for group in groups.values())
    rng = random.Random(int(seed))
    balanced: list[Mapping[str, Any]] = []
    counts: dict[str, int] = {}
    for action in sorted(groups):
        group = list(groups[action])
        repeated = [group[index % len(group)] for index in range(target)]
        rng.shuffle(repeated)
        balanced.extend(repeated)
        counts[action] = len(repeated)
    rng.shuffle(balanced)
    return balanced, counts


def _predictive_entropy(model: CausalMoELanguageModel, rows: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], device: Any) -> float:
    import torch
    import torch.nn.functional as F

    values: list[float] = []
    model.eval()
    with torch.inference_mode():
        for row in rows:
            tokens = [*row["context_tokens"], *row["target_tokens"]]
            ids = [int(vocabulary.get(str(token), vocabulary[UNK])) for token in tokens]
            if len(ids) < 2:
                continue
            input_ids = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(input_ids, valid_mask=torch.ones_like(input_ids, dtype=torch.bool))
            probabilities = F.softmax(logits[0], dim=-1)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
            values.extend(float(value) for value in entropy.detach().cpu())
    return round(sum(values) / max(len(values), 1), 6)


def _generate_constrained_target(model: CausalMoELanguageModel, context_tokens: Sequence[str], vocabulary: Mapping[str, int], device: Any) -> list[str]:
    """Decode Rule-IR values under the fixed abstract field grammar.

    This does not supply the answer values.  At each position the model still
    selects among all vocabulary values for that field; the mask only blocks
    malformed cross-slot tokens and repeated BOS/EOS drift.
    """
    import torch

    unknown = int(vocabulary[UNK])
    reverse = {int(index): str(token) for token, index in vocabulary.items()}
    sequence = [int(vocabulary.get(str(token), unknown)) for token in context_tokens]
    field_ids: dict[str, list[int]] = {
        key: [int(index) for token, index in vocabulary.items() if str(token).startswith(key + "=")]
        for key in TARGET_KEY_ORDER
    }
    bos = int(vocabulary["[TARGET_BOS]"])
    eos = int(vocabulary["[TARGET_EOS]"])
    output: list[str] = ["[TARGET_BOS]"]
    with torch.inference_mode():
        # Consume the target BOS at the context boundary before decoding any
        # field.  Omitting this step shifts every Rule-IR slot by one.
        input_ids = torch.tensor(sequence[-model.config.max_length :], dtype=torch.long, device=device).unsqueeze(0)
        logits, _ = model(input_ids, valid_mask=torch.ones_like(input_ids, dtype=torch.bool))
        bos_scores = logits[0, -1].clone()
        bos_only = torch.full_like(bos_scores, float("-inf"))
        bos_only[bos] = bos_scores[bos]
        sequence.append(int(bos_only.argmax(-1).detach().cpu()))
        for position, key in enumerate(TARGET_KEY_ORDER):
            input_ids = torch.tensor(sequence[-model.config.max_length :], dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(input_ids, valid_mask=torch.ones_like(input_ids, dtype=torch.bool))
            allowed = field_ids[key]
            if not allowed:
                return []
            scores = logits[0, -1].clone()
            mask = torch.full_like(scores, float("-inf"))
            mask[torch.tensor(allowed, dtype=torch.long, device=device)] = scores[torch.tensor(allowed, dtype=torch.long, device=device)]
            next_id = int(mask.argmax(-1).detach().cpu())
            sequence.append(next_id)
            output.append(reverse.get(next_id, UNK))
        input_ids = torch.tensor(sequence[-model.config.max_length :], dtype=torch.long, device=device).unsqueeze(0)
        logits, _ = model(input_ids, valid_mask=torch.ones_like(input_ids, dtype=torch.bool))
        scores = logits[0, -1].clone()
        eos_only = torch.full_like(scores, float("-inf"))
        eos_only[eos] = scores[eos]
        sequence.append(eos)
        output.append(reverse.get(int(eos_only.argmax(-1).detach().cpu()), UNK))
    return output


def _generated_metrics(model: CausalMoELanguageModel, rows: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], device: Any, *, constrained: bool = False) -> dict[str, Any]:
    totals = Counter()
    for row in rows:
        expected = [str(token) for token in row["target_tokens"]]
        predicted = _generate_constrained_target(model, row["context_tokens"], vocabulary, device) if constrained else generate_target(model, row["context_tokens"], len(expected), vocabulary, device)
        expected_body = expected[1:]
        predicted_body = predicted[1 : 1 + len(expected_body)] if predicted and predicted[0] == "[TARGET_BOS]" else predicted[: len(expected_body)]
        totals["rows"] += 1
        totals["token_total"] += len(expected_body)
        totals["token_correct"] += sum(int(a == b) for a, b in zip(predicted_body, expected_body))
        totals["sequence_exact"] += int(predicted_body == expected_body)
        expected_question = _target_value(expected_body, "question", "none")
        predicted_question = _target_value(predicted_body, "question", "none")
        expected_action = _target_value(expected_body, "next_action", "none")
        predicted_action = _target_value(predicted_body, "next_action", "none")
        expected_safe = bool(row.get("safe_to_send"))
        predicted_safe = "safe_to_send=1" in predicted_body
        if expected_question.startswith("ask_"):
            totals["ask_total"] += 1
            totals["ask_correct"] += int(predicted_question == expected_question)
        else:
            totals["normal_total"] += 1
            totals["unnecessary_question"] += int(predicted_question != "none")
        if expected_action == "repair":
            totals["repair_total"] += 1
            totals["repair_correct"] += int(predicted_action == "repair")
        if expected_action == "abstain":
            totals["abstain_total"] += 1
            totals["abstain_correct"] += int(predicted_action == "abstain")
        if expected_action in {"select_probe_variant", "replay"}:
            totals["positive_action_total"] += 1
            totals["positive_action_correct"] += int(predicted_action == expected_action)
        if expected_safe:
            totals["positive_total"] += 1
            totals["positive_correct"] += int(predicted_safe)
        else:
            totals["negative_total"] += 1
            totals["negative_false_allow"] += int(predicted_safe)
    return {
        "rows": totals["rows"],
        "token_accuracy": round(totals["token_correct"] / max(totals["token_total"], 1), 6),
        "sequence_exact_accuracy": round(totals["sequence_exact"] / max(totals["rows"], 1), 6),
        "ask_recall": round(totals["ask_correct"] / max(totals["ask_total"], 1), 6) if totals["ask_total"] else None,
        "repair_recall": round(totals["repair_correct"] / max(totals["repair_total"], 1), 6) if totals["repair_total"] else None,
        "abstain_recall": round(totals["abstain_correct"] / max(totals["abstain_total"], 1), 6) if totals["abstain_total"] else None,
        "positive_action_recall": round(totals["positive_action_correct"] / max(totals["positive_action_total"], 1), 6) if totals["positive_action_total"] else None,
        "positive_recall": round(totals["positive_correct"] / max(totals["positive_total"], 1), 6) if totals["positive_total"] else None,
        "negative_false_allow": totals["negative_false_allow"],
        "negative_total": totals["negative_total"],
        "unnecessary_question_rate": round(totals["unnecessary_question"] / max(totals["normal_total"], 1), 6) if totals["normal_total"] else None,
    }


def evaluate_gate(*, dataset: Mapping[str, Any], audit: Mapping[str, Any], env: Mapping[str, str], device: Mapping[str, Any], locks: Mapping[str, str], train_rows: Sequence[Mapping[str, Any]], train_failures: Sequence[str], holdout_rows: Sequence[Mapping[str, Any]], holdout_failures: Sequence[str], now: datetime) -> dict[str, Any]:
    train_coverage = _coverage(train_rows)
    holdout_coverage = _coverage(holdout_rows)
    vocabulary = dataset.get("vocabulary") if isinstance(dataset.get("vocabulary"), Mapping) else {}
    vocab = {str(token) for token in [*(vocabulary.get("context_tokens") or []), *(vocabulary.get("target_tokens") or [])]}
    unknown = sorted({str(token) for row in [*train_rows, *holdout_rows] for token in [*row["context_tokens"], *row["target_tokens"]]} - vocab)
    checks = {
        "weekend_remote_lane": _weekend(now),
        "explicit_training_flag": env.get("BLACKBOX_REMOTE_A800_TRAIN") == "1",
        "cuda_visible_devices_zero": env.get("CUDA_VISIBLE_DEVICES") == "0",
        "single_visible_a800_gpu0": bool(device.get("cuda_available") is True and device.get("visible_device_count") == 1 and device.get("current_device") == 0 and "A800" in str(device.get("name", ""))),
        "dataset_status_candidate_only": dataset.get("status") == "diagnostic_candidate_only",
        "audit_status_candidate_only": audit.get("status") == "diagnostic_candidate_only" and not audit.get("failures"),
        "target_coverage_train": all(train_coverage[key] for key in ("ask_present", "repair_present", "abstain_present", "positive_present")),
        "target_coverage_holdout": all(holdout_coverage[key] for key in ("ask_present", "repair_present", "abstain_present", "positive_present")),
        "vocabulary_locked": not unknown and bool(vocab),
        "promotion_closed": all(value is False for value in dict(audit.get("promotion") or {}).values()),
        "data_code_vocab_rules_hashes_locked": all(len(str(value)) == 64 for value in locks.values()),
    }
    failures = [key for key, passed in checks.items() if not passed] + list(train_failures) + list(holdout_failures)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ready_target_conditioned_candidate" if not failures else "blocked",
        "checks": checks,
        "failures": sorted(set(failures)),
        "training_allowed": not failures,
        "split_counts": {"train": len(train_rows), "implementation_holdout": len(holdout_rows)},
        "target_coverage": {"train": train_coverage, "implementation_holdout": holdout_coverage},
        "unknown_token_count": len(unknown),
        "candidate_only": True,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PG-351 abstract ASK/oracle A800 candidate")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--target-weight", type=float, default=8.0)
    parser.add_argument("--context-weight", type=float, default=0.25)
    parser.add_argument("--repair-action-weight", type=float, default=None)
    parser.add_argument("--abstain-action-weight", type=float, default=None)
    parser.add_argument("--replay-action-weight", type=float, default=None)
    parser.add_argument("--safe-zero-weight", type=float, default=None)
    parser.add_argument("--safe-one-weight", type=float, default=None)
    parser.add_argument("--constrained-rule-ir", action="store_true")
    parser.add_argument("--balance-actions", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--experts", type=int, default=2)
    parser.add_argument("--expert-hidden", type=int, default=256)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text(encoding="utf-8-sig"))
    dataset, audit, rules = load(args.dataset), load(args.audit), load(args.rules)
    train_rows, train_failures = _rows(dataset, "train")
    holdout_rows, holdout_failures = _rows(dataset, "implementation_holdout")
    locks = {"dataset": _sha_file(args.dataset), "audit": _sha_file(args.audit), "rules": _sha_file(args.rules), "script": _sha_file(Path(__file__)), "model": _sha_file(ROOT / "app" / "pg295_causal_moe.py")}
    import torch

    device_info = {"cuda_available": bool(torch.cuda.is_available()), "visible_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0, "current_device": int(torch.cuda.current_device()) if torch.cuda.is_available() else -1, "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    gate = evaluate_gate(dataset=dataset, audit=audit, env=os.environ, device=device_info, locks=locks, train_rows=train_rows, train_failures=train_failures, holdout_rows=holdout_rows, holdout_failures=holdout_failures, now=datetime.now(TZ))
    if not gate["training_allowed"]:
        raise RuntimeError("PG-351 target-conditioned gate blocked: " + ",".join(gate["failures"]))
    if not 0 < args.learning_rate <= 0.01 or not 1 <= args.epochs <= 32 or not 1 <= args.batch_size <= 256:
        raise ValueError("training bounds invalid")
    vocabulary = dataset["vocabulary"]
    ordered = [PAD, UNK, *list(dict.fromkeys([*(vocabulary.get("context_tokens") or []), *(vocabulary.get("target_tokens") or [])]))]
    vocabulary_map = {str(token): index for index, token in enumerate(dict.fromkeys(ordered))}
    max_length = max([len(row["context_tokens"]) + len(row["target_tokens"]) for row in [*train_rows, *holdout_rows]] or [2])
    if args.d_model <= 0 or args.n_heads <= 0 or args.n_layers <= 0 or args.experts <= 0 or args.expert_hidden <= 0 or args.d_model % args.n_heads:
        raise ValueError("model capacity arguments invalid")
    config = CausalMoEConfig(d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, experts=args.experts, expert_hidden=args.expert_hidden, max_length=max_length)
    target_weight = {str(token): float(args.context_weight) for token in vocabulary.get("context_tokens") or []}
    target_weight.update({str(token): float(args.target_weight) for token in vocabulary.get("target_tokens") or []})
    for token, override in {
        "next_action=repair": args.repair_action_weight,
        "next_action=abstain": args.abstain_action_weight,
        "next_action=replay": args.replay_action_weight,
        "safe_to_send=0": args.safe_zero_weight,
        "safe_to_send=1": args.safe_one_weight,
    }.items():
        if override is not None:
            if float(override) <= 0:
                raise ValueError(f"{token} weight must be positive")
            target_weight[token] = float(override)
    device = torch.device("cuda:0")
    candidates: list[dict[str, Any]] = []
    states: dict[str, Mapping[str, Any]] = {}
    fit_counts: dict[str, dict[str, int]] = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        baseline = CausalMoELanguageModel(vocab_size=len(vocabulary_map), config=config).to(device)
        baseline_entropy = _predictive_entropy(baseline, holdout_rows, vocabulary_map, device)
        fit_rows, action_counts = _balanced_action_rows(train_rows, seed) if args.balance_actions else (list(train_rows), dict(_coverage(train_rows)["next_actions"]))
        fit_counts[str(seed)] = action_counts
        model = train_causal_moe(fit_rows, vocabulary_map, device, seed=seed, config=config, epochs=args.epochs, learning_rate=args.learning_rate, token_weights=target_weight, normalize_weighted_loss=True, batch_size=args.batch_size)
        post_entropy = _predictive_entropy(model, holdout_rows, vocabulary_map, device)
        relative_drop = round((baseline_entropy - post_entropy) / max(abs(baseline_entropy), 1e-12), 6)
        train_metrics = _generated_metrics(model, train_rows, vocabulary_map, device, constrained=args.constrained_rule_ir)
        holdout_metrics = _generated_metrics(model, holdout_rows, vocabulary_map, device, constrained=args.constrained_rule_ir)
        candidates.append({"seed": seed, "train": train_metrics, "implementation_holdout": holdout_metrics, "baseline_holdout_predictive_entropy": baseline_entropy, "post_holdout_predictive_entropy": post_entropy, "relative_entropy_drop": relative_drop})
        states[str(seed)] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "target_conditioned_candidate_only",
        "gate": gate,
        "locks": locks,
        "training": {"device": "cuda:0", "gpu": device_info.get("name"), "seeds": list(SEEDS), "epochs": args.epochs, "learning_rate": args.learning_rate, "target_weight": args.target_weight, "context_weight": args.context_weight, "action_weight_overrides": {"next_action=repair": args.repair_action_weight, "next_action=abstain": args.abstain_action_weight, "next_action=replay": args.replay_action_weight, "safe_to_send=0": args.safe_zero_weight, "safe_to_send=1": args.safe_one_weight}, "constrained_rule_ir": args.constrained_rule_ir, "balance_actions": args.balance_actions, "fit_action_counts": fit_counts, "model_capacity": {"d_model": args.d_model, "n_heads": args.n_heads, "n_layers": args.n_layers, "experts": args.experts, "expert_hidden": args.expert_hidden, "max_length": max_length}, "batch_size": args.batch_size, "context_only": False, "target_tokens_read": True, "required_max_length": max_length, "candidate_only": True},
        "target_coverage": gate["target_coverage"],
        "candidates": candidates,
        "worst_seed": {
            "max_relative_entropy_drop": max(float(item["relative_entropy_drop"]) for item in candidates),
            "entropy_gate_passed": max(float(item["relative_entropy_drop"]) for item in candidates) <= 0.25,
            "negative_false_allow_max": max(int(item["implementation_holdout"]["negative_false_allow"]) for item in candidates),
            "ask_recall_min": min(float(item["implementation_holdout"]["ask_recall"] or 0.0) for item in candidates),
            "repair_recall_min": min(float(item["implementation_holdout"]["repair_recall"] or 0.0) for item in candidates),
            "abstain_recall_min": min(float(item["implementation_holdout"]["abstain_recall"] or 0.0) for item in candidates),
            "positive_action_recall_min": min(float(item["implementation_holdout"]["positive_action_recall"] or 0.0) for item in candidates),
            "positive_recall_min": min(float(item["implementation_holdout"]["positive_recall"] or 0.0) for item in candidates),
        },
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
