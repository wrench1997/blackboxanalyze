"""PG-343 full-axis target-conditioned ASK/repair/negative A800 smoke.

This is a candidate-only causal next-token experiment.  It reads abstract
context and abstract Rule-IR target tokens; it never reads raw payloads,
response bodies, routes, evaluator answers, or sidecars.  The dataset is
diagnostic-only, so this runner can produce a research checkpoint but can
never open training/memory/payload/vulnerability promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from app.pg295_causal_moe import CausalMoEConfig, CausalMoELanguageModel, evaluate_causal_moe, generate_target, train_causal_moe  # noqa: E402

SCHEMA_VERSION = "pg343-a800-target-conditioned-full-axis-diagnostic-v1"
TZ = ZoneInfo("Asia/Shanghai")
SEEDS = (34311, 34312, 34313)
FORBIDDEN = ("payload=", "payload_", "response_body=", "response_body_text=", "raw_", "oracle=", "evaluator=", "family=", "route=", "route_literal=", "implementation=", "image=", "url=", "path=", "source=")
TARGET_PREFIXES = ("[TARGET_BOS]", "[TARGET_EOS]", "question=", "next_action=", "repair_action=", "action_changed=", "failure_class=", "safe_to_send=", "transport_ref=", "field_role_ref=", "encoding_ref=", "probe_variant_ref=")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _weekend(now: datetime) -> bool:
    local = now.astimezone(TZ) if now.tzinfo else now.replace(tzinfo=TZ)
    return local.weekday() >= 5


def _load_rows(dataset: Mapping[str, Any], split: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for index, raw in enumerate(dataset.get("records") or []):
        if not isinstance(raw, Mapping) or str(raw.get("split")) != split:
            continue
        context = raw.get("context_tokens")
        target = raw.get("target_tokens")
        if not isinstance(context, list) or len(context) < 2 or not isinstance(target, list) or target[:1] != ["[TARGET_BOS]"] or target[-1:] != ["[TARGET_EOS]"]:
            failures.append(f"row_{index}_token_stream")
            continue
        if not isinstance(raw.get("role_step_binding"), Mapping) or raw["role_step_binding"].get("source_attested") is not True:
            failures.append(f"row_{index}_role_step_binding")
            continue
        if not any(str(token).startswith("belief_probe_role=") for token in context) or not any(str(token).startswith("belief_process_step=") for token in context):
            failures.append(f"row_{index}_role_step_token_missing")
            continue
        firewall = raw.get("context_firewall")
        if firewall != {"forbidden_token_count": 0, "sidecars_off_context": True}:
            failures.append(f"row_{index}_firewall")
            continue
        if any(any(fragment in str(token).casefold() for fragment in FORBIDDEN) for token in [*context, *target]):
            failures.append(f"row_{index}_forbidden_token")
            continue
        if any(not str(token).startswith(TARGET_PREFIXES) for token in target):
            failures.append(f"row_{index}_target_not_abstract")
            continue
        item = {"context_tokens": [str(token) for token in context], "target_tokens": [str(token) for token in target], "safe_to_send": bool((raw.get("target_projection") or {}).get("safe_to_send", False)), "target_projection": dict(raw.get("target_projection") or {}), "role_step_binding": dict(raw.get("role_step_binding") or {})}
        rows.append(item)
    return rows, sorted(set(failures))


def _target_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    questions = Counter()
    actions = Counter()
    role_counts = Counter()
    step_counts = Counter()
    for row in rows:
        target = row.get("target_tokens") or []
        for token in target:
            text = str(token)
            if text.startswith("question="):
                questions[text.split("=", 1)[1]] += 1
            if text.startswith("next_action="):
                actions[text.split("=", 1)[1]] += 1
        for token in row.get("context_tokens") or []:
            text = str(token)
            if text.startswith("belief_probe_role="):
                role_counts[text.split("=", 1)[1]] += 1
            if text.startswith("belief_process_step="):
                step_counts[text.split("=", 1)[1]] += 1
    return {
        "rows": len(rows),
        "questions": dict(sorted(questions.items())),
        "next_actions": dict(sorted(actions.items())),
        "roles": dict(sorted(role_counts.items())),
        "steps": dict(sorted(step_counts.items())),
        "ask_present": any(key.startswith("ask_") for key in questions),
        "repair_present": actions.get("repair", 0) > 0,
        "abstain_present": actions.get("abstain", 0) > 0,
        "positive_present": actions.get("send_probe", 0) > 0 or actions.get("select_probe_variant", 0) > 0 or actions.get("assemble_rule_ir", 0) > 0,
    }


def evaluate_gate(*, dataset: Mapping[str, Any], audit: Mapping[str, Any], vocabulary: Mapping[str, Any], rules: Mapping[str, Any], env: Mapping[str, str], device: Mapping[str, Any], locks: Mapping[str, str], train_rows: Sequence[Mapping[str, Any]], train_failures: Sequence[str], holdout_rows: Sequence[Mapping[str, Any]], holdout_failures: Sequence[str], now: datetime) -> dict[str, Any]:
    coverage_train = _target_coverage(train_rows)
    coverage_holdout = _target_coverage(holdout_rows)
    vocab_tokens = {str(token) for token in [*(vocabulary.get("context_tokens") or []), *(vocabulary.get("target_tokens") or [])]}
    unknown = sorted({str(token) for row in [*train_rows, *holdout_rows] for token in [*row["context_tokens"], *row["target_tokens"]]} - vocab_tokens)
    audit_failures = list(audit.get("failures") or [])
    # PG-347's independent audit keeps aggregate axis measurements under
    # ``counts``.  Accept that canonical location while retaining support for
    # the earlier top-level shape; an absent axis map still fails closed.
    axis_stats = dict(
        audit.get("axis_token_sequence_entropy")
        or (audit.get("counts") or {}).get("axis_token_sequence_entropy")
        or {}
    )
    checks = {
        "weekend_remote_lane": _weekend(now),
        "explicit_training_flag": env.get("BLACKBOX_REMOTE_A800_TRAIN") == "1",
        "cuda_visible_devices_zero": env.get("CUDA_VISIBLE_DEVICES") == "0",
        "single_visible_a800_gpu0": bool(device.get("cuda_available") is True and device.get("visible_device_count") == 1 and device.get("current_device") == 0 and "A800" in str(device.get("name", ""))),
        "dataset_role_bound": all(bool(row.get("role_step_binding", {}).get("source_attested")) for row in [*train_rows, *holdout_rows]),
        "dataset_audit_passed": audit.get("status") == "diagnostic_passed_not_training_eligible" and not audit_failures,
        "full_axis_target_coverage": all(all(coverage[key] for key in ("ask_present", "repair_present", "abstain_present", "positive_present")) for coverage in (coverage_train, coverage_holdout)),
        "implementation_split_isolated": (
            not bool(audit.get("counts", {}).get("context_split_leaks", 1))
            and not bool(audit.get("counts", {}).get("source_record_split_leaks", 1))
            and not bool(audit.get("counts", {}).get("implementation_split_leaks", 1))
        ),
        "axis_sequence_entropy": all(int(item.get("unique_sequences", 0)) >= 2 for item in axis_stats.values()) and len(axis_stats) == 7,
        "context_vocabulary_locked": bool(vocab_tokens) and not unknown and vocabulary.get("append_only") is True and vocabulary.get("forbidden_tokens") == [],
        "promotion_closed": all(value is False for value in dict(audit.get("promotion") or {}).values()) and all(value is False for value in dict(vocabulary.get("promotion") or {}).values()),
        "data_code_vocab_rules_hashes_locked": all(len(str(value)) == 64 for value in locks.values()),
        "rules_schema_present": isinstance(rules, Mapping),
    }
    failures = [key for key, passed in checks.items() if not passed] + list(train_failures) + list(holdout_failures)
    if unknown:
        failures.append("unknown_context_or_target_token")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ready_target_conditioned_diagnostic" if not failures else "blocked",
        "checks": checks,
        "failures": sorted(set(failures)),
        "training_allowed": not failures,
        "track": "full_axis_target_conditioned_candidate_only",
        "target_tokens_read": True,
        "split_counts": {"train": len(train_rows), "implementation_holdout": len(holdout_rows)},
        "target_coverage": {"train": coverage_train, "implementation_holdout": coverage_holdout},
        "unknown_token_count": len(unknown),
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }


def _predictive_entropy(model: CausalMoELanguageModel, rows: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], device: Any) -> float:
    import torch
    from torch.nn import functional as F

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


def _generated_metrics(model: CausalMoELanguageModel, rows: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], device: Any) -> dict[str, Any]:
    totals = Counter()
    variant_actions = {"select_probe_variant", "assemble_rule_ir", "send_probe"}
    for row in rows:
        expected = [str(token) for token in row["target_tokens"]]
        predicted = generate_target(model, row["context_tokens"], len(expected), vocabulary, device)
        expected_actions = {token.split("=", 1)[1] for token in expected if token.startswith("next_action=")}
        expected_question = next((token.split("=", 1)[1] for token in expected if token.startswith("question=")), "none")
        predicted_actions = {token.split("=", 1)[1] for token in predicted if token.startswith("next_action=")}
        predicted_question = next((token.split("=", 1)[1] for token in predicted if token.startswith("question=")), "none")
        expected_safe = bool(row.get("safe_to_send", False))
        predicted_safe = "safe_to_send=1" in predicted
        totals["rows"] += 1
        totals["sequence_exact"] += int(predicted[1:1 + len(expected[1:])] == expected[1:])
        totals["ask_total"] += int(expected_question.startswith("ask_"))
        totals["ask_correct"] += int(expected_question.startswith("ask_") and predicted_question == expected_question)
        totals["repair_total"] += int("repair" in expected_actions)
        totals["repair_correct"] += int("repair" in expected_actions and "repair" in predicted_actions)
        totals["abstain_total"] += int("abstain" in expected_actions)
        totals["abstain_correct"] += int("abstain" in expected_actions and "abstain" in predicted_actions)
        totals["variant_total"] += int(bool(expected_actions & variant_actions))
        totals["variant_correct"] += int(bool(expected_actions & variant_actions) and bool(predicted_actions & variant_actions))
        totals["positive_total"] += int(expected_safe)
        totals["positive_correct"] += int(expected_safe and predicted_safe)
        totals["negative_false_allow"] += int(not expected_safe and predicted_safe)
    return {
        "rows": totals["rows"],
        "sequence_exact_accuracy": round(totals["sequence_exact"] / max(totals["rows"], 1), 6),
        "ask_recall": round(totals["ask_correct"] / max(totals["ask_total"], 1), 6) if totals["ask_total"] else None,
        "repair_recall": round(totals["repair_correct"] / max(totals["repair_total"], 1), 6) if totals["repair_total"] else None,
        "abstain_recall": round(totals["abstain_correct"] / max(totals["abstain_total"], 1), 6) if totals["abstain_total"] else None,
        "variant_recall": round(totals["variant_correct"] / max(totals["variant_total"], 1), 6) if totals["variant_total"] else None,
        "positive_recall": round(totals["positive_correct"] / max(totals["positive_total"], 1), 6) if totals["positive_total"] else None,
        "negative_false_allow": totals["negative_false_allow"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PG-343 target-conditioned full-axis A800 smoke")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--target-weight", type=float, default=2.0)
    parser.add_argument("--context-weight", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text(encoding="utf-8-sig"))
    dataset, audit, vocabulary, rules = load(args.dataset), load(args.audit), load(args.vocabulary), load(args.rules)
    train_rows, train_failures = _load_rows(dataset, "train")
    holdout_rows, holdout_failures = _load_rows(dataset, "implementation_holdout")
    locks = {"dataset": _sha_file(args.dataset), "audit": _sha_file(args.audit), "vocabulary": _sha_file(args.vocabulary), "rules": _sha_file(args.rules), "script": _sha_file(Path(__file__)), "model": _sha_file(ROOT / "app" / "pg295_causal_moe.py")}
    import torch

    device_info = {"cuda_available": bool(torch.cuda.is_available()), "visible_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0, "current_device": int(torch.cuda.current_device()) if torch.cuda.is_available() else -1, "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    gate = evaluate_gate(dataset=dataset, audit=audit, vocabulary=vocabulary, rules=rules, env=os.environ, device=device_info, locks=locks, train_rows=train_rows, train_failures=train_failures, holdout_rows=holdout_rows, holdout_failures=holdout_failures, now=datetime.now(TZ))
    if not gate["training_allowed"]:
        raise RuntimeError("PG-343 target-conditioned gate blocked: " + ",".join(gate["failures"]))
    if not 0 < args.learning_rate <= 0.01 or not 1 <= args.epochs <= 32:
        raise ValueError("learning rate/epochs outside smoke bounds")
    max_length = max([len(row["context_tokens"]) + len(row["target_tokens"]) for row in [*train_rows, *holdout_rows]] or [2])
    vocabulary_map = {str(token): index for index, token in enumerate([PAD, UNK, *list(dict.fromkeys([*(vocabulary.get("context_tokens") or []), *(vocabulary.get("target_tokens") or [])]))])}
    config = CausalMoEConfig(d_model=128, n_layers=2, experts=2, expert_hidden=256, max_length=max_length)
    device = torch.device("cuda:0")
    target_weights = {str(token): float(args.context_weight) for token in vocabulary.get("context_tokens") or []}
    target_weights.update({str(token): float(args.target_weight) for token in vocabulary.get("target_tokens") or []})
    states: dict[str, Mapping[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        baseline = CausalMoELanguageModel(vocab_size=len(vocabulary_map), config=config).to(device)
        baseline_entropy = _predictive_entropy(baseline, holdout_rows, vocabulary_map, device)
        model = train_causal_moe(train_rows, vocabulary_map, device, seed=seed, config=config, epochs=args.epochs, learning_rate=args.learning_rate, token_weights=target_weights, normalize_weighted_loss=True)
        post_entropy = _predictive_entropy(model, holdout_rows, vocabulary_map, device)
        relative_drop = round((baseline_entropy - post_entropy) / max(abs(baseline_entropy), 1e-12), 6)
        train_eval = evaluate_causal_moe(model, train_rows, vocabulary_map, device)
        holdout_eval = evaluate_causal_moe(model, holdout_rows, vocabulary_map, device)
        train_generated = _generated_metrics(model, train_rows, vocabulary_map, device)
        holdout_generated = _generated_metrics(model, holdout_rows, vocabulary_map, device)
        candidates.append({"seed": seed, "train": train_eval, "implementation_holdout": holdout_eval, "generated_train": train_generated, "generated_holdout": holdout_generated, "baseline_holdout_predictive_entropy": baseline_entropy, "post_holdout_predictive_entropy": post_entropy, "relative_entropy_drop": relative_drop})
        states[str(seed)] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    max_drop = max(float(item["relative_entropy_drop"]) for item in candidates)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "target_conditioned_full_axis_diagnostic_candidate_only",
        "gate": gate,
        "locks": locks,
        "training": {"device": "cuda:0", "gpu": device_info.get("name"), "seeds": list(SEEDS), "epochs": args.epochs, "learning_rate": args.learning_rate, "target_weight": args.target_weight, "context_weight": args.context_weight, "context_only": False, "target_tokens_read": True, "required_max_length": max_length, "candidate_only": True},
        "target_coverage": gate["target_coverage"],
        "candidates": candidates,
        "worst_seed": {"max_relative_entropy_drop": max_drop, "entropy_gate_passed": max_drop <= 0.25, "negative_false_allow_max": max(int(item["generated_holdout"]["negative_false_allow"]) for item in candidates), "ask_recall_min": min(float(item["generated_holdout"]["ask_recall"] or 0.0) for item in candidates), "repair_recall_min": min(float(item["generated_holdout"]["repair_recall"] or 0.0) for item in candidates), "abstain_recall_min": min(float(item["generated_holdout"]["abstain_recall"] or 0.0) for item in candidates), "variant_recall_min": min(float(item["generated_holdout"]["variant_recall"] or 0.0) for item in candidates), "positive_recall_min": min(float(item["generated_holdout"]["positive_recall"] or 0.0) for item in candidates)},
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
