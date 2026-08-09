"""PG-341 target-conditioned diagnostic smoke on the authorized A800 lane.

This runner trains only the explicitly separated ``coarse_process`` view. It
is the smallest honest experiment that checks whether the decoder can emit
abstract ASK/repair/negative decisions after seeing process tokens. The
full-axis view is loaded only by the audit and is never used as a training
shortcut; its missing ASK/repair training coverage remains a hard blocker for
the unified page model.

No raw payload, response body, route, family, evaluator answer or sidecar is
read by the model.  All reports/checkpoints are diagnostic candidates with
promotion disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg293_failure_next_action import PAD, UNK  # noqa: E402
from app.pg295_causal_moe import CausalMoEConfig, evaluate_causal_moe, train_causal_moe  # noqa: E402


SCHEMA_VERSION = "pg341-a800-target-conditioned-coarse-diagnostic-v1"
TZ = ZoneInfo("Asia/Shanghai")
SEEDS = (34101, 34102, 34103)
FORBIDDEN = ("payload=", "payload_", "response_body=", "response_body_text=", "raw_", "oracle=", "evaluator=", "family=", "route=", "route_literal=", "implementation=", "image=", "url=", "path=", "source=")
TARGET_KEYS = ("question=", "next_action=", "repair_action=", "action_changed=", "failure_class=", "safe_to_send=", "transport_ref=", "field_role_ref=", "encoding_ref=", "probe_variant_ref=")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _weekend(now: datetime) -> bool:
    local = now.astimezone(TZ) if now.tzinfo else now.replace(tzinfo=TZ)
    return local.weekday() >= 5


def _rows(dataset: Mapping[str, Any], split: str) -> tuple[list[dict[str, Any]], list[str]]:
    result: list[dict[str, Any]] = []
    failures: list[str] = []
    for index, raw in enumerate(dataset.get("records") or []):
        if not isinstance(raw, Mapping) or raw.get("view") != "coarse_process" or raw.get("split") != split:
            continue
        context = raw.get("context_tokens")
        target = raw.get("target_tokens")
        firewall = raw.get("context_firewall")
        if raw.get("target_conditioned_diagnostic_eligible") is not True:
            failures.append(f"row_{index}_diagnostic_flag")
            continue
        if not isinstance(context, list) or not isinstance(target, list) or len(context) < 2 or target[:1] != ["[TARGET_BOS]"] or target[-1:] != ["[TARGET_EOS]"]:
            failures.append(f"row_{index}_token_stream")
            continue
        if not isinstance(firewall, Mapping) or firewall.get("forbidden_token_count") != 0 or firewall.get("sidecars_off_context") is not True:
            failures.append(f"row_{index}_firewall")
            continue
        if any(any(fragment in str(token).casefold() for fragment in FORBIDDEN) for token in [*context, *target]):
            failures.append(f"row_{index}_forbidden_token")
            continue
        if any(not str(token).startswith(TARGET_KEYS) and str(token) not in {"[TARGET_BOS]", "[TARGET_EOS]"} for token in target):
            failures.append(f"row_{index}_target_not_abstract")
            continue
        # evaluate_causal_moe expects this abstract flag; it is derived only
        # from the target token and is not an evaluator sidecar.
        item = {"context_tokens": [str(token) for token in context], "target_tokens": [str(token) for token in target], "safe_to_send": "safe_to_send=1" in target}
        result.append(item)
    return result, sorted(set(failures))


def _vocabulary(manifest: Mapping[str, Any]) -> tuple[dict[str, int], list[str]]:
    context = [str(token) for token in manifest.get("context_tokens") or []]
    target = [str(token) for token in manifest.get("target_tokens") or []]
    if not context or not target:
        return {}, ["vocabulary_context_or_target_missing"]
    ordered = list(dict.fromkeys([PAD, UNK, *context, *target]))
    return {token: index for index, token in enumerate(ordered)}, []


def _target_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    questions = {str(token).split("=", 1)[1] for row in rows for token in row["target_tokens"] if str(token).startswith("question=")}
    actions = {str(token).split("=", 1)[1] for row in rows for token in row["target_tokens"] if str(token).startswith("next_action=")}
    return {"rows": len(rows), "questions": sorted(questions), "next_actions": sorted(actions), "ask_present": any(value != "none" for value in questions), "repair_present": bool(actions & {"repair", "repair_abstract_plan"}), "abstain_present": "abstain" in actions}


def evaluate_gate(*, dataset: Mapping[str, Any], audit: Mapping[str, Any], vocabulary: Mapping[str, Any], rules: Mapping[str, Any], env: Mapping[str, str], device: Mapping[str, Any], locks: Mapping[str, str], train_rows: list[dict[str, Any]], train_failures: list[str], holdout_rows: list[dict[str, Any]], holdout_failures: list[str], now: datetime) -> dict[str, Any]:
    vocab_tokens = {str(token) for token in [*(vocabulary.get("context_tokens") or []), *(vocabulary.get("target_tokens") or [])]}
    unknown = sorted({str(token) for row in [*train_rows, *holdout_rows] for token in [*row["context_tokens"], *row["target_tokens"]]} - vocab_tokens)
    coarse_report = dict(audit.get("coarse_process") or {})
    checks = {
        "weekend_remote_lane": _weekend(now),
        "explicit_training_flag": env.get("BLACKBOX_REMOTE_A800_TRAIN") == "1",
        "cuda_visible_devices_zero": env.get("CUDA_VISIBLE_DEVICES") == "0",
        "single_visible_a800_gpu0": bool(device.get("cuda_available") is True and device.get("visible_device_count") == 1 and device.get("current_device") == 0 and "A800" in str(device.get("name", ""))),
        "coarse_train_rows_valid": bool(train_rows) and not train_failures,
        "coarse_holdout_rows_valid": bool(holdout_rows) and not holdout_failures,
        "coarse_target_diagnostic_allowed": coarse_report.get("diagnostic_training_allowed") is True,
        "coarse_target_coverage": all((_target_coverage(train_rows)[key] and _target_coverage(holdout_rows)[key]) for key in ("ask_present", "repair_present", "abstain_present")),
        "full_axis_gap_not_bypassed": str(audit.get("status")) == "blocked_full_axis_target_gap" and dict(audit.get("full_axis") or {}).get("target_training_allowed") is False,
        "context_vocabulary_locked": bool(vocab_tokens) and not unknown,
        "promotion_closed": all(value is False for value in dict(audit.get("promotion") or {}).values()),
        "data_code_vocab_rules_hashes_locked": all(len(str(value)) == 64 for value in locks.values()),
        "rules_schema_present": isinstance(rules, Mapping),
    }
    failures = [key for key, passed in checks.items() if not passed] + train_failures + holdout_failures
    if unknown:
        failures.append("unknown_context_or_target_token")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ready_target_conditioned_diagnostic" if not failures else "blocked",
        "checks": checks,
        "failures": sorted(set(failures)),
        "training_allowed": not failures,
        "track": "coarse_process_only",
        "full_axis_training_allowed": False,
        "target_tokens_read": True,
        "split_counts": {"train": len(train_rows), "implementation_holdout": len(holdout_rows)},
        "target_coverage": {"train": _target_coverage(train_rows), "implementation_holdout": _target_coverage(holdout_rows)},
        "unknown_token_count": len(unknown),
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PG-341 coarse target-conditioned A800 diagnostic")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--target-weight", type=float, default=2.0)
    parser.add_argument("--context-weight", type=float, default=1.0)
    parser.add_argument("--positive-weight", type=float, default=0.0, help="extra weight for abstract safe_to_send=1/assemble targets; zero keeps the baseline")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text(encoding="utf-8-sig"))
    dataset, audit, vocabulary, rules = load(args.dataset), load(args.audit), load(args.vocabulary), load(args.rules)
    train_rows, train_failures = _rows(dataset, "train")
    holdout_rows, holdout_failures = _rows(dataset, "implementation_holdout")
    locks = {"dataset": _sha(args.dataset), "audit": _sha(args.audit), "vocabulary": _sha(args.vocabulary), "rules": _sha(args.rules), "script": _sha(Path(__file__)), "model": _sha(ROOT / "app" / "pg295_causal_moe.py")}
    import torch

    device_info = {"cuda_available": bool(torch.cuda.is_available()), "visible_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0, "current_device": int(torch.cuda.current_device()) if torch.cuda.is_available() else -1, "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    gate = evaluate_gate(dataset=dataset, audit=audit, vocabulary=vocabulary, rules=rules, env=os.environ, device=device_info, locks=locks, train_rows=train_rows, train_failures=train_failures, holdout_rows=holdout_rows, holdout_failures=holdout_failures, now=datetime.now(TZ))
    if not gate["training_allowed"]:
        raise RuntimeError("PG-341 target-conditioned gate blocked: " + ",".join(gate["failures"]))
    if not 0 < args.learning_rate <= 0.01 or not 1 <= args.epochs <= 32:
        raise ValueError("learning rate/epochs outside conservative smoke bounds")
    if not 0 < args.target_weight <= 32 or not 0 < args.context_weight <= 8 or not 0 <= args.positive_weight <= 64:
        raise ValueError("target/context weights outside conservative diagnostic bounds")
    vocabulary_map, vocabulary_failures = _vocabulary(vocabulary)
    if vocabulary_failures:
        raise RuntimeError("PG-341 vocabulary blocked: " + ",".join(vocabulary_failures))
    unknown = sorted({str(token) for row in [*train_rows, *holdout_rows] for token in [*row["context_tokens"], *row["target_tokens"]]} - set(vocabulary_map))
    if unknown:
        raise RuntimeError("PG-341 append-only vocabulary missing tokens")
    max_length = max([len(row["context_tokens"]) + len(row["target_tokens"]) for row in [*train_rows, *holdout_rows]] or [2])
    config = CausalMoEConfig(d_model=128, n_layers=2, experts=2, expert_hidden=256, max_length=max_length)
    device = torch.device("cuda:0")
    target_weights = {str(token): float(args.context_weight) for token in vocabulary.get("context_tokens") or []}
    target_weights.update({str(token): float(args.target_weight) for token in vocabulary.get("target_tokens") or []})
    if args.positive_weight:
        target_weights["safe_to_send=1"] = float(args.positive_weight)
        target_weights["next_action=assemble_rule_ir"] = float(args.positive_weight)
    candidates: list[dict[str, Any]] = []
    states: dict[str, Mapping[str, Any]] = {}
    for seed in SEEDS:
        model = train_causal_moe(train_rows, vocabulary_map, device, seed=seed, config=config, epochs=args.epochs, learning_rate=args.learning_rate, token_weights=target_weights)
        candidates.append({"seed": seed, "train": evaluate_causal_moe(model, train_rows, vocabulary_map, device), "implementation_holdout": evaluate_causal_moe(model, holdout_rows, vocabulary_map, device), "target_weight": args.target_weight, "context_weight": args.context_weight, "positive_weight": args.positive_weight, "forgetting": "not_applicable_no_prior_target_conditioned_candidate"})
        states[str(seed)] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": SCHEMA_VERSION, "track": "coarse_process_only", "config": config.__dict__, "vocabulary": vocabulary_map, "states": states, "promotion": gate["promotion"], "full_axis_training_allowed": False}, args.checkpoint)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "target_conditioned_diagnostic_candidate_only",
        "gate": gate,
        "locks": locks,
        "training": {"device": "cuda:0", "gpu": device_info.get("name"), "seeds": list(SEEDS), "context_only": False, "target_tokens_read": True, "track": "coarse_process_only", "full_axis_training_allowed": False, "learning_rate": args.learning_rate, "epochs": args.epochs, "target_weight": args.target_weight, "context_weight": args.context_weight, "positive_weight": args.positive_weight, "checkpoint_contains_all_seed_states": True},
        "context_capacity_requirement": {"train_max": max((len(row["context_tokens"]) + len(row["target_tokens"]) for row in train_rows), default=0), "implementation_holdout_max": max((len(row["context_tokens"]) + len(row["target_tokens"]) for row in holdout_rows), default=0), "required_max_length": max_length},
        "target_coverage": gate["target_coverage"],
        "candidates": candidates,
        "full_axis_target_gap": {"status": "blocked", "reason": "full_axis_train_has_no_ask_repair_negative_targets", "training_allowed": False},
        "promotion": gate["promotion"],
    }
    report["report_sha256"] = _json_hash(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report if args.json else {"status": report["status"], "train_rows": len(train_rows), "holdout_rows": len(holdout_rows)}, ensure_ascii=False, indent=2 if args.json else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
