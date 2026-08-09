"""PG-348 abstract payload-shape candidate smoke on the authorized A800.

This runner trains the decoder on the *abstract* Rule-IR target, including the
optional ``payload_shape_ref`` slot.  It never reads evaluator sidecars,
route/family metadata, raw payloads, response bodies, or source HTML.  The
information audit is diagnostic by contract; therefore this is a research
candidate only and all promotion flags remain false.
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

SCHEMA_VERSION = "pg348-a800-payload-shape-candidate-v1"
TZ = ZoneInfo("Asia/Shanghai")
SEEDS = (34801, 34802, 34803)
FORBIDDEN = ("payload=", "payload_", "response_body=", "response_body_text=", "raw_", "oracle=", "evaluator=", "family=", "route_literal=", "implementation=", "image=", "url=", "path=", "source=")
PROMOTION = {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain an object")
    return value


def _weekend(now: datetime) -> bool:
    local = now.astimezone(TZ) if now.tzinfo else now.replace(tzinfo=TZ)
    return local.weekday() >= 5


def _vocabulary(manifest: Mapping[str, Any]) -> dict[str, int]:
    context = manifest.get("context_tokens")
    target = manifest.get("target_tokens")
    if not isinstance(context, list) or not isinstance(target, list) or not context or not target:
        raise ValueError("vocabulary context/target is missing")
    ordered = [PAD, UNK, *[str(token) for token in [*context, *target] if str(token) not in {PAD, UNK}]]
    return {token: index for index, token in enumerate(dict.fromkeys(ordered))}


def _rows(dataset: Mapping[str, Any], split: str, vocabulary: Mapping[str, int]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for index, raw in enumerate(dataset.get("records") or []):
        if not isinstance(raw, Mapping) or str(raw.get("split")) != split:
            continue
        context = raw.get("context_tokens")
        target = raw.get("target_tokens")
        if not isinstance(context, list) or not isinstance(target, list) or len(context) < 2 or not target:
            failures.append(f"row_{index}_token_stream")
            continue
        tokens = [str(token) for token in [*context, *target]]
        if any(
            any(fragment in token.casefold() for fragment in FORBIDDEN)
            for token in tokens
            if not token.casefold().startswith("payload_shape_ref=")
        ):
            failures.append(f"row_{index}_context_firewall")
            continue
        unknown = sorted({token for token in tokens if token not in vocabulary})
        if unknown:
            failures.append(f"row_{index}_unknown_token")
            continue
        firewall = raw.get("context_firewall")
        if not isinstance(firewall, Mapping) or firewall.get("forbidden_token_count") != 0 or firewall.get("sidecars_off_context") is not True:
            failures.append(f"row_{index}_firewall")
            continue
        rows.append({
            "context_tokens": [str(token) for token in context],
            "target_tokens": [str(token) for token in target],
            "safe_to_send": bool((raw.get("target_projection") or {}).get("safe_to_send", False)),
        })
    return rows, sorted(set(failures))


def _capacity_integrity(capacity: Mapping[str, Any], vocabulary: Mapping[str, int]) -> bool:
    unsigned = dict(capacity)
    recorded = str(unsigned.pop("audit_sha256", ""))
    expected_vocab = _sha_json(sorted(vocabulary))
    return bool(recorded and _sha_json({**unsigned, "audit_sha256": ""}) == recorded and int(capacity.get("model_vocabulary_size", -1)) == len(vocabulary) and str(capacity.get("model_vocabulary_sha256", "")) == expected_vocab)


def _gate(*, dataset: Mapping[str, Any], info: Mapping[str, Any], capacity: Mapping[str, Any], vocabulary_manifest: Mapping[str, Any], rules: Mapping[str, Any], train_rows: list[dict[str, Any]], train_failures: list[str], holdout_rows: list[dict[str, Any]], holdout_failures: list[str], device: Mapping[str, Any], now: datetime) -> dict[str, Any]:
    validation = dict(info.get("validation") or {})
    coverage = dict(info.get("vocabulary_coverage") or {})
    variants = [item for item in list(capacity.get("variants") or []) if isinstance(item, Mapping)]
    balanced = next((item for item in variants if dict(item.get("config") or {}).get("id") == "pg331_balanced"), {})
    vocabulary = _vocabulary(vocabulary_manifest)
    required_window = int(capacity.get("required_context_window", 0) or 0)
    max_length = max(1024, required_window)
    checks = {
        "weekend_remote_lane": _weekend(now),
        "explicit_training_flag": os.environ.get("BLACKBOX_REMOTE_A800_TRAIN") == "1",
        "cuda_visible_devices_zero": os.environ.get("CUDA_VISIBLE_DEVICES") == "0",
        "single_visible_a800_gpu0": bool(device.get("cuda_available") is True and int(device.get("visible_device_count", 0)) == 1 and int(device.get("current_device", -1)) == 0 and "A800" in str(device.get("name", ""))),
        "train_rows_valid": bool(train_rows) and not train_failures,
        "implementation_holdout_valid": bool(holdout_rows) and not holdout_failures,
        "information_diagnostic_without_failures": info.get("status") == "diagnostic" and not list(info.get("failures") or []),
        "multi_implementation_source_split": int(validation.get("implementation_count", 0)) >= 2 and int(validation.get("source_count", 0)) >= 2 and not any(validation.get("cross_split_groups", {}).values()),
        "typed_fresh_negative_replay_complete": all(int(validation.get(key, 0)) == int(validation.get("record_count", 0)) for key in ("typed_complete_count", "fresh_reset_complete_count", "negative_control_complete_count", "replay_state_complete_count")),
        "vocabulary_coverage_complete": coverage.get("context_missing_token_count") == 0 and coverage.get("target_missing_token_count") == 0,
        "capacity_balanced_1024": bool(balanced.get("capacity_pass")) and int(dict(balanced.get("config") or {}).get("max_length", 0)) >= max_length and _capacity_integrity(capacity, vocabulary),
        "rules_schema_present": isinstance(rules.get("execution_location_policy"), Mapping),
    }
    failures = [name for name, passed in checks.items() if not passed] + train_failures + holdout_failures
    return {"schema_version": SCHEMA_VERSION, "status": "ready_payload_shape_candidate" if not failures else "blocked", "checks": checks, "failures": sorted(set(failures)), "training_allowed": not failures, "max_length": max_length, "vocabulary_size": len(vocabulary), "promotion": dict(PROMOTION)}


def main() -> int:
    parser = argparse.ArgumentParser(description="PG-348 abstract payload-shape A800 candidate")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--information-audit", type=Path, required=True)
    parser.add_argument("--capacity-audit", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--target-weight", type=float, default=8.0)
    parser.add_argument("--context-weight", type=float, default=0.25)
    parser.add_argument("--negative-safe-weight", type=float, default=24.0)
    parser.add_argument("--positive-safe-weight", type=float, default=3.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 8 or not 0 < args.learning_rate <= 0.01 or not 0 < args.context_weight <= 2.0 or not 0 < args.target_weight <= 32.0 or not 0 < args.negative_safe_weight <= 64.0 or not 0 < args.positive_safe_weight <= 32.0:
        raise ValueError("candidate hyperparameters outside conservative bounds")
    dataset, info, capacity, vocabulary_manifest, rules = (_load(path) for path in (args.dataset, args.information_audit, args.capacity_audit, args.vocabulary, args.rules))
    vocabulary = _vocabulary(vocabulary_manifest)
    train_rows, train_failures = _rows(dataset, "train", vocabulary)
    holdout_rows, holdout_failures = _rows(dataset, "implementation_holdout", vocabulary)
    import torch
    device_info = {"cuda_available": bool(torch.cuda.is_available()), "visible_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0, "current_device": int(torch.cuda.current_device()) if torch.cuda.is_available() else -1, "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
    gate = _gate(dataset=dataset, info=info, capacity=capacity, vocabulary_manifest=vocabulary_manifest, rules=rules, train_rows=train_rows, train_failures=train_failures, holdout_rows=holdout_rows, holdout_failures=holdout_failures, device=device_info, now=datetime.now(TZ))
    if not gate["training_allowed"]:
        print(json.dumps(gate, ensure_ascii=False, indent=2 if args.json else None))
        return 2
    config = CausalMoEConfig(d_model=192, n_heads=4, n_layers=6, experts=4, expert_hidden=768, top_k=2, dropout=0.05, max_length=int(gate["max_length"]))
    device = torch.device("cuda:0")
    target_prefixes = ("[TARGET_", "question=", "next_action=", "repair_action=", "transport_ref=", "field_role_ref=", "encoding_ref=", "probe_variant_ref=", "payload_shape_ref=", "safe_to_send=")
    token_weights = {
        token: (
            args.negative_safe_weight if token == "safe_to_send=0" else
            args.positive_safe_weight if token == "safe_to_send=1" else
            args.target_weight if token.startswith(target_prefixes) else
            args.context_weight
        )
        for token in vocabulary
    }
    candidates: list[dict[str, Any]] = []
    selected_state: Mapping[str, torch.Tensor] | None = None
    for seed in SEEDS:
        model = train_causal_moe(train_rows, vocabulary, device, seed=seed, config=config, epochs=args.epochs, learning_rate=args.learning_rate, initial_state=selected_state, batch_size=32, token_weights=token_weights, normalize_weighted_loss=True)
        candidates.append({"seed": seed, "train": evaluate_causal_moe(model, train_rows, vocabulary, device), "implementation_holdout": evaluate_causal_moe(model, holdout_rows, vocabulary, device)})
        if selected_state is None:
            selected_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if selected_state is None:
        raise RuntimeError("candidate gate passed but no checkpoint was produced")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": SCHEMA_VERSION, "config": config.__dict__, "vocabulary": vocabulary, "state": selected_state, "promotion": dict(PROMOTION), "gate": gate}, args.checkpoint)
    report = {"schema_version": SCHEMA_VERSION, "status": "completed_payload_shape_candidate_only", "gate": gate, "training": {"device": "cuda:0", "gpu": device_info.get("name"), "seeds": list(SEEDS), "epochs": args.epochs, "learning_rate": args.learning_rate, "target_weight": args.target_weight, "context_weight": args.context_weight, "negative_safe_weight": args.negative_safe_weight, "positive_safe_weight": args.positive_safe_weight, "config": config.__dict__}, "candidates": candidates, "locks": {"dataset": _sha_file(args.dataset), "information_audit": _sha_file(args.information_audit), "capacity_audit": _sha_file(args.capacity_audit), "vocabulary": _sha_file(args.vocabulary), "rules": _sha_file(args.rules), "script": _sha_file(Path(__file__)), "model": _sha_file(ROOT / "app" / "pg295_causal_moe.py")}, "checkpoint_sha256": _sha_file(args.checkpoint), "promotion": dict(PROMOTION)}
    report["report_sha256"] = _sha_json(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report if args.json else {"status": report["status"], "checkpoint": str(args.checkpoint), "report": str(args.report)}, ensure_ascii=False, indent=2 if args.json else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
