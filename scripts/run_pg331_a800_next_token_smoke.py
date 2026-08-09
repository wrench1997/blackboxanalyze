"""Strict PG-331 next-token smoke for the authorized weekend A800 lane.

This runner is intentionally fail-closed.  It cannot train from the current
diagnostic rows: the source-row dataset, information-preservation audit,
capacity audit, vocabulary eligibility and remote-device gate must all pass
before importing a checkpoint or touching CUDA.  The model receives only
``context_tokens`` and ``target_tokens`` from eligible abstract rows; evaluator
sidecars, route/family labels, payloads and response bodies never enter the
training stream.

The script is a candidate smoke, not a promotion path.  Every emitted report
and checkpoint is marked research-only until an independent fresh holdout
passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg293_failure_next_action import PAD, UNK  # noqa: E402
from app.pg295_causal_moe import CausalMoEConfig, evaluate_causal_moe, train_causal_moe  # noqa: E402


SCHEMA_VERSION = "pg331-a800-next-token-smoke-v1"
TIMEZONE = "Asia/Shanghai"
DATASET = ROOT / "research" / "pg331_source_row_collection_v1.json"
INFO_AUDIT = ROOT / "research" / "pg331_information_preservation_audit_v1.json"
CAPACITY_AUDIT = ROOT / "research" / "pg331_model_capacity_audit_v1.json"
VOCABULARY = ROOT / "research" / "pg331_web_token_vocabulary_v1.json"
RULES = ROOT / "research" / "improvement_rules.json"
REPORT = ROOT / "research" / "pg331_a800_next_token_smoke_report_v1.json"
CHECKPOINT = ROOT / "artifacts" / "pg331-a800-next-token-smoke" / "candidate.pt"
MAX_LENGTH = 768
SEEDS = (33101, 33102, 33103)


def _effective_max_length(capacity_audit: Mapping[str, Any] | None) -> int:
    """Choose a window that can hold the measured source rows.

    The 768-token baseline is only a lower bound.  A capacity report derived
    from a real page may require a larger window; silently keeping the
    baseline would truncate the very information PG-331 is meant to preserve.
    """

    if not isinstance(capacity_audit, Mapping):
        return MAX_LENGTH
    try:
        required = int(capacity_audit.get("required_context_window", MAX_LENGTH))
    except (TypeError, ValueError):
        required = MAX_LENGTH
    return max(MAX_LENGTH, required)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, Mapping) else None


def _is_weekend(now: datetime) -> bool:
    local = now.astimezone(ZoneInfo(TIMEZONE)) if now.tzinfo else now.replace(tzinfo=ZoneInfo(TIMEZONE))
    return local.strftime("%A") in {"Saturday", "Sunday"}


def _rules_hash_lock_valid(rules: Mapping[str, Any]) -> bool:
    try:
        contract = dict(rules["pg331_model_training_contract"])
        for path_key, hash_key in (
            ("implementation", "implementation_sha256"),
            ("test", "test_sha256"),
            ("model_implementation", "model_implementation_sha256"),
        ):
            path = ROOT / str(contract[path_key])
            if not path.is_file() or _sha256_file(path) != str(contract[hash_key]).casefold():
                return False
        return True
    except (KeyError, TypeError, OSError):
        return False


def _capacity_report_integrity(capacity_audit: Mapping[str, Any] | None, vocabulary: Mapping[str, int]) -> bool:
    if not isinstance(capacity_audit, Mapping):
        return False
    recorded = str(capacity_audit.get("audit_sha256", ""))
    if not recorded:
        return False
    unsigned = dict(capacity_audit)
    unsigned["audit_sha256"] = ""
    if _sha256_json(unsigned) != recorded:
        return False
    expected_tokens = _sha256_json(sorted(str(token) for token in vocabulary))
    return int(capacity_audit.get("model_vocabulary_size", -1)) == len(vocabulary) and str(capacity_audit.get("model_vocabulary_sha256", "")) == expected_tokens


def _eligible_rows(
    dataset: Mapping[str, Any] | None,
    vocabulary: Mapping[str, int] | None = None,
) -> tuple[list[Mapping[str, Any]], list[str]]:
    failures: list[str] = []
    if dataset is None:
        return [], ["dataset_missing_or_invalid"]
    records = dataset.get("records")
    if not isinstance(records, list) or not records:
        return [], ["dataset_has_no_records"]
    eligible: list[Mapping[str, Any]] = []
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            failures.append(f"row_{index}_not_mapping")
            continue
        if row.get("training_eligible") is not True:
            continue
        context = row.get("context_tokens")
        target = row.get("target_tokens")
        if not isinstance(context, list) or not isinstance(target, list) or not context or not target:
            failures.append(f"row_{index}_token_stream_missing")
            continue
        forbidden = ("payload", "response_body", "family=", "route_literal=", "oracle=", "evaluator=")
        if any(any(fragment in str(token).casefold() for fragment in forbidden) for token in [*context, *target]):
            failures.append(f"row_{index}_context_firewall")
            continue
        if vocabulary is not None:
            unknown_tokens = sorted({str(token) for token in [*context, *target] if str(token) not in vocabulary})
            if unknown_tokens:
                # Do not silently map a newly collected ontology field to UNK;
                # that would make a capacity/coverage failure look trainable.
                failures.append(f"row_{index}_token_not_in_append_only_vocabulary")
                continue
        eligible.append(row)
    if not eligible:
        failures.append("no_training_eligible_rows")
    return eligible, failures


def _vocabulary(manifest: Mapping[str, Any] | None) -> tuple[dict[str, int], list[str]]:
    if manifest is None:
        return {}, ["vocabulary_missing_or_invalid"]
    context = manifest.get("context_tokens")
    target = manifest.get("target_tokens")
    if not isinstance(context, list) or not isinstance(target, list) or not context or not target:
        return {}, ["vocabulary_context_or_target_missing"]
    ordered = list(dict.fromkeys([str(token) for token in [*context, *target]]))
    for special in (PAD, UNK):
        if special not in ordered:
            ordered.insert(0, special)
    return {token: index for index, token in enumerate(ordered)}, []


def evaluate_training_gate(
    *,
    now: datetime,
    env: Mapping[str, str],
    dataset: Mapping[str, Any] | None,
    information_audit: Mapping[str, Any] | None,
    capacity_audit: Mapping[str, Any] | None,
    vocabulary_manifest: Mapping[str, Any] | None,
    rules: Mapping[str, Any] | None,
    device: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate all gates without importing CUDA or starting a target."""

    effective_max_length = _effective_max_length(capacity_audit)
    vocabulary, vocab_failures = _vocabulary(vocabulary_manifest)
    rows, row_failures = _eligible_rows(dataset, vocabulary if vocabulary else None)
    expected_model_vocab = int(capacity_audit.get("model_vocabulary_size", -1)) if capacity_audit else -1
    checks: dict[str, bool] = {
        "weekend_remote_lane": _is_weekend(now),
        "explicit_training_flag": env.get("BLACKBOX_REMOTE_A800_TRAIN") == "1",
        "cuda_visible_devices_zero": env.get("CUDA_VISIBLE_DEVICES") == "0",
        "dataset_present": dataset is not None,
        "training_eligible_rows": bool(rows) and not row_failures,
        "information_preservation_passed": bool(information_audit and information_audit.get("status") == "passed"),
        "capacity_passed": bool(
            capacity_audit
            and capacity_audit.get("status") == "passed"
            and _capacity_report_integrity(capacity_audit, vocabulary)
            and int(capacity_audit.get("required_context_window", MAX_LENGTH)) <= effective_max_length
            and any(bool(item.get("capacity_pass")) and int(dict(item.get("config") or {}).get("max_length", 0)) >= effective_max_length for item in list(capacity_audit.get("variants") or []) if isinstance(item, Mapping))
        ),
        "model_vocabulary_consistent": bool(expected_model_vocab > 0 and expected_model_vocab == len(vocabulary)),
        "capacity_report_integrity": _capacity_report_integrity(capacity_audit, vocabulary),
        "vocabulary_eligible": bool(vocabulary_manifest and dict(vocabulary_manifest.get("training_eligibility") or {}).get("allowed") is True and vocabulary),
        "code_hash_lock": bool(rules and _rules_hash_lock_valid(rules)),
        "single_visible_a800_gpu0": bool(device.get("cuda_available") is True and int(device.get("visible_device_count", 0)) == 1 and int(device.get("current_device", -1)) == 0 and "A800" in str(device.get("name", ""))),
    }
    failures = [name for name, passed in checks.items() if not passed]
    failures.extend(row_failures)
    failures.extend(vocab_failures)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ready_candidate_smoke" if not failures else "blocked",
        "checks": checks,
        "failures": sorted(set(failures)),
        "training_allowed": not failures,
        "promotion": {
            "training_allowed": not failures,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
        "eligible_row_count": len(rows),
        "vocabulary_size": len(vocabulary),
        "max_length": effective_max_length,
    }


def _device_probe() -> dict[str, Any]:
    import torch

    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if available else 0
    name = torch.cuda.get_device_name(0) if available and count else ""
    current = int(torch.cuda.current_device()) if available and count else -1
    return {"cuda_available": available, "visible_device_count": count, "current_device": current, "name": name}


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed PG-331 weekend A800 next-token candidate smoke")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--information-audit", type=Path, default=INFO_AUDIT)
    parser.add_argument("--capacity-audit", type=Path, default=CAPACITY_AUDIT)
    parser.add_argument("--vocabulary", type=Path, default=VOCABULARY)
    parser.add_argument("--rules", type=Path, default=RULES)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    now = datetime.now(ZoneInfo(TIMEZONE))
    dataset = _load(args.dataset)
    info = _load(args.information_audit)
    capacity = _load(args.capacity_audit)
    vocabulary = _load(args.vocabulary)
    rules = _load(args.rules)
    remote_lane_requested = bool(
        os.environ.get("BLACKBOX_REMOTE_A800_TRAIN") == "1"
        and os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
        and _is_weekend(now)
    )
    device_probe = _device_probe() if remote_lane_requested else {"cuda_available": False, "visible_device_count": 0, "current_device": -1, "name": "not_queried"}
    gate = evaluate_training_gate(
        now=now,
        env=os.environ,
        dataset=dataset,
        information_audit=info,
        capacity_audit=capacity,
        vocabulary_manifest=vocabulary,
        rules=rules,
        device=device_probe,
    )
    if not gate["training_allowed"]:
        print(json.dumps(gate, ensure_ascii=False, indent=2 if args.json else None))
        return 2

    import torch

    vocabulary_map, _ = _vocabulary(vocabulary)
    records, _ = _eligible_rows(dataset, vocabulary_map)
    device = torch.device("cuda:0")
    config = CausalMoEConfig(d_model=128, n_heads=4, n_layers=4, experts=4, expert_hidden=512, top_k=2, dropout=0.05, max_length=int(gate["max_length"]))
    candidates: list[dict[str, Any]] = []
    selected_state: Mapping[str, torch.Tensor] | None = None
    for seed in SEEDS:
        model = train_causal_moe(records, vocabulary_map, device, seed=seed, config=config, epochs=8, learning_rate=0.001, initial_state=selected_state)
        metrics = evaluate_causal_moe(model, records, vocabulary_map, device)
        candidates.append({"seed": seed, "metrics": metrics})
        if selected_state is None:
            selected_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if selected_state is None:
        raise RuntimeError("PG-331 gate passed but no candidate checkpoint was produced")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": SCHEMA_VERSION, "config": config.__dict__, "vocabulary": vocabulary_map, "state": selected_state, "promotion_blocked": True, "gate": gate}, args.checkpoint)
    report = {"schema_version": SCHEMA_VERSION, "status": "completed_candidate_only", "gate": gate, "training": {"device": "cuda:0", "gpu": _device_probe().get("name"), "seeds": list(SEEDS), "config": config.__dict__}, "candidates": candidates, "artifacts": {"dataset": str(args.dataset), "information_audit": str(args.information_audit), "capacity_audit": str(args.capacity_audit), "vocabulary": str(args.vocabulary), "rules": str(args.rules)}, "checkpoint_sha256": _sha256_file(args.checkpoint), "promotion": gate["promotion"]}
    report["report_sha256"] = _sha256_json(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report if args.json else {"status": report["status"], "checkpoint": str(args.checkpoint), "report": str(args.report)}, ensure_ascii=False, indent=2 if args.json else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
