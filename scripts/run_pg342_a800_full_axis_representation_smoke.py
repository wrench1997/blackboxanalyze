"""PG-342 context-only full-axis representation smoke on A800 GPU0.

This runner is deliberately narrower than capability training.  It reads only
the abstract ``context_tokens`` from the PG-342 diagnostic dataset.  Target
tokens, evaluator sidecars, raw payloads, raw response bodies, and source
identifiers are never read into the training loop.  The run measures whether a
decoder-only CausalMoE can retain predictive entropy and use the seven-axis
segments on an implementation holdout; it never enables model, memory,
payload-catalog, or vulnerability promotion.
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
from app.pg295_causal_moe import CausalMoEConfig, CausalMoELanguageModel, train_causal_moe  # noqa: E402
from scripts.run_pg336_a800_real_failure_representation_smoke import _metrics, _weekend  # noqa: E402

SCHEMA = "pg342-a800-full-axis-representation-smoke-v1"
TZ = ZoneInfo("Asia/Shanghai")
SEEDS = (34201, 34202, 34203)
AXES = (
    "document_structure",
    "navigation",
    "request_transport",
    "response_transport",
    "javascript_surface",
    "failure_feedback",
    "belief_and_replay",
)
FORBIDDEN_PREFIXES = (
    "family=",
    "implementation=",
    "route=",
    "route_literal=",
    "source=",
    "image=",
    "path=",
    "url=",
    "payload=",
    "payload_",
    "raw_",
    "response_body=",
    "response_body_text=",
    "oracle=",
    "evaluator=",
    "canary=",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jhash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _rows(dataset: Mapping[str, Any], split: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Extract context only; deliberately never indexes ``target_tokens``."""

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    records = dataset.get("records")
    if not isinstance(records, list):
        return [], ["dataset_missing_or_invalid"]
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping) or str(raw.get("split")) != split:
            continue
        context = raw.get("context_tokens")
        manifest = raw.get("field_capture_manifest")
        axis_presence = raw.get("axis_presence")
        firewall = raw.get("context_firewall")
        if not isinstance(context, list) or len(context) < 32:
            failures.append(f"row_{index}_context_short")
            continue
        if not isinstance(manifest, Mapping) or set(manifest) != set(AXES):
            failures.append(f"row_{index}_manifest")
            continue
        if not isinstance(axis_presence, Mapping) or any(axis_presence.get(axis) is not True for axis in AXES):
            failures.append(f"row_{index}_axis_presence")
            continue
        if not isinstance(firewall, Mapping) or firewall.get("forbidden_token_count") != 0 or firewall.get("sidecars_off_context") is not True:
            failures.append(f"row_{index}_firewall")
            continue
        if any(raw.get(key) is not False for key in ("raw_payload_stored", "raw_response_body_stored", "oracle_answer_in_context")):
            failures.append(f"row_{index}_raw_flag")
            continue
        normalized = [str(token) for token in context]
        if any(token.casefold().startswith(FORBIDDEN_PREFIXES) for token in normalized):
            failures.append(f"row_{index}_forbidden_token")
            continue
        # target_tokens is intentionally not inspected; this is a context-only lane.
        rows.append({"context_tokens": normalized, "target_tokens": []})
    return rows, sorted(set(failures))


def _implementation_isolation(dataset: Mapping[str, Any], train_split: str, holdout_split: str) -> dict[str, Any]:
    groups: dict[str, set[str]] = {train_split: set(), holdout_split: set()}
    for raw in list(dataset.get("records") or []):
        if not isinstance(raw, Mapping):
            continue
        split = str(raw.get("split"))
        if split in groups and isinstance(raw.get("source_implementation_hash"), str):
            groups[split].add(str(raw["source_implementation_hash"]))
    overlap = sorted(groups[train_split] & groups[holdout_split])
    return {
        "train_hash_count": len(groups[train_split]),
        "implementation_holdout_hash_count": len(groups[holdout_split]),
        "overlap_count": len(overlap),
        "passed": bool(groups[train_split]) and bool(groups[holdout_split]) and not overlap,
    }


def _ablate_axis(rows: list[dict[str, Any]], axis: str) -> list[dict[str, Any]]:
    begin = f"axis_begin={axis}"
    end = f"axis_end={axis}"
    ablated: list[dict[str, Any]] = []
    for row in rows:
        tokens = list(row["context_tokens"])
        try:
            start = tokens.index(begin)
            stop = tokens.index(end, start + 1)
        except ValueError:
            ablated.append({"context_tokens": tokens, "target_tokens": []})
            continue
        ablated.append({"context_tokens": [*tokens[:start], *tokens[stop + 1 :]], "target_tokens": []})
    return ablated


def _relative_drop(before: Any, after: Any) -> float | None:
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)) or float(before) <= 0:
        return None
    return round((float(before) - float(after)) / float(before), 6)


def _gate(
    *,
    dataset: Mapping[str, Any],
    audit: Mapping[str, Any],
    vocabulary: Mapping[str, Any],
    env: Mapping[str, str],
    device: Mapping[str, Any],
    locks: Mapping[str, str],
    train_rows: list[dict[str, Any]],
    train_failures: list[str],
    holdout_rows: list[dict[str, Any]],
    holdout_failures: list[str],
    now: datetime,
) -> dict[str, Any]:
    vocab_tokens = {str(token) for token in list(vocabulary.get("context_tokens") or [])}
    unknown_train = sorted({token for row in train_rows for token in row["context_tokens"]} - vocab_tokens)
    unknown_holdout = sorted({token for row in holdout_rows for token in row["context_tokens"]} - vocab_tokens)
    information_gate = dict(audit.get("information_gate") or {})
    firewall = dict(audit.get("context_firewall") or {})
    isolation = _implementation_isolation(dataset, "train", "implementation_holdout")
    checks = {
        "weekend_remote_lane": _weekend(now),
        "explicit_training_flag": env.get("BLACKBOX_REMOTE_A800_TRAIN") == "1",
        "cuda_visible_devices_zero": env.get("CUDA_VISIBLE_DEVICES") == "0",
        "single_visible_a800_gpu0": bool(
            device.get("cuda_available") is True
            and device.get("visible_device_count") == 1
            and device.get("current_device") == 0
            and "A800" in str(device.get("name", ""))
        ),
        "train_context_rows_valid": bool(train_rows) and not train_failures,
        "implementation_holdout_context_rows_valid": bool(holdout_rows) and not holdout_failures,
        "context_vocabulary_locked": bool(vocab_tokens) and not unknown_train and not unknown_holdout,
        "audit_diagnostic_only": str(audit.get("status")) == "diagnostic_only",
        "all_seven_axes_present": information_gate.get("all_axes_present") is True,
        "context_target_alignment_complete": dict(audit.get("context_target_alignment") or {}).get("rate") == 1.0,
        "audit_context_firewall_clean": firewall.get("forbidden_token_count") == 0,
        "implementation_split_isolated": isolation["passed"],
        "data_code_vocab_rules_hashes_locked": all(len(str(value)) == 64 for value in locks.values()),
    }
    failures = [key for key, value in checks.items() if not value] + train_failures + holdout_failures
    if unknown_train or unknown_holdout:
        failures.append("context_vocabulary_unknown_token")
    promotion = {
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    return {
        "schema_version": SCHEMA,
        "status": "ready_representation_pretrain_candidate" if not failures else "blocked",
        "checks": checks,
        "failures": sorted(set(failures)),
        "representation_training_allowed": not failures,
        "information_gate_status": str(audit.get("status", "missing")),
        "information_promotion_gate_passed": False,
        "train_split": "train",
        "holdout_split": "implementation_holdout",
        "split_counts": {"train": len(train_rows), "implementation_holdout": len(holdout_rows)},
        "implementation_isolation": isolation,
        "unknown_context_token_count": len(set(unknown_train + unknown_holdout)),
        "target_tokens_read": False,
        "promotion": promotion,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PG-342 full-axis context-only A800 representation candidate")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--information-audit", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not 0 < args.learning_rate <= 0.01 or not 1 <= args.epochs <= 8:
        raise ValueError("learning rate/epochs outside conservative smoke bounds")

    dataset = _load(args.dataset)
    audit = _load(args.information_audit)
    vocabulary = _load(args.vocabulary)
    locks = {
        "dataset": _sha(args.dataset),
        "information_audit": _sha(args.information_audit),
        "vocabulary": _sha(args.vocabulary),
        "rules": _sha(args.rules),
        "script": _sha(Path(__file__)),
        "model": _sha(ROOT / "app" / "pg295_causal_moe.py"),
    }
    train_rows, train_failures = _rows(dataset, "train")
    holdout_rows, holdout_failures = _rows(dataset, "implementation_holdout")

    import torch

    device_info = {
        "cuda_available": bool(torch.cuda.is_available()),
        "visible_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "current_device": int(torch.cuda.current_device()) if torch.cuda.is_available() else -1,
        "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }
    gate = _gate(
        dataset=dataset,
        audit=audit,
        vocabulary=vocabulary,
        env=os.environ,
        device=device_info,
        locks=locks,
        train_rows=train_rows,
        train_failures=train_failures,
        holdout_rows=holdout_rows,
        holdout_failures=holdout_failures,
        now=datetime.now(TZ),
    )
    if not gate["representation_training_allowed"]:
        raise RuntimeError("PG-342 representation candidate gate blocked: " + ",".join(gate["failures"]))

    vocab_tokens = [PAD, UNK] + [str(token) for token in list(vocabulary.get("context_tokens") or []) if str(token) not in {PAD, UNK}]
    vocabulary_map = {token: index for index, token in enumerate(dict.fromkeys(vocab_tokens))}
    max_length = max([len(row["context_tokens"]) for row in [*train_rows, *holdout_rows]] or [2])
    config = CausalMoEConfig(d_model=128, n_layers=2, experts=2, expert_hidden=256, max_length=max_length)
    device = torch.device("cuda:0")
    candidates: list[dict[str, Any]] = []
    for seed in SEEDS:
        torch.manual_seed(int(seed))
        baseline_model = CausalMoELanguageModel(vocab_size=len(vocabulary_map), config=config).to(device)
        baseline_model.eval()
        baseline = _metrics(baseline_model, holdout_rows, vocabulary_map, device)
        model = train_causal_moe(
            train_rows,
            vocabulary_map,
            device,
            seed=seed,
            config=config,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
        train_metrics = _metrics(model, train_rows, vocabulary_map, device)
        holdout_metrics = {
            **_metrics(model, holdout_rows, vocabulary_map, device),
            "context_row_count": len(holdout_rows),
            "split": "implementation_holdout",
        }
        ablation: dict[str, Any] = {}
        for axis in AXES:
            ablated_metrics = _metrics(model, _ablate_axis(holdout_rows, axis), vocabulary_map, device)
            entropy_full = holdout_metrics.get("mean_predictive_entropy_nats")
            entropy_ablated = ablated_metrics.get("mean_predictive_entropy_nats")
            loss_full = holdout_metrics.get("mean_next_token_loss")
            loss_ablated = ablated_metrics.get("mean_next_token_loss")
            ablation[axis] = {
                "full_entropy_nats": entropy_full,
                "ablated_entropy_nats": entropy_ablated,
                "entropy_delta_nats": round(float(entropy_ablated) - float(entropy_full), 6)
                if isinstance(entropy_ablated, (int, float)) and isinstance(entropy_full, (int, float))
                else None,
                "full_loss": loss_full,
                "ablated_loss": loss_ablated,
                "loss_delta": round(float(loss_ablated) - float(loss_full), 6)
                if isinstance(loss_ablated, (int, float)) and isinstance(loss_full, (int, float))
                else None,
            }
        candidates.append(
            {
                "seed": seed,
                "baseline_implementation_holdout": baseline,
                "train": train_metrics,
                "implementation_holdout": holdout_metrics,
                "implementation_holdout_ablation": ablation,
                "relative_entropy_drop": _relative_drop(
                    baseline.get("mean_predictive_entropy_nats"), holdout_metrics.get("mean_predictive_entropy_nats")
                ),
                "forgetting": "not_applicable_no_prior_candidate",
            }
        )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": SCHEMA,
            "representation_pretrain_candidate_only": True,
            "context_only": True,
            "target_tokens_read": False,
            "vocabulary": vocabulary_map,
            "promotion": gate["promotion"],
        },
        args.checkpoint,
    )
    report = {
        "schema_version": SCHEMA,
        "status": "representation_pretrain_candidate_only",
        "gate": gate,
        "locks": locks,
        "training": {
            "device": "cuda:0",
            "seeds": list(SEEDS),
            "context_only": True,
            "target_tokens_read": False,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
        },
        "context_capacity_requirement": {
            "train_max": max((len(row["context_tokens"]) for row in train_rows), default=0),
            "implementation_holdout_max": max((len(row["context_tokens"]) for row in holdout_rows), default=0),
            "required_max_length": max_length,
        },
        "loss": candidates,
        "entropy_gate": {
            "relative_drop_limit": 0.25,
            "max_relative_entropy_drop": max(
                [float(item["relative_entropy_drop"]) for item in candidates if isinstance(item.get("relative_entropy_drop"), (int, float))]
                or [0.0]
            ),
            "passed": all(
                float(item.get("relative_entropy_drop", 1.0)) <= 0.25
                for item in candidates
                if isinstance(item.get("relative_entropy_drop"), (int, float))
            )
            and all(item.get("relative_entropy_drop") is not None for item in candidates),
            "interpretation": "same-seed baseline versus post-training predictive entropy; diagnostic only, not capability promotion",
        },
        "forgetting": "not_applicable_no_prior_candidate",
        "promotion": gate["promotion"],
    }
    report["report_sha256"] = _jhash(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report if args.json else {"status": report["status"], "train_rows": len(train_rows), "implementation_holdout_rows": len(holdout_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
