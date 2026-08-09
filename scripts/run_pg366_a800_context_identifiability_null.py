"""PG-366 A800 null control for context-to-Rule-IR shortcut detection.

This is a diagnostic, not a capability run.  It keeps every abstract context
unchanged, permutes only the *training labels in memory*, and evaluates on the
unaltered implementation holdout.  If a model still scores well, the split or
evaluator is leaking.  A normal model is not promoted from this run; all
payload/memory/vulnerability flags remain false.

The runner never starts a target, sends a request, reads evaluator sidecars or
stores raw payload/response data.  It is deliberately restricted to the
authorized remote A800 GPU0 weekend lane.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from app.pg293_failure_next_action import PAD, UNK
from app.pg295_causal_moe import CausalMoEConfig
from app.pg363_pooled_rule_ir import (
    PooledRuleIRDecoder,
    PooledSlotConfig,
    build_slot_candidates,
    evaluate_pooled_rule_ir,
    train_pooled_rule_ir,
)
from scripts.audit_pg366_context_identifiability import audit_document
from scripts.run_pg363_a800_pooled_rule_ir_candidate import _predictive_entropy

SCHEMA_VERSION = "pg366-context-identifiability-null-v1"
SEEDS = (36601, 36602, 36603)


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _vocabulary_map(vocabulary: Mapping[str, Any]) -> dict[str, int]:
    tokens = [PAD, UNK, *(vocabulary.get("context_tokens") or []), *(vocabulary.get("target_tokens") or [])]
    ordered = list(dict.fromkeys(str(token) for token in tokens))
    return {token: index for index, token in enumerate(ordered)}


def permute_training_targets(records: Sequence[Mapping[str, Any]], *, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return copies with target streams permuted but contexts byte-for-byte intact."""
    if not records:
        raise ValueError("cannot permute empty training set")
    targets = [copy.deepcopy(row.get("target_tokens")) for row in records]
    if any(not isinstance(target, list) or not target for target in targets):
        raise ValueError("training target missing")
    order = list(range(len(targets)))
    rng = random.Random(int(seed))
    for _ in range(32):
        rng.shuffle(order)
        identity = sum(targets[index] == targets[order[index]] for index in range(len(order)))
        if identity <= max(1, len(order) // 100):
            break
    output: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        clone = dict(row)
        clone["target_tokens"] = copy.deepcopy(targets[order[index]])
        output.append(clone)
    return output, {
        "row_count": len(output),
        "permutation_seed": int(seed),
        "identity_target_rows": sum(targets[index] == targets[order[index]] for index in range(len(order))),
        "target_multiset_preserved": sorted(map(str, targets)) == sorted(map(str, [row["target_tokens"] for row in output])),
        "contexts_unchanged": all(row.get("context_tokens") == records[index].get("context_tokens") for index, row in enumerate(output)),
    }


def _require_remote_a800() -> tuple[torch.device, str]:
    if str(__import__("os").environ.get("BLACKBOX_REMOTE_A800_TRAIN", "")) != "1":
        raise RuntimeError("explicit_remote_training_flag_missing")
    if str(__import__("os").environ.get("CUDA_VISIBLE_DEVICES", "")) != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES_must_be_0")
    now = datetime.now().astimezone()
    if now.weekday() not in (5, 6):
        raise RuntimeError("weekend_remote_lane_required")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly_one_visible_cuda_device_required")
    torch.cuda.set_device(0)
    name = torch.cuda.get_device_name(0)
    if "A800" not in name:
        raise RuntimeError("visible_device_is_not_A800")
    return torch.device("cuda:0"), name


def run(args: argparse.Namespace) -> dict[str, Any]:
    device, gpu_name = _require_remote_a800()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
    audit = json.loads(args.audit.read_text(encoding="utf-8-sig"))
    dataset_sha = _sha_file(args.dataset)
    audit_sha = _sha_file(args.audit)
    rules_sha = _sha_file(args.rules)
    source_rows = dataset.get("records")
    if not isinstance(source_rows, list) or not source_rows:
        raise RuntimeError("dataset_records_missing")
    if audit.get("status") not in {"diagnostic_shortcut_risk", "diagnostic"}:
        raise RuntimeError("identifiability_audit_status_unexpected")
    checked = audit_document(dataset, source_path=str(args.dataset), source_sha256=dataset_sha)
    if checked["counts"]["invalid_rows"]:
        raise RuntimeError("dataset_has_invalid_rows")
    train_rows = [row for row in source_rows if row.get("split") == "train"]
    holdout_rows = [row for row in source_rows if row.get("split") == "implementation_holdout"]
    if not train_rows or not holdout_rows:
        raise RuntimeError("train_holdout_split_missing")
    vocabulary = _vocabulary_map(dataset.get("vocabulary") or {})
    config = CausalMoEConfig(
        d_model=int(args.d_model), n_heads=int(args.n_heads), n_layers=int(args.n_layers),
        experts=int(args.experts), expert_hidden=int(args.expert_hidden), max_length=int(args.max_length),
    )
    slot_config = PooledSlotConfig(language_model_weight=float(args.lm_weight), slot_weight=1.0, label_smoothing=float(args.label_smoothing))
    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        null_rows, permutation = permute_training_targets(train_rows, seed=seed + 100000)
        baseline = PooledRuleIRDecoder(vocab_size=len(vocabulary), config=config, slot_candidates=build_slot_candidates(vocabulary)).to(device)
        baseline_entropy = _predictive_entropy(baseline, holdout_rows, vocabulary, device, batch_size=int(args.batch_size))
        model = train_pooled_rule_ir(null_rows, vocabulary, device, seed=seed, config=config, slot_config=slot_config, epochs=int(args.epochs), learning_rate=float(args.learning_rate), batch_size=int(args.batch_size))
        post_entropy = _predictive_entropy(model, holdout_rows, vocabulary, device, batch_size=int(args.batch_size))
        null_train_eval = evaluate_pooled_rule_ir(model, null_rows, vocabulary, device, batch_size=int(args.batch_size))
        untouched_holdout_eval = evaluate_pooled_rule_ir(model, holdout_rows, vocabulary, device, batch_size=int(args.batch_size))
        per_seed.append({
            "seed": seed,
            "permutation": permutation,
            "null_train": null_train_eval,
            "untouched_holdout": untouched_holdout_eval,
            "baseline_holdout_predictive_entropy": round(float(baseline_entropy), 6),
            "post_holdout_predictive_entropy": round(float(post_entropy), 6),
            "relative_entropy_drop": round(max(0.0, (float(baseline_entropy) - float(post_entropy)) / max(float(baseline_entropy), 1e-12)), 6),
        })
        del model, baseline
        torch.cuda.empty_cache()
    worst = {
        "holdout_sequence_exact_max": max(float(item["untouched_holdout"]["sequence_exact_accuracy"]) for item in per_seed),
        "holdout_sequence_exact_min": min(float(item["untouched_holdout"]["sequence_exact_accuracy"]) for item in per_seed),
        "holdout_negative_false_allow_max": max(int(item["untouched_holdout"]["negative_false_allow"]) for item in per_seed),
        "max_relative_entropy_drop": max(float(item["relative_entropy_drop"]) for item in per_seed),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_null_control_only",
        "purpose": "test whether normal holdout performance survives when training targets are permuted in memory",
        "dataset": str(args.dataset.resolve().relative_to(ROOT.resolve())),
        "dataset_sha256": dataset_sha,
        "audit": str(args.audit.resolve().relative_to(ROOT.resolve())),
        "audit_sha256": audit_sha,
        "rules_sha256": rules_sha,
        "training": {"device": str(device), "gpu": gpu_name, "cuda_visible_devices": "0", "seeds": list(SEEDS), "epochs": int(args.epochs), "batch_size": int(args.batch_size), "target_tokens_used_as_permuted_labels_only": True, "contexts_unchanged": True, "holdout_targets_unmodified": True},
        "counts": {"train_rows": len(train_rows), "implementation_holdout_rows": len(holdout_rows)},
        "per_seed": per_seed,
        "worst": worst,
        "interpretation": "null control is diagnostic only; good holdout performance would indicate a split/evaluator leak, while poor holdout performance confirms the normal result depends on context-target correlation",
        "raw_payload_in_context": False,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    report["report_sha256"] = _sha_json(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": SCHEMA_VERSION, "dataset_sha256": dataset_sha, "rules_sha256": rules_sha, "promotion": report["promotion"]}, args.checkpoint)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="PG-366 A800 context identifiability null control")
    parser.add_argument("--dataset", type=Path, default=ROOT / "research" / "pg364_compositional_rule_ir_dataset_v1.json")
    parser.add_argument("--audit", type=Path, default=ROOT / "research" / "pg366_context_identifiability_audit_v1.json")
    parser.add_argument("--rules", type=Path, default=ROOT / "research" / "improvement_rules.json")
    parser.add_argument("--report", type=Path, default=ROOT / "research" / "pg366_a800_context_identifiability_null_v1.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts" / "pg366-context-identifiability-null" / "pg366_a800_context_identifiability_null_v1.pt")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lm-weight", type=float, default=0.15)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=621)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--experts", type=int, default=2)
    parser.add_argument("--expert-hidden", type=int, default=256)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"status": report["status"], "worst": report["worst"], "report_sha256": report["report_sha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
