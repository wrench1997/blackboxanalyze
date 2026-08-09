"""Run a high-capacity abstract Rule-IR composition candidate.

PG-380 showed that a shared backbone plus independent slot heads can learn
ASK/repair/negative safety while failing to compose the complete 13-slot
Rule-IR sequence.  PG-381 reuses the reviewed autoregressive composition
decoder from PG-375 on the isolated PG-380 abstract matrix.

This runner never loads raw payloads, response bodies, URLs, evaluator
answers, or wire data.  It has no Docker/network path.  A CUDA run is a
candidate-only abstract reasoning experiment and requires the explicit
weekend/A800/GPU0 gate; all promotion and capability flags stay false.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg295_causal_moe import CausalMoEConfig  # noqa: E402
from scripts.run_pg370_multitask_moe_candidate import (  # noqa: E402
    _declared_slot_values,
    _slot_classes_from_values,
    build_declared_vocabulary,
)
from scripts.run_pg375_composed_rule_ir_candidate import (  # noqa: E402
    DEFAULT_WEIGHTS,
    SLOTS,
    _device_gate,
    _run_seed,
)
from scripts.run_pg380_abstract_reasoning_sft import (  # noqa: E402
    DEFAULT_DATASET,
    PROMOTION,
    _load,
    _safe_rows,
    _sha_file,
    _sha_json,
)

SCHEMA_VERSION = "pg381-abstract-composition-candidate-v1"
SEEDS = (38101, 38102, 38103)


def _declared_manifest_compatible(dataset: Mapping[str, Any]) -> tuple[list[str], dict[str, list[str]]]:
    """Read PG-380 or PG-382 declared abstract ontology coordinates."""

    vocabulary = dataset.get("vocabulary")
    if not isinstance(vocabulary, Mapping) or vocabulary.get("scope") not in {
        "declared_abstract_ontology",
        "declared_abstract_factorized_ontology",
        "declared_abstract_factorized_binding_ontology",
    }:
        raise ValueError("declared abstract ontology vocabulary is required")
    context_tokens = vocabulary.get("context_tokens")
    target_tokens = vocabulary.get("target_tokens")
    if not isinstance(context_tokens, list) or not isinstance(target_tokens, list):
        raise ValueError("declared abstract ontology vocabulary is incomplete")
    declared = [str(token) for token in [*context_tokens, *target_tokens]]
    slot_values: dict[str, set[str]] = {key: set() for key in SLOTS}
    for token in target_tokens:
        text = str(token)
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        if key in slot_values and value:
            slot_values[key].add(value)
    if any(not values for values in slot_values.values()):
        raise ValueError("declared target slot inventory is incomplete")
    return declared, {key: sorted(values) for key, values in slot_values.items()}


def _execution_gate(device: str) -> dict[str, Any]:
    now = dt.datetime.now(ZoneInfo("Asia/Shanghai"))
    weekend = now.weekday() >= 5
    explicit = os.environ.get("BLACKBOX_REMOTE_A800_TRAIN") == "1"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return {
        "timestamp": now.isoformat(),
        "weekend": weekend,
        "allowed_time_window": weekend or (8 <= now.hour < 18),
        "explicit_remote_flag": explicit,
        "cuda_visible_devices": visible,
        "device": device,
        "passed": device == "cpu" or (weekend and explicit and visible == "0"),
    }


def _closed_rows(dataset: Mapping[str, Any], row_limit: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train, holdout = _safe_rows(dict(dataset))
    if row_limit is not None:
        limit = max(1, int(row_limit))
        train, holdout = train[:limit], holdout[:limit]
    declared, declared_slot_values = _declared_manifest_compatible(dataset)
    vocabulary = build_declared_vocabulary(declared)
    slot_classes = _slot_classes_from_values(declared_slot_values)
    unknown_tokens = sorted(
        {
            str(token)
            for row in [*train, *holdout]
            for token in [*row["context_tokens"], *row["target_tokens"]]
            if str(token) not in vocabulary
        }
    )
    unknown_slots: dict[str, list[str]] = {}
    for key in SLOTS:
        values = sorted(
            {
                str(row["target_tokens"][1 + list(SLOTS).index(key)]).split("=", 1)[1]
                for row in holdout
                if str(row["target_tokens"][1 + list(SLOTS).index(key)]).split("=", 1)[1] not in slot_classes[key]
            }
        )
        if values:
            unknown_slots[key] = values
    if unknown_tokens or unknown_slots:
        raise ValueError("PG-381 declared abstract vocabulary is not closed")
    # The loader intentionally returns only context/target/split.  Attach the
    # slot classes as a private training sidecar, never in the model batch.
    return train, holdout


def _worst(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "sequence_exact_min": min(float(item["post"]["sequence_exact"]) for item in results),
        "slot_composition_exact_min": min(float(item["post"]["slot_composition_exact"]) for item in results),
        "slot_accuracy_min": min(float(item["post"]["slot_accuracy"]) for item in results),
        "ask_recall_min": min(float(item["post"]["ask_recall"] or 0.0) for item in results),
        "repair_recall_min": min(float(item["post"]["repair_recall"] or 0.0) for item in results),
        "positive_recall_min": min(float(item["post"]["positive_recall"] or 0.0) for item in results),
        "negative_false_allow_max": max(int(item["post"]["negative_false_allow"]) for item in results),
        "entropy_relative_drop_max": max(float(item["entropy_relative_drop"]) for item in results),
    }


def run_candidate_report(
    *,
    dataset_path: Path = DEFAULT_DATASET,
    device: str = "cpu",
    pretrain_epochs: int = 1,
    posttrain_epochs: int = 1,
    microbatch: int = 4,
    grad_accum: int = 1,
    d_model: int = 32,
    n_layers: int = 1,
    experts: int = 2,
    expert_hidden: int = 64,
    slot_decoder_layers: int = 1,
    slot_decoder_heads: int = 2,
    max_length: int = 128,
    lr_pretrain: float = 1e-4,
    lr_posttrain: float = 2.5e-5,
    kl_weight: float = 0.05,
    row_limit: int | None = None,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    dataset = _load(dataset_path)
    train, holdout = _closed_rows(dataset, row_limit=row_limit)
    declared, declared_slot_values = _declared_manifest_compatible(dataset)
    vocabulary = build_declared_vocabulary(declared)
    slot_classes = _slot_classes_from_values(declared_slot_values)
    gate = _execution_gate(device)
    if device != "cpu" and not gate["passed"]:
        raise RuntimeError("PG-381 remote A800 gate failed")
    effective = CausalMoEConfig(
        d_model=int(d_model),
        n_heads=int(slot_decoder_heads),
        n_layers=int(n_layers),
        experts=int(experts),
        expert_hidden=int(expert_hidden),
        max_length=int(max_length),
        top_k=min(2, int(experts)),
    )
    required_window = max((len(row["context_tokens"]) + len(row["target_tokens"]) for row in [*train, *holdout]), default=1)
    if int(max_length) < required_window:
        raise ValueError(f"PG-381 max_length={max_length} below required window {required_window}")
    weights = dict(DEFAULT_WEIGHTS)
    weights.update({"next_token": 0.4, "composition": 2.0, "slot_aux": 0.25, "ask": 1.0, "repair": 1.5, "negative": 2.0, "balance": 0.01})
    torch_device = _device_gate(device)
    results = [
        _run_seed(
            train,
            holdout,
            vocabulary,
            slot_classes,
            seed=int(seed),
            config=effective,
            slot_decoder_layers=int(slot_decoder_layers),
            slot_decoder_heads=int(slot_decoder_heads),
            pretrain_epochs=int(pretrain_epochs),
            posttrain_epochs=int(posttrain_epochs),
            microbatch=int(microbatch),
            grad_accum=int(grad_accum),
            lr_pretrain=float(lr_pretrain),
            lr_posttrain=float(lr_posttrain),
            kl_weight=float(kl_weight),
            weights=weights,
            device=torch_device,
            checkpoint_dir=checkpoint_dir,
        )
        for seed in SEEDS
    ]
    worst = _worst(results)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "abstract_composition_candidate_only" if device != "cpu" else "cpu_smoke_candidate_only",
        "dataset": str(dataset_path),
        "dataset_sha256": _sha_file(dataset_path),
        "dataset_internal_sha256": dataset.get("dataset_sha256"),
        "data": {
            "train_rows": len(train),
            "implementation_holdout_rows": len(holdout),
            "vocabulary_scope": "declared_abstract_ontology",
            "vocabulary_size": len(vocabulary),
            "target_slots": len(SLOTS),
            "required_context_window": required_window,
        },
        "execution_gate": gate,
        "training": {
            "device": device,
            "seeds": list(SEEDS),
            "pretrain_epochs": int(pretrain_epochs),
            "posttrain_epochs": int(posttrain_epochs),
            "microbatch": int(microbatch),
            "grad_accum": int(grad_accum),
            "config": dict(effective.__dict__),
            "slot_decoder_layers": int(slot_decoder_layers),
            "slot_decoder_heads": int(slot_decoder_heads),
            "lr_pretrain": float(lr_pretrain),
            "lr_posttrain": float(lr_posttrain),
            "kl_weight": float(kl_weight),
            "loss_weights": weights,
            "target_tokens_read_for_evaluator": False,
            "raw_rows_loaded": False,
        },
        "candidates": results,
        "worst_seed": worst,
        "scientific_gate": {
            "abstract_reasoning_only": True,
            "typed_live_replay_with_model_selected_wire": False,
            "model_selected_wire_replay": False,
            "claim_allowed": False,
            "later_layer_entropy": "diagnostic_only",
        },
        "safety": {
            "raw_payload_in_context": False,
            "raw_response_in_context": False,
            "evaluator_answer_in_context": False,
            "concrete_wire": "evaluator_template_only",
            "external_network": False,
            "persistent_state_write": False,
        },
        "promotion": dict(PROMOTION),
        "interpretation": "PG-381 只验证 13-slot Rule-IR 抽象组合、ASK/repair/negative；不生成任意原始攻击字节，不代表漏洞能力。",
    }
    report["report_sha256"] = _sha_json(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=ROOT / "research/pg381_abstract_composition_candidate_v1.json")
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--remote-candidate", action="store_true")
    parser.add_argument("--pretrain-epochs", type=int, default=2)
    parser.add_argument("--posttrain-epochs", type=int, default=4)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--expert-hidden", type=int, default=2048)
    parser.add_argument("--slot-decoder-layers", type=int, default=2)
    parser.add_argument("--slot-decoder-heads", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr-pretrain", type=float, default=1e-4)
    parser.add_argument("--lr-posttrain", type=float, default=2.5e-5)
    parser.add_argument("--kl-weight", type=float, default=0.05)
    parser.add_argument("--row-limit", type=int)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "artifacts/pg381-abstract-composition-a800")
    args = parser.parse_args()
    if args.cpu_smoke and args.remote_candidate:
        parser.error("--cpu-smoke and --remote-candidate are mutually exclusive")
    if not args.cpu_smoke and not args.remote_candidate:
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "plan_only",
            "dataset": str(args.dataset),
            "dataset_sha256": _sha_file(args.dataset),
            "execution": {"optimizer_started": False, "gpu_touched": False, "docker_started": False, "network_used": False},
            "promotion": dict(PROMOTION),
        }
    else:
        remote = bool(args.remote_candidate)
        report = run_candidate_report(
            dataset_path=args.dataset,
            device="cuda:0" if remote else "cpu",
            pretrain_epochs=min(int(args.pretrain_epochs), 1) if args.cpu_smoke else int(args.pretrain_epochs),
            posttrain_epochs=min(int(args.posttrain_epochs), 1) if args.cpu_smoke else int(args.posttrain_epochs),
            microbatch=min(int(args.microbatch), 2) if args.cpu_smoke else int(args.microbatch),
            grad_accum=1 if args.cpu_smoke else int(args.grad_accum),
            d_model=min(int(args.d_model), 32) if args.cpu_smoke else int(args.d_model),
            n_layers=min(int(args.n_layers), 1) if args.cpu_smoke else int(args.n_layers),
            experts=min(int(args.experts), 2) if args.cpu_smoke else int(args.experts),
            expert_hidden=min(int(args.expert_hidden), 64) if args.cpu_smoke else int(args.expert_hidden),
            slot_decoder_layers=min(int(args.slot_decoder_layers), 1) if args.cpu_smoke else int(args.slot_decoder_layers),
            slot_decoder_heads=min(int(args.slot_decoder_heads), 2) if args.cpu_smoke else int(args.slot_decoder_heads),
            max_length=int(args.max_length),
            lr_pretrain=float(args.lr_pretrain),
            lr_posttrain=float(args.lr_posttrain),
            kl_weight=float(args.kl_weight),
            row_limit=args.row_limit or (16 if args.cpu_smoke else None),
            checkpoint_dir=None if args.cpu_smoke else args.checkpoint_dir,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "worst_seed": report.get("worst_seed"), "promotion": report.get("promotion")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_candidate_report"]
