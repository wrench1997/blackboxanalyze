"""PG-363 A800 candidate: full-context pooled multi-slot Rule-IR decoding.

The causal LM remains the representation anchor, while pooled slot heads use
all valid context positions instead of only the final token.  This is an
offline candidate experiment: no Docker, network target, raw payload, or
evaluator sidecar is read, and promotion is permanently disabled.
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
from app.pg295_causal_moe import CausalMoEConfig  # noqa: E402
from app.pg363_pooled_rule_ir import (  # noqa: E402
    PooledRuleIRDecoder,
    PooledSlotConfig,
    SLOT_PREFIXES,
    build_slot_candidates,
    evaluate_pooled_rule_ir,
    train_pooled_rule_ir,
)
from scripts import run_pg351_a800_ask_oracle_composition_candidate as _base_candidate  # noqa: E402

# The PG-351 loader is reused for its strict abstract-token/firewall checks;
# PG-363 extends its append-only target grammar with the syntax slot.
if "syntax_category_ref=" not in _base_candidate.TARGET_PREFIXES:
    _base_candidate.TARGET_PREFIXES = tuple(_base_candidate.TARGET_PREFIXES) + ("syntax_category_ref=",)
_rows = _base_candidate._rows
evaluate_gate = _base_candidate.evaluate_gate


SCHEMA_VERSION = "pg363-a800-pooled-rule-ir-candidate-v1"
TZ = ZoneInfo("Asia/Shanghai")
SEEDS = (36301, 36302, 36303)


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _vocabulary_map(vocabulary: Mapping[str, Any]) -> dict[str, int]:
    tokens = [PAD, UNK, *(vocabulary.get("context_tokens") or []), *(vocabulary.get("target_tokens") or [])]
    return {str(token): index for index, token in enumerate(dict.fromkeys(str(token) for token in tokens))}


def _predictive_entropy(
    model: PooledRuleIRDecoder,
    rows: Sequence[Mapping[str, Any]],
    vocabulary: Mapping[str, int],
    device: Any,
    *,
    batch_size: int = 32,
) -> float:
    import torch
    from torch.nn import functional as F
    from app.pg363_pooled_rule_ir import _context_batch

    if not rows:
        return 0.0
    model.eval()
    values: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(rows), max(1, int(batch_size))):
            batch = rows[start : start + max(1, int(batch_size))]
            ids, valid = _context_batch(batch, vocabulary, device, max_length=model.config.max_length)
            logits, _ = model.backbone(ids, valid_mask=valid)
            probabilities = F.softmax(logits, dim=-1)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
            values.extend(entropy[valid].detach().cpu().tolist())
    return round(sum(float(value) for value in values) / max(len(values), 1), 6)


def main() -> int:
    parser = argparse.ArgumentParser(description="PG-363 pooled full-context Rule-IR A800 candidate")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lm-weight", type=float, default=0.15)
    parser.add_argument("--slot-weight", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--experts", type=int, default=2)
    parser.add_argument("--expert-hidden", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text(encoding="utf-8-sig"))
    dataset, audit, rules = load(args.dataset), load(args.audit), load(args.rules)
    train_rows, train_failures = _rows(dataset, "train")
    holdout_rows, holdout_failures = _rows(dataset, "implementation_holdout")
    locks = {
        "dataset": _sha_file(args.dataset),
        "audit": _sha_file(args.audit),
        "rules": _sha_file(args.rules),
        "script": _sha_file(Path(__file__)),
        "model": _sha_file(ROOT / "app" / "pg295_causal_moe.py"),
        "decoder": _sha_file(ROOT / "app" / "pg363_pooled_rule_ir.py"),
    }
    import torch

    device_info = {
        "cuda_available": bool(torch.cuda.is_available()),
        "visible_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "current_device": int(torch.cuda.current_device()) if torch.cuda.is_available() else -1,
        "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }
    gate = evaluate_gate(
        dataset=dataset,
        audit=audit,
        env=os.environ,
        device=device_info,
        locks=locks,
        train_rows=train_rows,
        train_failures=train_failures,
        holdout_rows=holdout_rows,
        holdout_failures=holdout_failures,
        now=datetime.now(TZ),
    )
    if not gate["training_allowed"]:
        raise RuntimeError("PG-363 pooled Rule-IR gate blocked: " + ",".join(gate["failures"]))
    if not 0 < args.learning_rate <= 0.01 or not 1 <= args.epochs <= 32 or not 1 <= args.batch_size <= 256:
        raise ValueError("training bounds invalid")
    if args.d_model <= 0 or args.n_heads <= 0 or args.n_layers <= 0 or args.experts <= 0 or args.expert_hidden <= 0 or args.d_model % args.n_heads:
        raise ValueError("invalid model dimensions")
    vocabulary = dataset["vocabulary"]
    vocabulary_map = _vocabulary_map(vocabulary)
    build_slot_candidates(vocabulary_map)
    max_length = max([len(row["context_tokens"]) + len(row["target_tokens"]) for row in [*train_rows, *holdout_rows]] or [2])
    config = CausalMoEConfig(d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, experts=args.experts, expert_hidden=args.expert_hidden, max_length=max_length)
    device = torch.device("cuda:0")
    if not 0.0 <= args.label_smoothing <= 0.5:
        raise ValueError("label smoothing outside bounds")
    slot_config = PooledSlotConfig(language_model_weight=args.lm_weight, slot_weight=args.slot_weight, label_smoothing=args.label_smoothing)
    candidates: list[dict[str, Any]] = []
    states: dict[str, Mapping[str, Any]] = {}
    for seed in SEEDS:
        baseline = PooledRuleIRDecoder(vocab_size=len(vocabulary_map), config=config, slot_candidates=build_slot_candidates(vocabulary_map)).to(device)
        baseline_entropy = _predictive_entropy(baseline, holdout_rows, vocabulary_map, device, batch_size=args.batch_size)
        model = train_pooled_rule_ir(train_rows, vocabulary_map, device, seed=seed, config=config, slot_config=slot_config, epochs=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size)
        post_entropy = _predictive_entropy(model, holdout_rows, vocabulary_map, device, batch_size=args.batch_size)
        relative_drop = round((baseline_entropy - post_entropy) / max(abs(baseline_entropy), 1e-12), 6)
        train_eval = evaluate_pooled_rule_ir(model, train_rows, vocabulary_map, device, batch_size=args.batch_size)
        holdout_eval = evaluate_pooled_rule_ir(model, holdout_rows, vocabulary_map, device, batch_size=args.batch_size)
        candidates.append({"seed": seed, "train": train_eval, "implementation_holdout": holdout_eval, "baseline_holdout_predictive_entropy": baseline_entropy, "post_holdout_predictive_entropy": post_entropy, "relative_entropy_drop": relative_drop})
        states[str(seed)] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    max_drop = max(float(item["relative_entropy_drop"]) for item in candidates)
    worst = {
        "max_relative_entropy_drop": max_drop,
        "entropy_gate_passed": max_drop <= 0.25,
        "negative_false_allow_max": max(int(item["implementation_holdout"]["negative_false_allow"]) for item in candidates),
        "ask_recall_min": min(float(item["implementation_holdout"]["ask_recall"] or 0.0) for item in candidates),
        "repair_recall_min": min(float(item["implementation_holdout"]["repair_recall"] or 0.0) for item in candidates),
        "abstain_recall_min": min(float(item["implementation_holdout"]["abstain_recall"] or 0.0) for item in candidates),
        "positive_action_recall_min": min(float(item["implementation_holdout"]["positive_action_recall"] or 0.0) for item in candidates),
        "positive_recall_min": min(float(item["implementation_holdout"]["positive_recall"] or 0.0) for item in candidates),
        "rule_ir_sequence_exact_min": min(float(item["implementation_holdout"]["sequence_exact_accuracy"] or 0.0) for item in candidates),
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "pooled_structured_candidate_only",
        "gate": gate,
        "locks": locks,
        "training": {"device": "cuda:0", "gpu": device_info.get("name"), "seeds": list(SEEDS), "epochs": args.epochs, "learning_rate": args.learning_rate, "batch_size": args.batch_size, "objective": {"language_model_weight": args.lm_weight, "slot_weight": args.slot_weight, "label_smoothing": args.label_smoothing}, "pooling": "learned_attention+mean+last_boundary", "context_only_slot_inference": True, "target_tokens_used_as_labels_only": True, "required_max_length": max_length, "candidate_only": True},
        "target_slots": [name for name, _ in SLOT_PREFIXES],
        "candidates": candidates,
        "worst_seed": worst,
        "scientific_gate": {"status": "blocked_candidate_only", "raw_payload_in_context": False, "typed_live_replay_with_model_selected_wire": False, "independent_implementation": False, "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "interpretation": "全上下文 pooling 只修表示读取位置，不把任何 raw wire/evaluator answer 送入模型；必须同时观察跨实现 ASK/repair/negative/entropy，不能用安全 guard 替代模型能力。",
    }
    report["report_sha256"] = _sha_json(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": SCHEMA_VERSION, "config": config.__dict__, "vocabulary": vocabulary_map, "slot_candidates": build_slot_candidates(vocabulary_map), "states": states, "promotion": report["promotion"]}, args.checkpoint)
    print(json.dumps(report if args.json else {"status": report["status"], "worst_seed": report["worst_seed"]}, ensure_ascii=False, indent=2 if args.json else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
