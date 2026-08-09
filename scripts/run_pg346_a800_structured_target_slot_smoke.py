"""PG-346 structured Rule-IR target-slot A800 candidate smoke.

The seven-axis abstract context is kept intact.  During slot inference the
model receives context tokens only; the evaluator target is used only as the
training label.  The causal LM loss remains active as a representation anchor,
while fixed Rule-IR slot heads provide a structured decoding diagnostic.

This runner is candidate-only and fail-closed: it never opens promotion,
payload catalog, vulnerability claims, or long-term memory.  It does not
contact a target or start Docker.
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
from app.pg346_structured_target_slot import (  # noqa: E402
    StructuredSlotConfig,
    StructuredTargetSlotDecoder,
    _context_batch,
    build_slot_candidates,
    evaluate_structured_slots,
    train_structured_slot_decoder,
)
from scripts.run_pg343_a800_target_conditioned_smoke import (  # noqa: E402
    _load_rows,
    _sha_file,
    evaluate_gate,
)


SCHEMA_VERSION = "pg346-a800-structured-target-slot-diagnostic-v1"
TZ = ZoneInfo("Asia/Shanghai")
SEEDS = (34601, 34602, 34603)


def _sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _predictive_entropy(model: StructuredTargetSlotDecoder, rows: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], device: Any) -> float:
    import torch
    from torch.nn import functional as F

    if not rows:
        return 0.0
    ids, valid = _context_batch(rows, vocabulary, device, max_length=model.config.max_length)
    model.eval()
    with torch.inference_mode():
        logits, _ = model.backbone(ids, valid_mask=valid)
        probabilities = F.softmax(logits, dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        values = entropy[valid].detach().cpu().tolist()
    return round(sum(float(value) for value in values) / max(len(values), 1), 6)


def _vocabulary_map(vocabulary: Mapping[str, Any]) -> dict[str, int]:
    tokens = [PAD, UNK, *(vocabulary.get("context_tokens") or []), *(vocabulary.get("target_tokens") or [])]
    return {str(token): index for index, token in enumerate(dict.fromkeys(str(token) for token in tokens))}


def main() -> int:
    parser = argparse.ArgumentParser(description="PG-346 structured Rule-IR target-slot A800 smoke")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lm-weight", type=float, default=0.25)
    parser.add_argument("--slot-weight", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text(encoding="utf-8-sig"))
    dataset, audit, vocabulary, rules = load(args.dataset), load(args.audit), load(args.vocabulary), load(args.rules)
    train_rows, train_failures = _load_rows(dataset, "train")
    holdout_rows, holdout_failures = _load_rows(dataset, "implementation_holdout")
    locks = {
        "dataset": _sha_file(args.dataset),
        "audit": _sha_file(args.audit),
        "vocabulary": _sha_file(args.vocabulary),
        "rules": _sha_file(args.rules),
        "script": _sha_file(Path(__file__)),
        "backbone": _sha_file(ROOT / "app" / "pg295_causal_moe.py"),
        "structured_decoder": _sha_file(ROOT / "app" / "pg346_structured_target_slot.py"),
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
        vocabulary=vocabulary,
        rules=rules,
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
        raise RuntimeError("PG-346 structured target-slot gate blocked: " + ",".join(gate["failures"]))
    if not 0 < args.learning_rate <= 0.01 or not 1 <= args.epochs <= 32:
        raise ValueError("learning rate/epochs outside smoke bounds")
    if not 0 < args.lm_weight <= 10 or not 0 < args.slot_weight <= 10:
        raise ValueError("objective weights outside smoke bounds")
    vocabulary_map = _vocabulary_map(vocabulary)
    build_slot_candidates(vocabulary_map)
    max_length = max([len(row["context_tokens"]) + len(row["target_tokens"]) for row in [*train_rows, *holdout_rows]] or [2])
    config = __import__("app.pg295_causal_moe", fromlist=["CausalMoEConfig"]).CausalMoEConfig(d_model=128, n_layers=2, experts=2, expert_hidden=256, max_length=max_length)
    device = torch.device("cuda:0")
    slot_config = StructuredSlotConfig(language_model_weight=args.lm_weight, slot_weight=args.slot_weight)
    candidates: list[dict[str, Any]] = []
    states: dict[str, Mapping[str, Any]] = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        baseline = StructuredTargetSlotDecoder(vocab_size=len(vocabulary_map), config=config, slot_candidates=build_slot_candidates(vocabulary_map)).to(device)
        baseline_entropy = _predictive_entropy(baseline, holdout_rows, vocabulary_map, device)
        model = train_structured_slot_decoder(train_rows, vocabulary_map, device, seed=seed, config=config, slot_config=slot_config, epochs=args.epochs, learning_rate=args.learning_rate)
        post_entropy = _predictive_entropy(model, holdout_rows, vocabulary_map, device)
        relative_drop = round((baseline_entropy - post_entropy) / max(abs(baseline_entropy), 1e-12), 6)
        train_eval = evaluate_structured_slots(model, train_rows, vocabulary_map, device)
        holdout_eval = evaluate_structured_slots(model, holdout_rows, vocabulary_map, device)
        candidates.append({"seed": seed, "train": train_eval, "implementation_holdout": holdout_eval, "baseline_holdout_predictive_entropy": baseline_entropy, "post_holdout_predictive_entropy": post_entropy, "relative_entropy_drop": relative_drop})
        states[str(seed)] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    max_drop = max(float(item["relative_entropy_drop"]) for item in candidates)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "structured_target_slot_diagnostic_candidate_only",
        "gate": gate,
        "locks": locks,
        "training": {"device": "cuda:0", "gpu": device_info.get("name"), "seeds": list(SEEDS), "epochs": args.epochs, "learning_rate": args.learning_rate, "objective": {"language_model_weight": args.lm_weight, "slot_weight": args.slot_weight}, "context_only_slot_inference": True, "target_tokens_used_as_labels_only": True, "required_max_length": max_length, "candidate_only": True},
        "candidates": candidates,
        "worst_seed": {"max_relative_entropy_drop": max_drop, "entropy_gate_passed": max_drop <= 0.25, "negative_false_allow_max": max(int(item["implementation_holdout"]["negative_false_allow"]) for item in candidates), "ask_recall_min": min(float(item["implementation_holdout"]["ask_recall"] or 0.0) for item in candidates), "repair_recall_min": min(float(item["implementation_holdout"]["repair_recall"] or 0.0) for item in candidates), "variant_recall_min": min(float(item["implementation_holdout"]["variant_recall"] or 0.0) for item in candidates), "positive_recall_min": min(float(item["implementation_holdout"]["positive_recall"] or 0.0) for item in candidates)},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "interpretation": "结构化 slot 头只改变目标解码，不压缩七轴上下文；必须同时观察 holdout ASK/repair/negative 与熵门，不能把 train slot accuracy 当作 payload 能力。",
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
