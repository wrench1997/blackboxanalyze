"""PG-373 staged next-token pretraining + structured Rule-IR candidate.

PG-370 compared the post-training decoder with a randomly initialized model,
so its predictive-entropy drop mixed normal learning with information
collapse.  PG-373 makes the comparison causal and explicit:

1. train a decoder-only next-token baseline on the training split only;
2. copy that baseline and add the structured Rule-IR/ASK/repair/negative
   objectives at a lower learning rate, with a KL anchor to the baseline;
3. evaluate both models on the untouched implementation holdout.

Only abstract context/target tokens are loaded.  This is a candidate-only
research runner: no Docker, network, evaluator, raw payload, response body,
or promotion path exists.  CUDA execution is explicitly restricted to the
authorized remote A800 GPU0 lane.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg295_causal_moe import CausalMoEConfig  # noqa: E402
from scripts.plan_pg369_multitask_moe_candidate import (  # noqa: E402
    PROMOTION_KEYS,
    SLOTS,
    _load_json,
)
from scripts.run_pg370_multitask_moe_candidate import (  # noqa: E402
    LOCKED_DATASET_SHA256,
    LOCKED_AUDIT_SHA256,
    SharedCausalMoEMultiTask,
    _batch_labels,
    _pad_context,
    _pad_lm,
    _sha_file,
    _sha_json,
    _slot_classes_from_values,
    _target_values,
    build_declared_vocabulary,
    evaluate_multitask,
    load_locked_rows,
)

SCHEMA_VERSION = "pg373-staged-pretrain-multitask-candidate-v1"
SEEDS = (37301, 37302, 37303)


def _full_lm_loss(model: SharedCausalMoEMultiTask, rows: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], device: torch.device) -> torch.Tensor:
    """Next-token loss over the complete abstract context+target sequence."""

    if not rows:
        return torch.zeros((), device=device, requires_grad=True)
    _, context_mask, _ = _pad_context(rows, vocabulary, device)
    lm_ids, lm_mask, _ = _pad_lm(rows, vocabulary, device)
    output = model(context_ids=lm_ids[:, :-1], context_mask=lm_mask[:, :-1], lm_ids=lm_ids[:, :-1], lm_mask=lm_mask[:, :-1])
    labels = lm_ids[:, 1:]
    valid = lm_mask[:, 1:]
    loss = F.cross_entropy(output["lm"].reshape(-1, output["lm"].shape[-1]), labels.reshape(-1), reduction="none").reshape(labels.shape)
    return loss[valid].mean() if bool(valid.any()) else loss.mean() * 0.0


def _target_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    if not bool(target_mask.any()):
        return student_logits.sum() * 0.0
    student = F.log_softmax(student_logits[target_mask], dim=-1)
    teacher = F.softmax(teacher_logits[target_mask].detach(), dim=-1)
    return F.kl_div(student, teacher, reduction="batchmean")


def _train_stage_a(model: SharedCausalMoEMultiTask, rows: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], *, epochs: int, microbatch: int, device: torch.device, lr: float) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=0.01)
    order = list(range(len(rows)))
    for epoch in range(max(1, int(epochs))):
        random.Random(1000 + epoch).shuffle(order)
        model.train()
        for start in range(0, len(order), max(1, int(microbatch))):
            batch = [rows[index] for index in order[start : start + max(1, int(microbatch))]]
            loss = _full_lm_loss(model, batch, vocabulary, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def _train_stage_b(
    model: SharedCausalMoEMultiTask,
    teacher: SharedCausalMoEMultiTask,
    rows: Sequence[Mapping[str, Any]],
    vocabulary: Mapping[str, int],
    slot_classes: Mapping[str, Mapping[str, int]],
    *,
    epochs: int,
    microbatch: int,
    device: torch.device,
    lr: float,
    kl_weight: float,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=0.01)
    teacher.eval()
    order = list(range(len(rows)))
    for epoch in range(max(1, int(epochs))):
        random.Random(2000 + epoch).shuffle(order)
        model.train()
        for start in range(0, len(order), max(1, int(microbatch))):
            batch = [rows[index] for index in order[start : start + max(1, int(microbatch))]]
            context_ids, context_mask, _ = _pad_context(batch, vocabulary, device)
            lm_ids, lm_mask, target_mask = _pad_lm(batch, vocabulary, device)
            labels = _batch_labels(batch, slot_classes, device)
            output = model(context_ids, context_mask, lm_ids=lm_ids[:, :-1], lm_mask=lm_mask[:, :-1])
            with torch.no_grad():
                teacher_output = teacher(context_ids, context_mask, lm_ids=lm_ids[:, :-1], lm_mask=lm_mask[:, :-1])
            lm_labels = lm_ids[:, 1:]
            lm_loss_all = F.cross_entropy(output["lm"].reshape(-1, output["lm"].shape[-1]), lm_labels.reshape(-1), ignore_index=int(vocabulary["[PAD]" ]), reduction="none").reshape(lm_labels.shape)
            lm_loss = lm_loss_all[target_mask].mean() if bool(target_mask.any()) else lm_loss_all.mean() * 0.0
            kl = _target_kl(output["lm"], teacher_output["lm"], target_mask)
            slot_loss = torch.stack([F.cross_entropy(output["slot"][key], labels["slot"][key]) for key in SLOTS]).mean()
            ask_loss = F.cross_entropy(output["ask"], labels["ask"])
            repair_loss = F.cross_entropy(output["repair"], labels["repair"])
            negative_loss = F.cross_entropy(output["negative"], labels["negative"])
            loss = 0.25 * lm_loss + 0.25 * kl + slot_loss + 1.5 * ask_loss + 1.5 * repair_loss + 2.0 * negative_loss + 0.01 * output["balance"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def _device_gate(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if os.environ.get("BLACKBOX_REMOTE_A800_TRAIN") != "1":
        raise RuntimeError("PG-373 CUDA requires BLACKBOX_REMOTE_A800_TRAIN=1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("PG-373 CUDA requires CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("PG-373 requires exactly one visible CUDA device")
    torch.cuda.set_device(0)
    if "A800" not in torch.cuda.get_device_name(0):
        raise RuntimeError("PG-373 requires NVIDIA A800 GPU0")
    return torch.device("cuda:0")


def _save_checkpoint(model: SharedCausalMoEMultiTask, vocabulary: Mapping[str, int], slot_classes: Mapping[str, Mapping[str, int]], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": SCHEMA_VERSION, "model_state": model.state_dict(), "vocabulary": dict(vocabulary), "slot_classes": {key: dict(value) for key, value in slot_classes.items()}, "raw_context": False, "raw_payload": False, "evaluator_answer": False}, path)
    return _sha_file(path)


def run_candidate(*, train_rows: Sequence[Mapping[str, Any]], holdout_rows: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], slot_classes: Mapping[str, Mapping[str, int]], seeds: Sequence[int] = SEEDS, device: str = "cpu", pretrain_epochs: int = 1, posttrain_epochs: int = 1, microbatch: int = 2, config: CausalMoEConfig | None = None, checkpoint_dir: Path | None = None) -> dict[str, Any]:
    torch_device = _device_gate(device)
    effective = config or CausalMoEConfig(d_model=32, n_heads=4, n_layers=1, experts=2, expert_hidden=64, max_length=768)
    required_window = max((len(row["context_tokens"]) + len(row["target_tokens"]) for row in [*train_rows, *holdout_rows]), default=1)
    if int(effective.max_length) < required_window:
        raise ValueError("PG-373 max_length is below the measured context window")
    results: list[dict[str, Any]] = []
    for seed in seeds:
        torch.manual_seed(int(seed))
        random.seed(int(seed))
        stage_a = SharedCausalMoEMultiTask(vocab_size=len(vocabulary), config=effective, slot_classes=slot_classes).to(torch_device)
        _train_stage_a(stage_a, train_rows, vocabulary, epochs=pretrain_epochs, microbatch=microbatch, device=torch_device, lr=1e-4)
        baseline = evaluate_multitask(stage_a, holdout_rows, vocabulary, slot_classes, torch_device)
        teacher = copy.deepcopy(stage_a).to(torch_device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        stage_b = copy.deepcopy(stage_a).to(torch_device)
        _train_stage_b(stage_b, teacher, train_rows, vocabulary, slot_classes, epochs=posttrain_epochs, microbatch=microbatch, device=torch_device, lr=5e-5, kl_weight=0.25)
        post = evaluate_multitask(stage_b, holdout_rows, vocabulary, slot_classes, torch_device)
        checkpoint = None
        if checkpoint_dir is not None:
            checkpoint_path = checkpoint_dir / f"pg373_seed_{int(seed)}.pt"
            checkpoint = {"path": str(checkpoint_path), "sha256": _save_checkpoint(stage_b, vocabulary, slot_classes, checkpoint_path)}
        results.append({"seed": int(seed), "baseline": baseline, "post": post, "entropy_relative_drop": round((float(baseline["predictive_entropy"]) - float(post["predictive_entropy"])) / max(abs(float(baseline["predictive_entropy"])), 1e-12), 6), "checkpoint": checkpoint})
    worst = {"sequence_exact_min": min(float(item["post"]["sequence_exact"]) for item in results), "slot_accuracy_min": min(float(item["post"]["slot_accuracy"]) for item in results), "ask_recall_min": min(float(item["post"]["ask_recall"] or 0.0) for item in results), "repair_recall_min": min(float(item["post"]["repair_recall"] or 0.0) for item in results), "positive_recall_min": min(float(item["post"]["positive_recall"] or 0.0) for item in results), "negative_false_allow_max": max(int(item["post"]["negative_false_allow"]) for item in results), "entropy_relative_drop_max": max(float(item["entropy_relative_drop"]) for item in results)}
    return {"status": "cpu_smoke_candidate_only" if device == "cpu" else "remote_candidate_only", "training": {"device": device, "seeds": [int(seed) for seed in seeds], "pretrain_epochs": int(pretrain_epochs), "posttrain_epochs": int(posttrain_epochs), "microbatch": int(microbatch), "baseline_kind": "train_only_next_token_pretrain", "kl_anchor_weight": 0.25, "config": effective.__dict__, "required_context_window": int(required_window), "vocabulary_size": len(vocabulary)}, "candidates": results, "worst_seed": worst, "promotion": {key: False for key in PROMOTION_KEYS}, "scientific_gate": {"typed_live_replay_with_model_selected_wire": False, "independent_implementation": False, "claim_allowed": False, "trained_baseline_entropy_comparison": True}}


def main() -> int:
    parser = argparse.ArgumentParser(description="PG-373 staged next-token pretrain and structured candidate")
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "pg373_staged_pretrain_candidate_v1.json")
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--remote-candidate", action="store_true")
    parser.add_argument("--pretrain-epochs", type=int, default=1)
    parser.add_argument("--posttrain-epochs", type=int, default=1)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--expert-hidden", type=int, default=512)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.cpu_smoke and args.remote_candidate:
        parser.error("--cpu-smoke and --remote-candidate are mutually exclusive")
    train, holdout, locks = load_locked_rows()
    vocabulary = build_declared_vocabulary(locks["declared_vocabulary"])
    slot_classes = _slot_classes_from_values(locks["declared_slot_values"])
    config = CausalMoEConfig(d_model=32 if args.cpu_smoke else int(args.d_model), n_heads=4, n_layers=1 if args.cpu_smoke else int(args.n_layers), experts=2 if args.cpu_smoke else int(args.experts), expert_hidden=64 if args.cpu_smoke else int(args.expert_hidden), max_length=768)
    if args.cpu_smoke:
        result = run_candidate(train_rows=train[:2], holdout_rows=holdout[:2], vocabulary=vocabulary, slot_classes=slot_classes, seeds=(37301,), device="cpu", pretrain_epochs=1, posttrain_epochs=1, microbatch=1, config=config)
    elif args.remote_candidate:
        result = run_candidate(train_rows=train, holdout_rows=holdout, vocabulary=vocabulary, slot_classes=slot_classes, seeds=SEEDS, device="cuda:0", pretrain_epochs=int(args.pretrain_epochs), posttrain_epochs=int(args.posttrain_epochs), microbatch=int(args.microbatch), config=config, checkpoint_dir=args.checkpoint_dir or ROOT / "artifacts" / "pg373-staged-pretrain")
    else:
        result = {"status": "plan_only", "training": {"device": "not_run", "baseline_kind": "train_only_next_token_pretrain"}, "promotion": {key: False for key in PROMOTION_KEYS}, "scientific_gate": {"trained_baseline_entropy_comparison": True, "claim_allowed": False}}
    result["schema_version"] = SCHEMA_VERSION
    result["locks"] = {"datasets": LOCKED_DATASET_SHA256, "audits": LOCKED_AUDIT_SHA256, "runner_sha256": _sha_file(Path(__file__)), "base_model_sha256": _sha_file(ROOT / "app" / "pg295_causal_moe.py"), "rules_sha256": _sha_file(ROOT / "research" / "improvement_rules.json")}
    result["report_sha256"] = _sha_json(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result if args.json else {"status": result["status"], "report_sha256": result["report_sha256"]}, ensure_ascii=False, indent=2 if args.json else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
