"""PG-378 function-preserving teacher-residual context candidate.

PG-376/377 showed that a randomly initialized high-capacity student improves
token accuracy while collapsing predictive entropy.  PG-378 tests one
specific cause: keep a trained PG-375 teacher as the zero-point and train only
a bounded residual around its logits.  The lane still reads context tokens
only; target tokens, raw wire, evaluator answers and payloads are never
loaded.  It is representation evidence, not capability/SFT/RL or a payload
generator.
"""

from __future__ import annotations

import argparse
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

from app.pg293_failure_next_action import PAD, UNK  # noqa: E402
from app.pg295_causal_moe import CausalMoEConfig, CausalMoELanguageModel  # noqa: E402
from scripts.run_pg377_entropy_preserved_context_candidate import (  # noqa: E402
    PROMOTION_KEYS,
    SEEDS as PG377_SEEDS,
    _batch_metrics,
    _build_vocabulary,
    _config,
    _encode,
    _entropy,
    _load,
    _load_teacher,
    _safe_context_rows,
    _sha,
    _sha_json,
    _validate_device as _pg377_validate_device,
)


SCHEMA_VERSION = "pg378-teacher-residual-context-candidate-v1"
SEEDS = (37801, 37802, 37803)


def _validate_device(device: str) -> None:
    """Keep the remote gate visible in this runner as well as in PG-377."""

    if device == "cpu":
        return
    if device != "cuda:0":
        raise RuntimeError("PG-378 remote lane only permits cuda:0")
    if os.environ.get("BLACKBOX_REMOTE_A800_TRAIN") != "1":
        raise RuntimeError("PG-378 remote lane requires BLACKBOX_REMOTE_A800_TRAIN=1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("PG-378 remote lane requires CUDA_VISIBLE_DEVICES=0")
    _pg377_validate_device(device)


def _residual_metrics(
    student: CausalMoELanguageModel,
    teacher: CausalMoELanguageModel,
    rows: Sequence[Mapping[str, Any]],
    vocabulary: Mapping[str, int],
    device: torch.device,
    *,
    max_length: int,
    batch_size: int,
    residual_scale: float,
) -> dict[str, Any]:
    """Evaluate teacher + scaled student residual without reading targets."""

    if not rows:
        return {"rows": 0, "next_token_count": 0, "mean_next_token_loss": None, "mean_predictive_entropy_nats": None, "token_accuracy": None, "teacher_kl": None, "entropy_relative_delta": None}
    student.eval()
    teacher.eval()
    total_loss = total_entropy = total_kl = total_delta = 0.0
    total_correct = total_tokens = 0
    with torch.inference_mode():
        for start in range(0, len(rows), max(1, int(batch_size))):
            batch = rows[start : start + max(1, int(batch_size))]
            ids, valid = _encode(batch, vocabulary, device, max_length=max_length)
            if ids.shape[1] <= 1:
                continue
            student_logits, _ = student(ids[:, :-1], valid_mask=valid[:, :-1])
            teacher_logits, _ = teacher(ids[:, :-1], valid_mask=valid[:, :-1])
            mixed = teacher_logits + float(residual_scale) * student_logits
            labels = ids[:, 1:]
            label_valid = valid[:, 1:]
            flat_valid = label_valid.reshape(-1)
            if not bool(flat_valid.any()):
                continue
            mixed_flat = mixed.reshape(-1, mixed.shape[-1])[flat_valid]
            teacher_flat = teacher_logits.reshape(-1, teacher_logits.shape[-1])[flat_valid]
            labels_flat = labels.reshape(-1)[flat_valid]
            count = int(labels_flat.numel())
            total_loss += float(F.cross_entropy(mixed_flat, labels_flat, reduction="sum").cpu())
            mixed_entropy = _entropy(mixed_flat)
            teacher_entropy = _entropy(teacher_flat)
            total_entropy += float(mixed_entropy.sum().cpu())
            total_delta += float((mixed_entropy - teacher_entropy).abs().sum().cpu())
            total_correct += int((mixed_flat.argmax(dim=-1) == labels_flat).sum().cpu())
            total_kl += float(F.kl_div(F.log_softmax(mixed_flat, dim=-1), F.softmax(teacher_flat, dim=-1), reduction="sum").cpu())
            total_tokens += count
    if total_tokens == 0:
        return {"rows": len(rows), "next_token_count": 0, "mean_next_token_loss": None, "mean_predictive_entropy_nats": None, "token_accuracy": None, "teacher_kl": None, "entropy_relative_delta": None}
    return {
        "rows": len(rows),
        "next_token_count": total_tokens,
        "mean_next_token_loss": round(total_loss / total_tokens, 6),
        "mean_predictive_entropy_nats": round(total_entropy / total_tokens, 6),
        "token_accuracy": round(total_correct / total_tokens, 6),
        "teacher_kl": round(total_kl / total_tokens, 6),
        "entropy_relative_delta": round(total_delta / total_tokens, 6),
    }


def _train_residual(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    vocabulary: Mapping[str, int],
    teacher: CausalMoELanguageModel,
    config: CausalMoEConfig,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    temperature: float,
    residual_scale: float,
    kl_weight: float,
    entropy_weight: float,
) -> CausalMoELanguageModel:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    student = CausalMoELanguageModel(vocab_size=len(vocabulary), config=config).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=float(learning_rate), weight_decay=0.01)
    pad = int(vocabulary[PAD])
    teacher.eval()
    for epoch in range(max(1, int(epochs))):
        order = list(range(len(train_rows)))
        random.Random(int(seed) + epoch).shuffle(order)
        for start in range(0, len(order), max(1, int(batch_size))):
            batch = [train_rows[index] for index in order[start : start + max(1, int(batch_size))]]
            ids, valid = _encode(batch, vocabulary, device, max_length=config.max_length)
            if ids.shape[1] <= 1:
                continue
            student.train()
            student_logits, balance = student(ids[:, :-1], valid_mask=valid[:, :-1])
            with torch.no_grad():
                teacher_logits, _ = teacher(ids[:, :-1], valid_mask=valid[:, :-1])
            mixed = teacher_logits.detach() + float(residual_scale) * student_logits
            labels = ids[:, 1:]
            label_valid = valid[:, 1:]
            flat_valid = label_valid.reshape(-1)
            if not bool(flat_valid.any()):
                continue
            mixed_flat = mixed.reshape(-1, mixed.shape[-1])[flat_valid]
            teacher_flat = teacher_logits.reshape(-1, teacher_logits.shape[-1])[flat_valid]
            labels_flat = labels.reshape(-1)[flat_valid]
            ce = F.cross_entropy(mixed_flat, labels_flat)
            temperature_value = max(float(temperature), 1e-4)
            teacher_prob = F.softmax(teacher_flat / temperature_value, dim=-1)
            mixed_log_prob = F.log_softmax(mixed_flat / temperature_value, dim=-1)
            kl = F.kl_div(mixed_log_prob, teacher_prob, reduction="batchmean") * (temperature_value * temperature_value)
            entropy_gap = (_entropy(mixed_flat) - _entropy(teacher_flat).detach()).pow(2).mean()
            residual_l2 = student_logits.pow(2).mean()
            loss = ce + float(kl_weight) * kl + float(entropy_weight) * entropy_gap + 1e-4 * residual_l2 + 0.01 * balance
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
    student.eval()
    return student


def run_candidate(
    *,
    dataset: Mapping[str, Any],
    audit: Mapping[str, Any],
    dataset_path: Path,
    audit_path: Path,
    rules_path: Path,
    teacher_checkpoint: Path,
    device: str = "cpu",
    seeds: Sequence[int] = SEEDS,
    epochs: int = 1,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    temperature: float = 2.0,
    residual_scale: float = 0.1,
    kl_weight: float = 1.0,
    entropy_weight: float = 0.25,
    config: CausalMoEConfig | None = None,
    checkpoint_dir: Path | None = None,
    train_limit: int | None = None,
    holdout_limit: int | None = None,
) -> dict[str, Any]:
    promotion = {key: False for key in PROMOTION_KEYS}
    if dataset.get("status") not in {"candidate_only", "diagnostic_candidate_only"}:
        return {"schema_version": SCHEMA_VERSION, "status": "blocked_representation_contract", "reason": "dataset_status", "promotion": promotion}
    if dataset.get("representation_pretrain_candidate_allowed") is not True or dataset.get("capability_training_allowed") is not False:
        return {"schema_version": SCHEMA_VERSION, "status": "blocked_representation_contract", "reason": "dataset_capability_gate", "promotion": promotion}
    if audit.get("status") != "passed_candidate_audit":
        return {"schema_version": SCHEMA_VERSION, "status": "blocked_representation_contract", "reason": "audit_not_passed", "promotion": promotion}
    if not 0 < float(residual_scale) <= 1:
        raise ValueError("PG-378 residual_scale must be in (0, 1]")
    _validate_device(device)
    train, train_failures, train_count = _safe_context_rows(dataset, split="train")
    holdout, holdout_failures, holdout_count = _safe_context_rows(dataset, split="implementation_holdout")
    vocabulary = _build_vocabulary(dataset, train)
    if train_limit is not None:
        train = train[: max(1, int(train_limit))]
    if holdout_limit is not None:
        holdout = holdout[: max(1, int(holdout_limit))]
    if train_failures or holdout_failures or not train or not holdout:
        return {"schema_version": SCHEMA_VERSION, "status": "blocked_representation_contract", "reason": "context_rows", "failures": train_failures + holdout_failures, "promotion": promotion}
    required_window = max(max(len(row["context_tokens"]) for row in train), max(len(row["context_tokens"]) for row in holdout))
    effective = config or CausalMoEConfig(d_model=512, n_heads=4, n_layers=8, experts=8, expert_hidden=2048, max_length=max(768, required_window))
    if effective.max_length < required_window:
        raise ValueError("PG-378 max_length cannot truncate context")
    torch_device = torch.device(device)
    teacher, teacher_meta = _load_teacher(teacher_checkpoint, vocabulary=vocabulary, device=torch_device)
    teacher_holdout = _batch_metrics(teacher, holdout, vocabulary, torch_device, max_length=effective.max_length, batch_size=batch_size)
    candidates: list[dict[str, Any]] = []
    for seed in seeds:
        student = _train_residual(train_rows=train, vocabulary=vocabulary, teacher=teacher, config=effective, device=torch_device, seed=int(seed), epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, temperature=temperature, residual_scale=residual_scale, kl_weight=kl_weight, entropy_weight=entropy_weight)
        metrics = _residual_metrics(student, teacher, holdout, vocabulary, torch_device, max_length=effective.max_length, batch_size=batch_size, residual_scale=residual_scale)
        before = float(teacher_holdout.get("mean_predictive_entropy_nats") or 0.0)
        after = float(metrics.get("mean_predictive_entropy_nats") or 0.0)
        relative_delta = 0.0 if before <= 0 else round((after - before) / before, 6)
        checkpoint = {"path": None, "sha256": None}
        if checkpoint_dir is not None:
            path = checkpoint_dir / f"pg378_teacher_residual_seed_{int(seed)}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"schema_version": SCHEMA_VERSION, "seed": int(seed), "config": dict(effective.__dict__), "model_state": {key: value.detach().cpu() for key, value in student.state_dict().items()}, "vocabulary": dict(vocabulary), "teacher": teacher_meta, "residual_scale": float(residual_scale), "context_only": True, "target_tokens_read": False, "promotion": promotion}, path)
            checkpoint = {"path": str(path), "sha256": _sha(path)}
        candidates.append({"seed": int(seed), "teacher_holdout": teacher_holdout, "student_residual_holdout": metrics, "entropy_relative_delta_vs_teacher": relative_delta, "checkpoint": checkpoint})
    worst_abs_delta = max(abs(float(item["entropy_relative_delta_vs_teacher"])) for item in candidates)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "residual_entropy_preserved_candidate_only" if worst_abs_delta <= 0.25 else "blocked_residual_entropy_preservation",
        "gate": {"dataset_candidate": True, "audit_passed_candidate": True, "train_only_vocabulary": True, "holdout_not_optimized": True, "target_tokens_read": False, "teacher_is_trained": True, "function_preserving_residual": True, "entropy_threshold": 0.25, "worst_entropy_relative_abs_delta": round(worst_abs_delta, 6), "entropy_gate_passed": worst_abs_delta <= 0.25, "training_allowed": True},
        "data": {"train_rows": len(train), "holdout_rows": len(holdout), "source_train_count": train_count, "source_holdout_count": holdout_count, "train_context_vocab_size": len(vocabulary) - 2, "vocabulary_size": len(vocabulary), "vocabulary_scope": "train_context_only", "required_context_window": required_window},
        "training": {"device": device, "seeds": [int(seed) for seed in seeds], "epochs": int(epochs), "batch_size": int(batch_size), "learning_rate": float(learning_rate), "temperature": float(temperature), "residual_scale": float(residual_scale), "kl_weight": float(kl_weight), "entropy_weight": float(entropy_weight), "context_only": True, "target_tokens_read": False, "capability_training": False, "holdout_used_for_optimization": False, "config": dict(effective.__dict__), "teacher": teacher_meta},
        "execution": {"optimizer_started": True, "gpu_touched": device != "cpu", "docker_started": False, "network_used": False, "checkpoint_written": bool(checkpoint_dir)},
        "candidates": candidates,
        "promotion": promotion,
    }
    result["report_sha256"] = _sha_json(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="PG-378 teacher-residual context representation candidate")
    parser.add_argument("--dataset", type=Path, default=ROOT / "research" / "pg375_strict_filtered_rule_ir_dataset_v1.json")
    parser.add_argument("--audit", type=Path, default=ROOT / "research" / "pg375_strict_filtered_rule_ir_audit_v1.json")
    parser.add_argument("--rules", type=Path, default=ROOT / "research" / "improvement_rules.json")
    parser.add_argument("--teacher-checkpoint", type=Path, default=ROOT / "artifacts" / "pg375-context-representation-a800" / "pg375_context_seed_37521.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "pg378_teacher_residual_context_candidate_v1.json")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--d-model", type=int, default=512); parser.add_argument("--layers", type=int, default=8); parser.add_argument("--experts", type=int, default=8); parser.add_argument("--hidden", type=int, default=2048); parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=1); parser.add_argument("--batch", type=int, default=2); parser.add_argument("--learning-rate", type=float, default=1e-4); parser.add_argument("--temperature", type=float, default=2.0); parser.add_argument("--residual-scale", type=float, default=0.1); parser.add_argument("--kl-weight", type=float, default=1.0); parser.add_argument("--entropy-weight", type=float, default=0.25)
    parser.add_argument("--cpu-smoke", action="store_true"); parser.add_argument("--smoke-rows", type=int, default=2); parser.add_argument("--remote-candidate", action="store_true"); parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.cpu_smoke and args.remote_candidate:
        parser.error("--cpu-smoke and --remote-candidate are mutually exclusive")
    if args.remote_candidate:
        args.device = "cuda:0"
    dataset = _load(args.dataset); audit = _load(args.audit)
    config = _config(d_model=args.d_model, layers=args.layers, experts=args.experts, hidden=args.hidden, max_length=args.max_length, cpu_smoke=args.cpu_smoke)
    if args.cpu_smoke:
        result = run_candidate(dataset=dataset, audit=audit, dataset_path=args.dataset, audit_path=args.audit, rules_path=args.rules, teacher_checkpoint=args.teacher_checkpoint, device="cpu", seeds=(SEEDS[0],), epochs=1, batch_size=min(2, args.batch), learning_rate=args.learning_rate, temperature=args.temperature, residual_scale=args.residual_scale, kl_weight=args.kl_weight, entropy_weight=args.entropy_weight, config=config, train_limit=args.smoke_rows, holdout_limit=args.smoke_rows)
    elif args.remote_candidate:
        result = run_candidate(dataset=dataset, audit=audit, dataset_path=args.dataset, audit_path=args.audit, rules_path=args.rules, teacher_checkpoint=args.teacher_checkpoint, device=args.device, seeds=SEEDS, epochs=args.epochs, batch_size=args.batch, learning_rate=args.learning_rate, temperature=args.temperature, residual_scale=args.residual_scale, kl_weight=args.kl_weight, entropy_weight=args.entropy_weight, config=config, checkpoint_dir=args.checkpoint_dir)
    else:
        result = {"schema_version": SCHEMA_VERSION, "status": "plan_only", "training": {"device": "not_run", "context_only": True, "target_tokens_read": False, "teacher_required": True, "function_preserving_residual": True, "config": dict(config.__dict__)}, "execution": {"optimizer_started": False, "gpu_touched": False, "docker_started": False, "network_used": False, "checkpoint_written": False}, "promotion": {key: False for key in PROMOTION_KEYS}}
    result["locks"] = {"dataset_sha256": _sha(args.dataset), "audit_sha256": _sha(args.audit), "rules_sha256": _sha(args.rules), "runner_sha256": _sha(Path(__file__)), "teacher_checkpoint_sha256": _sha(args.teacher_checkpoint)}
    result["report_sha256"] = _sha_json(result)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result if args.json else {"status": result["status"], "report_sha256": result["report_sha256"]}, ensure_ascii=False))
    return 0 if result["status"] not in {"blocked_representation_contract"} else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA_VERSION", "run_candidate", "_residual_metrics", "_train_residual"]
