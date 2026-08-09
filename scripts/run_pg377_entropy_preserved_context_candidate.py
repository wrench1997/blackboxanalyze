"""PG-377 teacher-KL/entropy-preserved context representation candidate.

This is deliberately a representation lane, not capability/SFT/RL.  It reads
only the abstract ``context_tokens`` from the strict PG-375 dataset.  A frozen
trained PG-375 context model is the Stage-A teacher; random initialization is
never used as the information-preservation baseline.  The student is trained
with next-token CE plus teacher KL and an entropy-matching penalty.  Holdout
rows are evaluated only after optimization and never enter vocabulary,
teacher fitting, or optimizer batches.

No raw payload, response, URL, evaluator answer, target token, wire value, or
route/family literal is loaded by this module.  All capability/payload/memory
promotion flags remain false even when the entropy gate passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg293_failure_next_action import PAD, UNK  # noqa: E402
from app.pg295_causal_moe import CausalMoEConfig, CausalMoELanguageModel  # noqa: E402
from scripts.run_pg375_context_representation_candidate import (  # noqa: E402
    _build_vocabulary as _pg375_build_vocabulary,
    _safe_context_rows as _pg375_safe_context_rows,
)


SCHEMA_VERSION = "pg377-entropy-preserved-context-candidate-v1"
TZ = ZoneInfo("Asia/Shanghai")
SEEDS = (37701, 37702, 37703)
PROMOTION_KEYS = (
    "training_allowed",
    "memory_promotion_allowed",
    "payload_catalog_promotion_allowed",
    "vulnerability_claim_allowed",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _weekend(now: datetime | None = None) -> bool:
    value = now or datetime.now(TZ)
    if value.tzinfo is None:
        value = value.replace(tzinfo=TZ)
    return value.astimezone(TZ).weekday() >= 5


def _safe_context_rows(dataset: Mapping[str, Any], *, split: str) -> tuple[list[dict[str, Any]], list[str], int]:
    rows, failures, count = _pg375_safe_context_rows(dataset, split=split)
    # Copy only context.  In particular, do not inspect target_tokens while
    # building batches or vocabulary.
    return [{"context_tokens": [str(token) for token in row["context_tokens"]]} for row in rows], failures, count


def _build_vocabulary(dataset: Mapping[str, Any], train_rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return _pg375_build_vocabulary(dataset, train_rows)


def _encode(
    rows: Sequence[Mapping[str, Any]],
    vocabulary: Mapping[str, int],
    device: torch.device,
    *,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not rows:
        return torch.empty((0, 1), dtype=torch.long, device=device), torch.empty((0, 1), dtype=torch.bool, device=device)
    sequences: list[list[int]] = []
    for row in rows:
        sequence: list[int] = []
        for token in row.get("context_tokens", []):
            token_text = str(token)
            if token_text not in vocabulary:
                raise ValueError("train/holdout context token outside locked vocabulary")
            sequence.append(int(vocabulary[token_text]))
        if len(sequence) > int(max_length):
            raise ValueError("PG-377 context sequence exceeds max_length; refusing silent truncation")
        sequences.append(sequence)
    width = max((len(item) for item in sequences), default=1)
    pad = int(vocabulary[PAD])
    ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device)
    valid = torch.zeros((len(sequences), width), dtype=torch.bool, device=device)
    for index, sequence in enumerate(sequences):
        if sequence:
            ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
            valid[index, : len(sequence)] = True
    return ids, valid


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return -(log_prob.exp() * log_prob).sum(dim=-1)


def _batch_metrics(
    model: CausalMoELanguageModel,
    rows: Sequence[Mapping[str, Any]],
    vocabulary: Mapping[str, int],
    device: torch.device,
    *,
    max_length: int,
    batch_size: int,
    teacher: CausalMoELanguageModel | None = None,
) -> dict[str, Any]:
    if not rows:
        return {"rows": 0, "next_token_count": 0, "mean_next_token_loss": None, "mean_predictive_entropy_nats": None, "token_accuracy": None, "teacher_kl": None, "entropy_gap_abs": None}
    total_loss = 0.0
    total_entropy = 0.0
    total_correct = 0
    total_tokens = 0
    total_kl = 0.0
    total_gap = 0.0
    model.eval()
    if teacher is not None:
        teacher.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), max(1, int(batch_size))):
            batch = rows[start : start + max(1, int(batch_size))]
            ids, valid = _encode(batch, vocabulary, device, max_length=max_length)
            if ids.shape[1] <= 1:
                continue
            logits, _ = model(ids[:, :-1], valid_mask=valid[:, :-1])
            labels = ids[:, 1:]
            label_valid = valid[:, 1:]
            flat_valid = label_valid.reshape(-1)
            if not bool(flat_valid.any()):
                continue
            flat_logits = logits.reshape(-1, logits.shape[-1])[flat_valid]
            flat_labels = labels.reshape(-1)[flat_valid]
            total_loss += float(F.cross_entropy(flat_logits, flat_labels, reduction="sum").cpu())
            total_entropy += float(_entropy(flat_logits).sum().cpu())
            total_correct += int((flat_logits.argmax(dim=-1) == flat_labels).sum().cpu())
            count = int(flat_labels.numel())
            total_tokens += count
            if teacher is not None:
                teacher_logits, _ = teacher(ids[:, :-1], valid_mask=valid[:, :-1])
                teacher_flat = teacher_logits.reshape(-1, teacher_logits.shape[-1])[flat_valid]
                student_log_prob = F.log_softmax(flat_logits, dim=-1)
                teacher_prob = F.softmax(teacher_flat, dim=-1)
                total_kl += float(F.kl_div(student_log_prob, teacher_prob, reduction="sum").cpu())
                total_gap += float((_entropy(flat_logits) - _entropy(teacher_flat)).abs().sum().cpu())
    if total_tokens == 0:
        return {"rows": len(rows), "next_token_count": 0, "mean_next_token_loss": None, "mean_predictive_entropy_nats": None, "token_accuracy": None, "teacher_kl": None, "entropy_gap_abs": None}
    return {
        "rows": len(rows),
        "next_token_count": total_tokens,
        "mean_next_token_loss": round(total_loss / total_tokens, 6),
        "mean_predictive_entropy_nats": round(total_entropy / total_tokens, 6),
        "token_accuracy": round(total_correct / total_tokens, 6),
        "teacher_kl": None if teacher is None else round(total_kl / total_tokens, 6),
        "entropy_gap_abs": None if teacher is None else round(total_gap / total_tokens, 6),
    }


def _load_teacher(
    checkpoint_path: Path,
    *,
    vocabulary: Mapping[str, int],
    device: torch.device,
) -> tuple[CausalMoELanguageModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("config"), Mapping) or not isinstance(payload.get("model_state"), Mapping):
        raise ValueError("PG-377 teacher checkpoint missing model contract")
    if payload.get("context_only") is not True:
        raise ValueError("PG-377 teacher is not a context-only checkpoint")
    target_tokens_read = payload.get("target_tokens_read")
    # PG-375's frozen checkpoint schema predates the explicit field, but its
    # schema name plus context_only flag is an immutable source contract.  Do
    # not generalize this fallback to arbitrary checkpoints: unknown schemas
    # with a missing field remain blocked rather than silently treated false.
    legacy_context_contract = target_tokens_read is None and payload.get("schema_version") == "pg375-context-representation-candidate-v1"
    if target_tokens_read is not False and not legacy_context_contract:
        raise ValueError("PG-377 teacher target_tokens_read contract is missing or unsafe")
    promotion = dict(payload.get("promotion") or {})
    if any(bool(promotion.get(key)) for key in PROMOTION_KEYS):
        raise ValueError("PG-377 teacher promotion flags are not closed")
    teacher_vocab = payload.get("vocabulary")
    if not isinstance(teacher_vocab, Mapping) or dict(teacher_vocab) != dict(vocabulary):
        raise ValueError("PG-377 teacher vocabulary is not identical to locked train-only vocabulary")
    config = CausalMoEConfig(**{str(key): value for key, value in dict(payload["config"]).items()})
    teacher = CausalMoELanguageModel(vocab_size=len(vocabulary), config=config).to(device)
    teacher.load_state_dict({str(key): value.to(device) for key, value in dict(payload["model_state"]).items()}, strict=True)
    teacher.eval()
    return teacher, {"checkpoint_sha256": _sha(checkpoint_path), "seed": int(payload.get("seed", 0)), "config": dict(config.__dict__), "target_tokens_read": False, "legacy_context_contract": bool(legacy_context_contract)}


def _validate_device(device: str) -> None:
    if device == "cpu":
        return
    if device != "cuda:0":
        raise RuntimeError("PG-377 remote lane only permits cuda:0")
    if not _weekend():
        raise RuntimeError("PG-377 remote A800 lane is weekend-only")
    if os.environ.get("BLACKBOX_REMOTE_A800_TRAIN") != "1":
        raise RuntimeError("PG-377 remote lane requires BLACKBOX_REMOTE_A800_TRAIN=1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("PG-377 remote lane requires CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("PG-377 remote lane requires exactly one visible CUDA device")
    torch.cuda.set_device(0)
    if "A800" not in torch.cuda.get_device_name(0):
        raise RuntimeError("PG-377 remote lane requires NVIDIA A800 GPU0")


def _config(*, d_model: int, layers: int, experts: int, hidden: int, max_length: int, cpu_smoke: bool) -> CausalMoEConfig:
    values = {"d_model": int(d_model), "n_heads": 4, "n_layers": int(layers), "experts": int(experts), "expert_hidden": int(hidden), "max_length": int(max_length)}
    if cpu_smoke:
        values.update({"d_model": 32, "n_heads": 4, "n_layers": 1, "experts": 2, "expert_hidden": 64, "max_length": max(64, int(max_length))})
    if any(int(values[key]) <= 0 for key in ("d_model", "n_layers", "experts", "expert_hidden", "max_length")) or int(values["d_model"]) % 4:
        raise ValueError("PG-377 invalid model dimensions")
    return CausalMoEConfig(**values)


def _train_student(
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
    kl_weight: float,
    entropy_weight: float,
) -> CausalMoELanguageModel:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    student = CausalMoELanguageModel(vocab_size=len(vocabulary), config=config).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=float(learning_rate), weight_decay=0.01)
    pad = int(vocabulary[PAD])
    for _ in range(max(1, int(epochs))):
        order = list(range(len(train_rows)))
        random.Random(int(seed) + _).shuffle(order)
        for start in range(0, len(order), max(1, int(batch_size))):
            batch = [train_rows[index] for index in order[start : start + max(1, int(batch_size))]]
            ids, valid = _encode(batch, vocabulary, device, max_length=config.max_length)
            if ids.shape[1] <= 1:
                continue
            student.train()
            logits, balance = student(ids[:, :-1], valid_mask=valid[:, :-1])
            with torch.no_grad():
                teacher_logits, _ = teacher(ids[:, :-1], valid_mask=valid[:, :-1])
            labels = ids[:, 1:]
            label_valid = valid[:, 1:]
            flat_valid = label_valid.reshape(-1)
            if not bool(flat_valid.any()):
                continue
            student_flat = logits.reshape(-1, logits.shape[-1])[flat_valid]
            teacher_flat = teacher_logits.reshape(-1, teacher_logits.shape[-1])[flat_valid]
            labels_flat = labels.reshape(-1)[flat_valid]
            ce = F.cross_entropy(student_flat, labels_flat)
            t = max(float(temperature), 1e-4)
            teacher_prob = F.softmax(teacher_flat / t, dim=-1)
            student_log_prob = F.log_softmax(student_flat / t, dim=-1)
            kl = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (t * t)
            student_entropy = _entropy(student_flat).mean()
            teacher_entropy = _entropy(teacher_flat).mean().detach()
            entropy_gap = (student_entropy - teacher_entropy).pow(2)
            loss = ce + float(kl_weight) * kl + float(entropy_weight) * entropy_gap + 0.01 * balance
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
    kl_weight: float = 1.0,
    entropy_weight: float = 0.25,
    config: CausalMoEConfig | None = None,
    checkpoint_dir: Path | None = None,
    train_limit: int | None = None,
    holdout_limit: int | None = None,
) -> dict[str, Any]:
    if dataset.get("status") not in {"candidate_only", "diagnostic_candidate_only"}:
        return {"schema_version": SCHEMA_VERSION, "status": "blocked_representation_contract", "reason": "dataset_status", "promotion": {key: False for key in PROMOTION_KEYS}}
    if dataset.get("representation_pretrain_candidate_allowed") is not True:
        return {"schema_version": SCHEMA_VERSION, "status": "blocked_representation_contract", "reason": "representation_pretrain_not_allowed", "promotion": {key: False for key in PROMOTION_KEYS}}
    if dataset.get("capability_training_allowed") is not False:
        return {"schema_version": SCHEMA_VERSION, "status": "blocked_representation_contract", "reason": "capability_gate_not_closed", "promotion": {key: False for key in PROMOTION_KEYS}}
    if audit.get("status") != "passed_candidate_audit":
        return {"schema_version": SCHEMA_VERSION, "status": "blocked_representation_contract", "reason": "audit_not_passed", "promotion": {key: False for key in PROMOTION_KEYS}}
    _validate_device(device)
    train, train_failures, train_count = _safe_context_rows(dataset, split="train")
    holdout, holdout_failures, holdout_count = _safe_context_rows(dataset, split="implementation_holdout")
    # Build the coordinate system from the complete observed train split before
    # applying bounded CPU-smoke limits.  Limiting first would accidentally
    # create a different vocabulary and reject a valid frozen teacher.
    vocabulary = _build_vocabulary(dataset, train)
    if train_limit is not None:
        train = train[: max(1, int(train_limit))]
    if holdout_limit is not None:
        holdout = holdout[: max(1, int(holdout_limit))]
    if train_failures or holdout_failures or not train or not holdout:
        return {"schema_version": SCHEMA_VERSION, "status": "blocked_representation_contract", "reason": "context_rows", "failures": train_failures + holdout_failures, "promotion": {key: False for key in PROMOTION_KEYS}}
    required_window = max(max(len(row["context_tokens"]) for row in train), max(len(row["context_tokens"]) for row in holdout))
    effective = config or CausalMoEConfig(d_model=384, n_heads=4, n_layers=6, experts=4, expert_hidden=1024, max_length=max(768, required_window))
    if effective.max_length < required_window:
        raise ValueError("PG-377 max_length cannot truncate context")
    torch_device = torch.device(device)
    teacher, teacher_meta = _load_teacher(teacher_checkpoint, vocabulary=vocabulary, device=torch_device)
    teacher_holdout = _batch_metrics(teacher, holdout, vocabulary, torch_device, max_length=effective.max_length, batch_size=batch_size)
    candidates: list[dict[str, Any]] = []
    for seed in seeds:
        student = _train_student(train_rows=train, vocabulary=vocabulary, teacher=teacher, config=effective, device=torch_device, seed=int(seed), epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, temperature=temperature, kl_weight=kl_weight, entropy_weight=entropy_weight)
        metrics = _batch_metrics(student, holdout, vocabulary, torch_device, max_length=effective.max_length, batch_size=batch_size, teacher=teacher)
        before = float(teacher_holdout.get("mean_predictive_entropy_nats") or 0.0)
        after = float(metrics.get("mean_predictive_entropy_nats") or 0.0)
        relative_drop = 0.0 if before <= 0 else round((before - after) / before, 6)
        checkpoint = {"path": None, "sha256": None}
        if checkpoint_dir is not None:
            path = checkpoint_dir / f"pg377_entropy_context_seed_{int(seed)}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"schema_version": SCHEMA_VERSION, "seed": int(seed), "config": dict(effective.__dict__), "model_state": {key: value.detach().cpu() for key, value in student.state_dict().items()}, "vocabulary": dict(vocabulary), "teacher": teacher_meta, "context_only": True, "target_tokens_read": False, "promotion": {key: False for key in PROMOTION_KEYS}}, path)
            checkpoint = {"path": str(path), "sha256": _sha(path)}
        candidates.append({"seed": int(seed), "teacher_holdout": teacher_holdout, "student_holdout": metrics, "entropy_relative_drop_vs_trained_teacher": relative_drop, "checkpoint": checkpoint})
    worst_drop = max(float(item["entropy_relative_drop_vs_trained_teacher"]) for item in candidates)
    worst_abs_delta = max(abs(float(item["entropy_relative_drop_vs_trained_teacher"])) for item in candidates)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "entropy_preserved_candidate_only" if worst_abs_delta <= 0.25 else "blocked_entropy_preservation",
        "gate": {"dataset_candidate": True, "audit_passed_candidate": True, "train_only_vocabulary": True, "holdout_not_optimized": True, "target_tokens_read": False, "teacher_is_trained": True, "entropy_threshold": 0.25, "worst_entropy_relative_drop": worst_drop, "worst_entropy_relative_abs_delta": round(worst_abs_delta, 6), "entropy_gate_passed": worst_abs_delta <= 0.25, "training_allowed": True},
        "data": {"train_rows": len(train), "holdout_rows": len(holdout), "source_train_count": train_count, "source_holdout_count": holdout_count, "train_context_vocab_size": len(vocabulary) - 2, "vocabulary_size": len(vocabulary), "vocabulary_scope": "train_context_only", "required_context_window": required_window},
        "training": {"device": device, "seeds": [int(seed) for seed in seeds], "epochs": int(epochs), "batch_size": int(batch_size), "learning_rate": float(learning_rate), "temperature": float(temperature), "kl_weight": float(kl_weight), "entropy_weight": float(entropy_weight), "context_only": True, "target_tokens_read": False, "capability_training": False, "holdout_used_for_optimization": False, "config": dict(effective.__dict__), "teacher": teacher_meta},
        "execution": {"optimizer_started": True, "gpu_touched": device != "cpu", "docker_started": False, "network_used": False, "checkpoint_written": bool(checkpoint_dir)},
        "candidates": candidates,
        "promotion": {key: False for key in PROMOTION_KEYS},
    }
    result["report_sha256"] = _sha_json(result)
    return result


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> int:
    parser = argparse.ArgumentParser(description="PG-377 trained-teacher entropy-preserved context candidate")
    parser.add_argument("--dataset", type=Path, default=ROOT / "research" / "pg375_strict_filtered_rule_ir_dataset_v1.json")
    parser.add_argument("--audit", type=Path, default=ROOT / "research" / "pg375_strict_filtered_rule_ir_audit_v1.json")
    parser.add_argument("--rules", type=Path, default=ROOT / "research" / "improvement_rules.json")
    parser.add_argument("--teacher-checkpoint", type=Path, default=ROOT / "artifacts" / "pg375-context-representation-a800" / "pg375_context_seed_37521.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "pg377_entropy_preserved_context_candidate_v1.json")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--d-model", type=int, default=384); parser.add_argument("--layers", type=int, default=6); parser.add_argument("--experts", type=int, default=4); parser.add_argument("--hidden", type=int, default=1024); parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=1); parser.add_argument("--batch", type=int, default=2); parser.add_argument("--learning-rate", type=float, default=1e-4); parser.add_argument("--temperature", type=float, default=2.0); parser.add_argument("--kl-weight", type=float, default=1.0); parser.add_argument("--entropy-weight", type=float, default=0.25)
    parser.add_argument("--cpu-smoke", action="store_true"); parser.add_argument("--smoke-rows", type=int, default=2); parser.add_argument("--remote-candidate", action="store_true"); parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.cpu_smoke and args.remote_candidate:
        parser.error("--cpu-smoke and --remote-candidate are mutually exclusive")
    if args.remote_candidate:
        args.device = "cuda:0"
    dataset = _load(args.dataset); audit = _load(args.audit)
    config = _config(d_model=args.d_model, layers=args.layers, experts=args.experts, hidden=args.hidden, max_length=args.max_length, cpu_smoke=args.cpu_smoke)
    if args.cpu_smoke:
        result = run_candidate(dataset=dataset, audit=audit, dataset_path=args.dataset, audit_path=args.audit, rules_path=args.rules, teacher_checkpoint=args.teacher_checkpoint, device="cpu", seeds=(SEEDS[0],), epochs=1, batch_size=min(2, args.batch), learning_rate=args.learning_rate, temperature=args.temperature, kl_weight=args.kl_weight, entropy_weight=args.entropy_weight, config=config, train_limit=args.smoke_rows, holdout_limit=args.smoke_rows)
    elif args.remote_candidate:
        result = run_candidate(dataset=dataset, audit=audit, dataset_path=args.dataset, audit_path=args.audit, rules_path=args.rules, teacher_checkpoint=args.teacher_checkpoint, device=args.device, seeds=SEEDS, epochs=args.epochs, batch_size=args.batch, learning_rate=args.learning_rate, temperature=args.temperature, kl_weight=args.kl_weight, entropy_weight=args.entropy_weight, config=config, checkpoint_dir=args.checkpoint_dir)
    else:
        result = {"schema_version": SCHEMA_VERSION, "status": "plan_only", "training": {"device": "not_run", "context_only": True, "target_tokens_read": False, "teacher_required": True, "config": dict(config.__dict__)}, "execution": {"optimizer_started": False, "gpu_touched": False, "docker_started": False, "network_used": False, "checkpoint_written": False}, "promotion": {key: False for key in PROMOTION_KEYS}}
    result["locks"] = {"dataset_sha256": _sha(args.dataset), "audit_sha256": _sha(args.audit), "rules_sha256": _sha(args.rules), "runner_sha256": _sha(Path(__file__)), "teacher_checkpoint_sha256": _sha(args.teacher_checkpoint)}
    result["report_sha256"] = _sha_json(result)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result if args.json else {"status": result["status"], "report_sha256": result["report_sha256"]}, ensure_ascii=False))
    return 0 if result["status"] != "blocked_representation_contract" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA_VERSION", "run_candidate", "_load_teacher", "_build_vocabulary"]
