"""PG-376 high-capacity context-only next-token pretraining candidate.

PG-376 is a representation-pretraining lane, kept separate from the
capability/SFT/RL and evaluator lanes.  The runner consumes the strict PG-375
dataset and audit, projects each row to ``context_tokens`` only, and builds a
vocabulary from the training split's observed context tokens.  The
implementation holdout is evaluated after optimization and is never passed to
the optimizer.  No target field, evaluator sidecar, raw payload/response, wire
material, or target identity is loaded into the model loop.

The default configuration is intentionally larger than the PG-375 CPU wiring
smoke (384 hidden dimensions, six decoder layers, four MoE experts and a 768
token position window).  CUDA execution is an explicitly authorized weekend
lane and must expose exactly NVIDIA A800 GPU0 via ``CUDA_VISIBLE_DEVICES=0``.
Even a successful representation run remains candidate-only: capability,
payload-catalog, memory, and vulnerability promotion are permanently closed.
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


SCHEMA_VERSION = "pg376-highcap-context-pretrain-v1"
TZ = ZoneInfo("Asia/Shanghai")
SEEDS = (37601, 37602, 37603)
DEFAULT_D_MODEL = 384
DEFAULT_LAYERS = 6
DEFAULT_EXPERTS = 4
DEFAULT_HIDDEN = 1024
DEFAULT_MAX_LENGTH = 768
DEFAULT_EPOCHS = 4
DEFAULT_BATCH = 16
DEFAULT_LEARNING_RATE = 1e-4
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
    """Return strict PG-375 context projections without reading target fields.

    The implementation delegates the row-level firewall to PG-375's frozen
    loader.  The returned dictionaries contain only ``context_tokens``; this
    makes it impossible for the training/evaluation helpers below to
    accidentally consume a target sequence.
    """

    rows, failures, count = _pg375_safe_context_rows(dataset, split=split)
    return [{"context_tokens": [str(token) for token in row["context_tokens"]]} for row in rows], failures, count


def _build_vocabulary(dataset: Mapping[str, Any], train_rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Build the model coordinate system from train context observations only."""

    return _pg375_build_vocabulary(dataset, train_rows)


def _encode(
    rows: Sequence[Mapping[str, Any]],
    vocabulary: Mapping[str, int],
    device: torch.device,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not rows:
        return (
            torch.empty((0, 1), dtype=torch.long, device=device),
            torch.empty((0, 1), dtype=torch.bool, device=device),
        )
    sequences: list[list[int]] = []
    for row in rows:
        sequence = [int(vocabulary[token]) for token in row["context_tokens"]]
        if len(sequence) > int(max_length):
            raise ValueError("PG-376 context sequence exceeds max_length; refusing silent truncation")
        sequences.append(sequence)
    width = max((len(sequence) for sequence in sequences), default=1)
    ids = torch.full((len(sequences), width), int(vocabulary[PAD]), dtype=torch.long, device=device)
    valid = torch.zeros((len(sequences), width), dtype=torch.bool, device=device)
    for index, sequence in enumerate(sequences):
        ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        valid[index, : len(sequence)] = True
    return ids, valid


def _metrics(
    model: CausalMoELanguageModel,
    rows: Sequence[Mapping[str, Any]],
    vocabulary: Mapping[str, int],
    device: torch.device,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = DEFAULT_BATCH,
) -> dict[str, Any]:
    """Measure next-token loss/entropy on context-only rows."""

    losses: list[float] = []
    entropies: list[float] = []
    token_correct = 0
    token_count = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), max(1, int(batch_size))):
            batch = rows[start : start + max(1, int(batch_size))]
            ids, valid = _encode(batch, vocabulary, device, max_length=max_length)
            if ids.numel() == 0 or ids.shape[1] < 2:
                continue
            logits, _ = model(ids[:, :-1], valid_mask=valid[:, :-1])
            labels = ids[:, 1:]
            label_valid = valid[:, 1:]
            per_token = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="none",
            ).reshape(labels.shape)
            probabilities = logits.softmax(dim=-1)
            log_probabilities = logits.log_softmax(dim=-1)
            entropy = -(probabilities * log_probabilities).sum(dim=-1)
            losses.extend(per_token[label_valid].detach().cpu().tolist())
            entropies.extend(entropy[label_valid].detach().cpu().tolist())
            token_correct += int((logits.argmax(dim=-1)[label_valid] == labels[label_valid]).sum().item())
            token_count += int(label_valid.sum().item())
    return {
        "rows": len(rows),
        "next_token_count": token_count,
        "mean_next_token_loss": round(sum(losses) / max(1, len(losses)), 6) if losses else None,
        "mean_predictive_entropy_nats": round(sum(entropies) / max(1, len(entropies)), 6) if entropies else None,
        "token_accuracy": round(token_correct / max(1, token_count), 6),
    }


def _device_gate(device: str, *, now: datetime | None = None) -> torch.device:
    """Gate CUDA to the explicitly authorized weekend A800 GPU0 lane."""

    if device == "cpu":
        return torch.device("cpu")
    if device != "cuda:0":
        raise RuntimeError("PG-376 remote representation lane only accepts cuda:0")
    if not _weekend(now):
        raise RuntimeError("PG-376 high-capacity remote lane is weekend-only")
    if os.environ.get("BLACKBOX_REMOTE_A800_TRAIN") != "1":
        raise RuntimeError("PG-376 CUDA requires explicit BLACKBOX_REMOTE_A800_TRAIN=1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("PG-376 CUDA requires CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("PG-376 requires exactly one visible CUDA device")
    torch.cuda.set_device(0)
    if "A800" not in torch.cuda.get_device_name(0):
        raise RuntimeError("PG-376 requires NVIDIA A800 GPU0")
    return torch.device("cuda:0")


def _train_context_lm(
    model: CausalMoELanguageModel,
    train_rows: Sequence[Mapping[str, Any]],
    vocabulary: Mapping[str, int],
    device: torch.device,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
) -> None:
    """Optimize only on train context rows; holdout rows never enter here."""

    if not train_rows:
        raise ValueError("PG-376 cannot train without context rows")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=0.01)
    order = list(range(len(train_rows)))
    for epoch in range(max(1, int(epochs))):
        random.Random(int(seed) + epoch).shuffle(order)
        model.train()
        for start in range(0, len(order), max(1, int(batch_size))):
            batch = [train_rows[index] for index in order[start : start + max(1, int(batch_size))]]
            ids, valid = _encode(batch, vocabulary, device, max_length=max_length)
            if ids.shape[1] < 2:
                continue
            logits, balance = model(ids[:, :-1], valid_mask=valid[:, :-1])
            labels = ids[:, 1:]
            label_valid = valid[:, 1:]
            if not bool(label_valid.any()):
                continue
            loss = F.cross_entropy(logits[label_valid], labels[label_valid]) + 0.01 * balance
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def _config_from_args(
    *,
    d_model: int,
    layers: int,
    experts: int,
    hidden: int,
    max_length: int,
    cpu_smoke: bool,
) -> CausalMoEConfig:
    values = {
        "d_model": int(d_model),
        "n_heads": 4,
        "n_layers": int(layers),
        "experts": int(experts),
        "expert_hidden": int(hidden),
        "max_length": int(max_length),
    }
    if cpu_smoke:
        values.update({"d_model": 32, "n_heads": 4, "n_layers": 1, "experts": 2, "expert_hidden": 64})
    if values["d_model"] <= 0 or values["n_layers"] <= 0 or values["experts"] <= 0 or values["expert_hidden"] <= 0 or values["max_length"] <= 0:
        raise ValueError("PG-376 model dimensions and max_length must be positive")
    if values["d_model"] % values["n_heads"]:
        raise ValueError("PG-376 d_model must be divisible by four attention heads")
    return CausalMoEConfig(**values)


def run_candidate(
    *,
    dataset: Mapping[str, Any],
    audit: Mapping[str, Any],
    dataset_path: Path,
    audit_path: Path,
    rules_path: Path,
    device: str = "cpu",
    seeds: Sequence[int] = SEEDS,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    config: CausalMoEConfig | None = None,
    checkpoint_dir: Path | None = None,
    train_limit: int | None = None,
    holdout_limit: int | None = None,
) -> dict[str, Any]:
    """Run a candidate-only context pretrain/evaluation cycle."""

    train_all, train_failures, train_source_count = _safe_context_rows(dataset, split="train")
    holdout_all, holdout_failures, holdout_source_count = _safe_context_rows(dataset, split="implementation_holdout")
    vocabulary = _build_vocabulary(dataset, train_all)
    train_tokens = {token for row in train_all for token in row["context_tokens"]}
    holdout_unknown = sorted({token for row in holdout_all for token in row["context_tokens"] if token not in vocabulary})
    train_signatures = {tuple(row["context_tokens"]) for row in train_all}
    holdout_signatures = {tuple(row["context_tokens"]) for row in holdout_all}
    context_overlap = len(train_signatures & holdout_signatures)
    audit_counts = audit.get("counts") if isinstance(audit.get("counts"), Mapping) else {}
    checks = {
        "dataset_status_candidate": str(dataset.get("status")) == "candidate_only",
        "representation_pretrain_candidate_allowed": dataset.get("representation_pretrain_candidate_allowed") is True,
        "capability_training_closed": dataset.get("capability_training_allowed") is False,
        "audit_passed_candidate_audit": str(audit.get("status")) == "passed_candidate_audit",
        "train_context_rows_valid": bool(train_all) and not train_failures,
        "holdout_context_rows_valid": bool(holdout_all) and not holdout_failures,
        "holdout_vocabulary_closed": not holdout_unknown,
        "active_overlap_zero": context_overlap == 0 and int(audit_counts.get("active_cross_split_exact_overlap", -1)) == 0,
        # Older test fixtures expose only the overlap counters.  The strict
        # PG-375 artifact includes ``unknown_context_tokens``; when that
        # optional counter is absent, the row-level train-only vocabulary and
        # holdout-unknown checks above remain authoritative.
        "audit_unknown_context_zero": audit_counts.get("unknown_context_tokens") in (None, 0),
    }
    failures = sorted(set([key for key, ok in checks.items() if not ok] + train_failures + holdout_failures))
    if holdout_unknown:
        failures.append("holdout_context_unknown_token")
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked_representation_contract" if failures else "representation_pretrain_candidate_only",
        "gate": {
            "checks": checks,
            "failures": sorted(set(failures)),
            "training_allowed": not failures,
            "holdout_used_for_optimization": False,
            "target_tokens_read": False,
        },
        "data": {
            "train_rows": len(train_all),
            "implementation_holdout_rows": len(holdout_all),
            "source_train_count": train_source_count,
            "source_holdout_count": holdout_source_count,
            "train_context_vocab_size": len(train_tokens),
            "vocabulary_size": len(vocabulary),
            "vocabulary_scope": "train_context_only",
            "holdout_unknown_context_count": len(holdout_unknown),
            "active_context_overlap_count": context_overlap,
        },
        "training": {
            "device": device,
            "seeds": [int(seed) for seed in seeds],
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "context_only": True,
            "target_tokens_read": False,
            "holdout_used_for_optimization": False,
            "capability_training": False,
            "config": None if config is None else dict(config.__dict__),
        },
        "locks": {
            "dataset_sha256": _sha(dataset_path),
            "audit_sha256": _sha(audit_path),
            "rules_sha256": _sha(rules_path),
            "runner_sha256": _sha(Path(__file__)),
            "model_sha256": _sha(ROOT / "app" / "pg295_causal_moe.py"),
        },
        "promotion": {key: False for key in PROMOTION_KEYS},
        "execution": {
            "optimizer_started": False,
            "gpu_touched": False,
            "docker_started": False,
            "network_used": False,
            "checkpoint_written": False,
        },
    }
    if failures:
        return result
    if int(batch_size) <= 0 or int(epochs) <= 0 or float(learning_rate) <= 0:
        raise ValueError("PG-376 epochs, batch, and learning_rate must be positive")

    torch_device = _device_gate(device)
    effective = config or CausalMoEConfig(
        d_model=DEFAULT_D_MODEL if device != "cpu" else 32,
        n_heads=4,
        n_layers=DEFAULT_LAYERS if device != "cpu" else 1,
        experts=DEFAULT_EXPERTS if device != "cpu" else 2,
        expert_hidden=DEFAULT_HIDDEN if device != "cpu" else 64,
        max_length=DEFAULT_MAX_LENGTH,
    )
    required_window = max((len(row["context_tokens"]) for row in [*train_all, *holdout_all]), default=1)
    if int(effective.max_length) < required_window:
        raise ValueError("PG-376 max_length is below the full context window; refusing truncation")
    train_rows = train_all[: max(1, int(train_limit))] if train_limit is not None else train_all
    holdout_rows = holdout_all[: max(1, int(holdout_limit))] if holdout_limit is not None else holdout_all
    # Validate the bounded smoke rows against the locked train-only vocabulary.
    if any(token not in vocabulary for row in [*train_rows, *holdout_rows] for token in row["context_tokens"]):
        raise ValueError("PG-376 smoke rows contain a token outside the locked train vocabulary")

    candidates: list[dict[str, Any]] = []
    for seed in seeds:
        torch.manual_seed(int(seed))
        random.seed(int(seed))
        model = CausalMoELanguageModel(vocab_size=len(vocabulary), config=effective).to(torch_device)
        baseline = _metrics(model, holdout_rows, vocabulary, torch_device, max_length=int(effective.max_length), batch_size=batch_size)
        _train_context_lm(
            model,
            train_rows,
            vocabulary,
            torch_device,
            seed=int(seed),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=int(effective.max_length),
        )
        train_metrics = _metrics(model, train_rows, vocabulary, torch_device, max_length=int(effective.max_length), batch_size=batch_size)
        holdout_metrics = _metrics(model, holdout_rows, vocabulary, torch_device, max_length=int(effective.max_length), batch_size=batch_size)
        before_entropy = baseline.get("mean_predictive_entropy_nats")
        after_entropy = holdout_metrics.get("mean_predictive_entropy_nats")
        entropy_drop = None
        if isinstance(before_entropy, (int, float)) and isinstance(after_entropy, (int, float)) and float(before_entropy) > 0:
            entropy_drop = round((float(before_entropy) - float(after_entropy)) / float(before_entropy), 6)
        checkpoint: dict[str, Any] = {"path": None, "sha256": None}
        if checkpoint_dir is not None:
            path = checkpoint_dir / f"pg376_highcap_context_seed_{int(seed)}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema_version": SCHEMA_VERSION,
                    "seed": int(seed),
                    "config": dict(effective.__dict__),
                    "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "vocabulary": dict(vocabulary),
                    "context_only": True,
                    "target_tokens_read": False,
                    "promotion": result["promotion"],
                },
                path,
            )
            checkpoint = {"path": str(path), "sha256": _sha(path)}
        candidates.append(
            {
                "seed": int(seed),
                "baseline_holdout": baseline,
                "train": train_metrics,
                "implementation_holdout": holdout_metrics,
                "holdout_predictive_entropy_relative_drop": entropy_drop,
                "checkpoint": checkpoint,
            }
        )
    result["training"].update(
        {
            "config": dict(effective.__dict__),
            "required_context_window": required_window,
            "optimized_train_rows": len(train_rows),
            "evaluated_holdout_rows": len(holdout_rows),
        }
    )
    result["candidates"] = candidates
    result["execution"] = {
        "optimizer_started": bool(candidates),
        "gpu_touched": device != "cpu",
        "docker_started": False,
        "network_used": False,
        "checkpoint_written": bool(checkpoint_dir),
    }
    return result


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> int:
    parser = argparse.ArgumentParser(description="PG-376 high-capacity context-only next-token pretraining candidate")
    parser.add_argument("--dataset", type=Path, default=ROOT / "research" / "pg375_strict_filtered_rule_ir_dataset_v1.json")
    parser.add_argument("--audit", type=Path, default=ROOT / "research" / "pg375_strict_filtered_rule_ir_audit_v1.json")
    parser.add_argument("--rules", type=Path, default=ROOT / "research" / "improvement_rules.json")
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "pg376_highcap_context_pretrain_v1.json")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--layers", "--n-layers", dest="layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--experts", type=int, default=DEFAULT_EXPERTS)
    parser.add_argument("--hidden", "--expert-hidden", dest="hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch", "--batch-size", dest="batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--cpu-smoke", action="store_true", help="bounded local wiring smoke; no CUDA, Docker, or network")
    parser.add_argument("--smoke-rows", type=int, default=2, help="bounded train/holdout rows for --cpu-smoke")
    parser.add_argument("--remote-candidate", action="store_true", help="explicit weekend A800 GPU0 candidate run")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.cpu_smoke and args.remote_candidate:
        parser.error("--cpu-smoke and --remote-candidate are mutually exclusive")
    if int(args.smoke_rows) <= 0:
        parser.error("--smoke-rows must be positive")
    dataset = _load(args.dataset)
    audit = _load(args.audit)
    config = _config_from_args(
        d_model=args.d_model,
        layers=args.layers,
        experts=args.experts,
        hidden=args.hidden,
        max_length=args.max_length,
        cpu_smoke=args.cpu_smoke,
    )
    if args.remote_candidate:
        args.device = "cuda:0"
    if not args.cpu_smoke and not args.remote_candidate:
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "plan_only",
            "training": {
                "device": "not_run",
                "context_only": True,
                "target_tokens_read": False,
                "holdout_used_for_optimization": False,
                "config": dict(config.__dict__),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch),
            },
            "execution": {"optimizer_started": False, "gpu_touched": False, "docker_started": False, "network_used": False, "checkpoint_written": False},
            "promotion": {key: False for key in PROMOTION_KEYS},
        }
    else:
        result = run_candidate(
            dataset=dataset,
            audit=audit,
            dataset_path=args.dataset,
            audit_path=args.audit,
            rules_path=args.rules,
            device=args.device,
            seeds=(SEEDS[0],) if args.cpu_smoke else SEEDS,
            epochs=1 if args.cpu_smoke else int(args.epochs),
            batch_size=min(2, int(args.batch)) if args.cpu_smoke else int(args.batch),
            learning_rate=float(args.learning_rate),
            config=config,
            checkpoint_dir=None if args.cpu_smoke else args.checkpoint_dir,
            train_limit=int(args.smoke_rows) if args.cpu_smoke else None,
            holdout_limit=int(args.smoke_rows) if args.cpu_smoke else None,
        )
    result["locks"] = {
        "dataset_sha256": _sha(args.dataset),
        "audit_sha256": _sha(args.audit),
        "rules_sha256": _sha(args.rules),
        "runner_sha256": _sha(Path(__file__)),
        "model_sha256": _sha(ROOT / "app" / "pg295_causal_moe.py"),
    }
    result["report_sha256"] = _sha_json(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result if args.json else {"status": result["status"], "report_sha256": result["report_sha256"]}, ensure_ascii=False))
    return 0 if result["status"] != "blocked_representation_contract" else 2


if __name__ == "__main__":
    raise SystemExit(main())
