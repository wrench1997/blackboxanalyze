"""PG-375 context-only representation candidate.

This is a separate lane from capability/SFT/RL training.  It reads only the
abstract ``context_tokens`` from a strict filtered dataset.  Target slots,
payloads, responses, evaluator sidecars and quarantine records are never
loaded into a model row.  A remote run is allowed only on the explicitly
authorized A800 GPU0 lane; every promotion and capability claim remains false.
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

from app.pg293_failure_next_action import PAD, UNK
from app.pg295_causal_moe import CausalMoEConfig, CausalMoELanguageModel

SCHEMA_VERSION = "pg375-context-representation-candidate-v1"
TZ = ZoneInfo("Asia/Shanghai")
SEEDS = (37521, 37522, 37523)
RAW_PREFIXES = (
    "raw_", "payload=", "payload_", "response_body=", "response_body_text=",
    "wire=", "oracle=", "evaluator=", "route_literal=", "family=",
    "http://", "https://",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _weekend(now: datetime | None = None) -> bool:
    value = now or datetime.now(TZ)
    if value.tzinfo is None:
        value = value.replace(tzinfo=TZ)
    return value.astimezone(TZ).weekday() >= 5


def _safe_context_rows(dataset: Mapping[str, Any], *, split: str) -> tuple[list[dict[str, Any]], list[str], int]:
    """Read only split + context/firewall fields; never inspect target_tokens."""

    records = dataset.get("records")
    if not isinstance(records, list):
        return [], ["dataset_records_missing"], 0
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    count = 0
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            failures.append(f"row_{index}_not_mapping")
            continue
        if str(raw.get("split", "")) != split:
            continue
        count += 1
        context = raw.get("context_tokens")
        firewall = raw.get("context_firewall")
        if not isinstance(context, list) or not context:
            failures.append(f"row_{index}_context_missing")
            continue
        if firewall != {"forbidden_token_count": 0, "sidecars_off_context": True}:
            failures.append(f"row_{index}_firewall")
            continue
        if any(raw.get(key) is not False for key in ("raw_payload_stored", "raw_response_body_stored", "oracle_answer_in_context")):
            failures.append(f"row_{index}_raw_flag")
            continue
        tokens = [str(token) for token in context]
        if any(token.casefold().startswith(RAW_PREFIXES) for token in tokens):
            failures.append(f"row_{index}_raw_token")
            continue
        rows.append({"context_tokens": tokens})
    return rows, sorted(set(failures)), count


def _build_vocabulary(dataset: Mapping[str, Any], train_rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    declared = dataset.get("vocabulary")
    if not isinstance(declared, Mapping) or not isinstance(declared.get("context_tokens"), list):
        raise ValueError("context vocabulary manifest missing")
    observed = {str(token) for row in train_rows for token in row["context_tokens"]}
    declared_tokens = {str(token) for token in declared["context_tokens"]}
    if not observed.issubset(declared_tokens):
        raise ValueError("train context token is absent from locked vocabulary")
    # The manifest is an append-only inventory, not a license to import a
    # category observed only in holdout.  The actual model coordinate system
    # is built from train context observations plus reserved PAD/UNK.
    ordered = [PAD, UNK, *sorted(observed - {PAD, UNK})]
    return {token: index for index, token in enumerate(dict.fromkeys(ordered))}


def _encode(rows: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = [[int(vocabulary[token]) for token in row["context_tokens"]] for row in rows]
    width = max((len(sequence) for sequence in sequences), default=1)
    ids = torch.full((len(sequences), width), int(vocabulary[PAD]), dtype=torch.long, device=device)
    valid = torch.zeros((len(sequences), width), dtype=torch.bool, device=device)
    for index, sequence in enumerate(sequences):
        ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        valid[index, : len(sequence)] = True
    return ids, valid


def _metrics(model: CausalMoELanguageModel, rows: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], device: torch.device, *, batch_size: int = 16) -> dict[str, Any]:
    import torch.nn.functional as F
    losses: list[float] = []
    entropies: list[float] = []
    token_correct = 0
    token_count = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), max(1, int(batch_size))):
            batch = rows[start : start + max(1, int(batch_size))]
            ids, valid = _encode(batch, vocabulary, device)
            if ids.shape[1] < 2:
                continue
            logits, _ = model(ids[:, :-1], valid_mask=valid[:, :-1])
            labels = ids[:, 1:]
            label_valid = valid[:, 1:]
            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_labels = labels.reshape(-1)
            per = F.cross_entropy(flat_logits, flat_labels, reduction="none").reshape(labels.shape)
            probability = logits.softmax(dim=-1)
            log_probability = logits.log_softmax(dim=-1)
            entropy = -(probability * log_probability).sum(dim=-1)
            losses.extend(per[label_valid].detach().cpu().tolist())
            entropies.extend(entropy[label_valid].detach().cpu().tolist())
            token_correct += int((logits.argmax(dim=-1)[label_valid] == labels[label_valid]).sum().item())
            token_count += int(label_valid.sum().item())
    return {
        "rows": len(rows),
        "next_token_count": token_count,
        "mean_loss": round(sum(losses) / max(1, len(losses)), 6) if losses else None,
        "mean_predictive_entropy_nats": round(sum(entropies) / max(1, len(entropies)), 6) if entropies else None,
        "token_accuracy": round(token_correct / max(1, token_count), 6),
    }


def _device_gate(device: str, *, now: datetime | None = None) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if not _weekend(now):
        raise RuntimeError("remote A800 representation lane is weekend-only")
    if os.environ.get("BLACKBOX_REMOTE_A800_TRAIN") != "1":
        raise RuntimeError("remote representation lane requires BLACKBOX_REMOTE_A800_TRAIN=1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("remote representation lane requires CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("remote representation lane requires exactly one visible CUDA device")
    torch.cuda.set_device(0)
    if "A800" not in torch.cuda.get_device_name(0):
        raise RuntimeError("remote representation lane requires NVIDIA A800 GPU0")
    return torch.device("cuda:0")


def run_candidate(*, dataset: Mapping[str, Any], audit: Mapping[str, Any], dataset_path: Path, audit_path: Path, rules_path: Path, device: str, seeds: Sequence[int] = SEEDS, epochs: int = 4, learning_rate: float = 1e-4, batch_size: int = 16, config: CausalMoEConfig | None = None, checkpoint_dir: Path | None = None) -> dict[str, Any]:
    train, train_failures, train_count = _safe_context_rows(dataset, split="train")
    holdout, holdout_failures, holdout_count = _safe_context_rows(dataset, split="implementation_holdout")
    vocabulary = _build_vocabulary(dataset, train)
    train_tokens = {token for row in train for token in row["context_tokens"]}
    holdout_unknown = sorted({token for row in holdout for token in row["context_tokens"] if token not in vocabulary})
    train_context_signatures = {tuple(row["context_tokens"]) for row in train}
    holdout_context_signatures = {tuple(row["context_tokens"]) for row in holdout}
    context_overlap = len(train_context_signatures & holdout_context_signatures)
    checks = {
        "dataset_status_candidate": str(dataset.get("status")) == "candidate_only",
        "representation_candidate_allowed": dataset.get("representation_pretrain_candidate_allowed") is True,
        "capability_training_closed": dataset.get("capability_training_allowed") is False,
        "audit_passed": str(audit.get("status")) == "passed_candidate_audit",
        "train_rows_valid": bool(train) and not train_failures,
        "holdout_rows_valid": bool(holdout) and not holdout_failures,
        "holdout_vocabulary_closed": not holdout_unknown,
        "active_overlap_zero": context_overlap == 0 and int((audit.get("counts") or {}).get("active_cross_split_exact_overlap", -1)) == 0,
    }
    failures = sorted(set([key for key, ok in checks.items() if not ok] + train_failures + holdout_failures))
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked_representation_contract" if failures else "representation_pretrain_candidate_only",
        "gate": {"checks": checks, "failures": failures, "training_allowed": not failures},
        "data": {"train_rows": len(train), "holdout_rows": len(holdout), "source_train_count": train_count, "source_holdout_count": holdout_count, "train_context_vocab_size": len(train_tokens), "vocabulary_size": len(vocabulary), "vocabulary_scope": "train_context_only", "holdout_unknown_context_count": len(holdout_unknown), "active_context_overlap_count": context_overlap},
        "training": {"device": device, "seeds": [int(seed) for seed in seeds], "epochs": int(epochs), "learning_rate": float(learning_rate), "batch_size": int(batch_size), "context_only": True, "target_tokens_read": False, "capability_training": False},
        "locks": {"dataset_sha256": _sha(dataset_path), "audit_sha256": _sha(audit_path), "rules_sha256": _sha(rules_path), "runner_sha256": _sha(Path(__file__)), "model_sha256": _sha(ROOT / "app" / "pg295_causal_moe.py")},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "execution": {"optimizer_started": False, "gpu_touched": False, "docker_started": False, "network_used": False, "checkpoint_written": False},
    }
    if failures:
        return result
    torch_device = _device_gate(device)
    max_length = max(len(row["context_tokens"]) for row in [*train, *holdout])
    effective = config or CausalMoEConfig(d_model=384 if device != "cpu" else 32, n_heads=4 if device != "cpu" else 2, n_layers=6 if device != "cpu" else 1, experts=4 if device != "cpu" else 2, expert_hidden=1024 if device != "cpu" else 64, max_length=max_length)
    if effective.max_length < max_length:
        raise ValueError("representation max_length is below full context window")
    candidates: list[dict[str, Any]] = []
    for seed in seeds:
        torch.manual_seed(int(seed)); random.seed(int(seed))
        model = CausalMoELanguageModel(vocab_size=len(vocabulary), config=effective).to(torch_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=0.01)
        order = list(range(len(train)))
        for epoch in range(max(1, int(epochs))):
            random.Random(int(seed) + epoch).shuffle(order)
            model.train()
            for start in range(0, len(order), max(1, int(batch_size))):
                batch = [train[index] for index in order[start : start + max(1, int(batch_size))]]
                ids, valid = _encode(batch, vocabulary, torch_device)
                if ids.shape[1] < 2:
                    continue
                logits, balance = model(ids[:, :-1], valid_mask=valid[:, :-1])
                labels = ids[:, 1:]
                mask = valid[:, 1:]
                loss = F.cross_entropy(logits[mask], labels[mask]) + 0.01 * balance
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        train_metrics = _metrics(model, train, vocabulary, torch_device, batch_size=batch_size)
        holdout_metrics = _metrics(model, holdout, vocabulary, torch_device, batch_size=batch_size)
        checkpoint = {"path": None, "sha256": None}
        if checkpoint_dir is not None:
            path = checkpoint_dir / f"pg375_context_seed_{int(seed)}.pt"; path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"schema_version": SCHEMA_VERSION, "seed": int(seed), "config": dict(effective.__dict__), "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "vocabulary": dict(vocabulary), "context_only": True, "promotion": result["promotion"]}, path)
            checkpoint = {"path": str(path), "sha256": _sha(path)}
        candidates.append({"seed": int(seed), "train": train_metrics, "holdout": holdout_metrics, "checkpoint": checkpoint})
    result["candidates"] = candidates
    result["execution"] = {"optimizer_started": True, "gpu_touched": device != "cpu", "docker_started": False, "network_used": False, "checkpoint_written": bool(checkpoint_dir)}
    result["training"]["required_context_window"] = max_length
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="PG-375 context-only representation candidate")
    parser.add_argument("--dataset", type=Path, default=ROOT / "research" / "pg375_strict_filtered_rule_ir_dataset_v1.json")
    parser.add_argument("--audit", type=Path, default=ROOT / "research" / "pg375_strict_filtered_rule_ir_audit_v1.json")
    parser.add_argument("--rules", type=Path, default=ROOT / "research" / "improvement_rules.json")
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "pg375_context_representation_candidate_v1.json")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--epochs", type=int, default=4); parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--learning-rate", type=float, default=1e-4); parser.add_argument("--cpu-smoke", action="store_true"); parser.add_argument("--smoke-rows", type=int, default=32, help="bounded rows per split for CPU wiring smoke"); parser.add_argument("--remote-candidate", action="store_true"); parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.cpu_smoke and args.remote_candidate: parser.error("--cpu-smoke and --remote-candidate are mutually exclusive")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8-sig")); audit = json.loads(args.audit.read_text(encoding="utf-8-sig"))
    if args.cpu_smoke:
        args.device, args.epochs, args.batch_size = "cpu", 1, min(2, int(args.batch_size))
        # Validate the complete artifact first, then use a bounded subset for
        # the local wiring check.  Subsetting after the audit cannot introduce
        # a hidden holdout token or overlap, and the report remains diagnostic.
        limit = max(1, int(args.smoke_rows))
        records = dataset.get("records") if isinstance(dataset, Mapping) else None
        if isinstance(records, list):
            train_records = [row for row in records if isinstance(row, Mapping) and str(row.get("split", "")) == "train"]
            train_tokens = {str(token) for row in train_records for token in (row.get("context_tokens") or []) if isinstance(row.get("context_tokens"), list)}
            holdout_records = [
                row for row in records
                if isinstance(row, Mapping)
                and str(row.get("split", "")) == "implementation_holdout"
                and isinstance(row.get("context_tokens"), list)
                and {str(token) for token in row["context_tokens"]}.issubset(train_tokens)
            ][:limit]
            dataset = {**dataset, "records": [*train_records, *holdout_records]}
    if args.remote_candidate: args.device = "cuda:0"
    result = run_candidate(dataset=dataset, audit=audit, dataset_path=args.dataset, audit_path=args.audit, rules_path=args.rules, device=args.device, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, checkpoint_dir=args.checkpoint_dir)
    result["report_sha256"] = _sha_json(result); args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result if args.json else {"status": result["status"], "report_sha256": result["report_sha256"]}, ensure_ascii=False)); return 0 if result["status"] != "blocked_representation_contract" else 2


if __name__ == "__main__": raise SystemExit(main())
