"""CPU-only candidate runner for PG-388 abstract logic Rule-IR decisions.

This is a bounded wiring experiment.  It trains heads over abstract context
tokens and never creates a request, loads a target, or emits a literal value.
The default CLI is plan-only; ``--cpu-smoke`` is intentionally candidate-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg293_failure_next_action import PAD, UNK  # noqa: E402
from app.pg295_causal_moe import CausalMoEConfig, CausalMoELanguageModel  # noqa: E402


SCHEMA_VERSION = "pg388-logic-token-candidate-v1"
DEFAULT_DATASET = ROOT / "research" / "pg388_logic_invariant_dataset_v1.json"
HEADS = ("question", "ask_reason", "next_action", "repair_action", "logic_invariant_ref", "state_transition_ref", "precondition_ref", "counterfactual_ref", "probe_variant_ref", "oracle_ref", "safe_to_send")
PROMOTION = {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}
FORBIDDEN = ("http://", "https://", "payload=", "wire=", "response_body=", "<script")


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict) or value.get("status") not in {"abstract_logic_candidate_only", "abstract_logic_supplement_candidate_only", "abstract_canary_trajectory_candidate_only"}:
        raise ValueError("pg388_dataset_status_mismatch")
    return value


def _labels(tokens: Sequence[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for token in tokens:
        text = str(token)
        if "=" in text:
            key, value = text.split("=", 1)
            if key in HEADS:
                values[key] = value
    missing = [key for key in HEADS if key not in values]
    if missing:
        raise ValueError("pg388_target_missing:" + ",".join(missing))
    return values


def _safe_rows(dataset: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for raw in dataset.get("rows", []):
        if not isinstance(raw, Mapping) or raw.get("training_eligible") is not False or raw.get("raw_source_stored") is not False or raw.get("oracle_answer_in_context") is not False:
            raise ValueError("pg388_row_firewall_open")
        context = [str(item) for item in raw.get("context_tokens", [])]
        target = [str(item) for item in raw.get("target_tokens", [])]
        if not context or not target or any(any(marker in token.casefold() for marker in FORBIDDEN) for token in [*context, *target]):
            raise ValueError("pg388_forbidden_or_empty_tokens")
        _labels(target)
        rows.append({"context_tokens": context, "target_tokens": target, "split": str(raw.get("split", ""))})
    train = [row for row in rows if row["split"] == "train"]
    holdout = [row for row in rows if row["split"] == "implementation_holdout"]
    if not train or not holdout:
        raise ValueError("pg388_split_empty")
    return train, holdout


def _vocab(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    tokens = {PAD, UNK}
    for row in rows:
        tokens.update(str(token) for token in row["context_tokens"])
    ordered = [PAD, UNK, *sorted(tokens - {PAD, UNK})]
    return {token: index for index, token in enumerate(ordered)}


def _classes(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    values = {key: set() for key in HEADS}
    for row in rows:
        labels = _labels(row["target_tokens"])
        for key in HEADS:
            values[key].add(labels[key])
    return {key: {value: index for index, value in enumerate(sorted(items))} for key, items in values.items()}


def _gaps(train: Sequence[Mapping[str, Any]], holdout: Sequence[Mapping[str, Any]], vocab: Mapping[str, int], classes: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    unknown_context = sorted({token for row in holdout for token in row["context_tokens"] if token not in vocab})
    unknown_heads = {key: sorted({_labels(row["target_tokens"])[key] for row in holdout if _labels(row["target_tokens"])[key] not in classes[key]}) for key in HEADS}
    unknown_heads = {key: values for key, values in unknown_heads.items() if values}
    return {"unknown_context_count": len(unknown_context), "unknown_context_sha256": _sha(unknown_context), "unknown_head_value_count": sum(len(values) for values in unknown_heads.values()), "unknown_head_values_sha256": {key: _sha(values) for key, values in unknown_heads.items()}, "blocked": bool(unknown_context or unknown_heads)}


class LogicDecisionModel(nn.Module):
    def __init__(self, *, vocab_size: int, config: CausalMoEConfig, classes: Mapping[str, Mapping[str, int]], pooling: str = "boundary") -> None:
        super().__init__()
        if pooling not in {"boundary", "mean", "mean_boundary", "anchor_mean_boundary"}:
            raise ValueError("pg388_pooling_invalid")
        self.backbone = CausalMoELanguageModel(vocab_size=vocab_size, config=config)
        self.pooling = pooling
        projection_width = config.d_model * 4 if pooling == "anchor_mean_boundary" else config.d_model * 2
        self.pool_projection = nn.Linear(projection_width, config.d_model) if pooling in {"mean_boundary", "anchor_mean_boundary"} else None
        self.heads = nn.ModuleDict({key: nn.Linear(config.d_model, len(values)) for key, values in classes.items()})

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden, _ = self.backbone.forward_hidden(ids, valid_mask=mask)
        lengths = mask.long().sum(dim=1).clamp_min(1) - 1
        boundary = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
        if self.pooling == "mean":
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            features = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        elif self.pooling == "mean_boundary":
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            mean = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            assert self.pool_projection is not None
            features = F.gelu(self.pool_projection(torch.cat([mean, boundary], dim=-1)))
        elif self.pooling == "anchor_mean_boundary":
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            mean = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            first = hidden[:, 0]
            second = hidden[:, 1] if hidden.shape[1] > 1 else first
            assert self.pool_projection is not None
            features = F.gelu(self.pool_projection(torch.cat([first, second, mean, boundary], dim=-1)))
        else:
            features = boundary
        return {key: head(features) for key, head in self.heads.items()}


def _pad(rows: Sequence[Mapping[str, Any]], vocab: Mapping[str, int], device: torch.device, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = [[int(vocab.get(token, vocab[UNK])) for token in row["context_tokens"]][:max_length] for row in rows]
    width = max((len(sequence) for sequence in sequences), default=1)
    ids = torch.full((len(sequences), width), int(vocab[PAD]), dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), width), dtype=torch.bool, device=device)
    for index, sequence in enumerate(sequences):
        ids[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[index, :len(sequence)] = True
    return ids, mask


def _label_tensors(rows: Sequence[Mapping[str, Any]], classes: Mapping[str, Mapping[str, int]], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.tensor([classes[key][_labels(row["target_tokens"])[key]] for row in rows], dtype=torch.long, device=device) for key in HEADS}


def _metrics(model: LogicDecisionModel, rows: Sequence[Mapping[str, Any]], vocab: Mapping[str, int], classes: Mapping[str, Mapping[str, int]], device: torch.device, max_length: int) -> dict[str, Any]:
    reverse = {key: {index: value for value, index in mapping.items()} for key, mapping in classes.items()}
    correct = Counter()
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), 64):
            batch = rows[start:start + 64]
            ids, mask = _pad(batch, vocab, device, max_length)
            output = model(ids, mask)
            for index, row in enumerate(batch):
                expected = _labels(row["target_tokens"])
                predicted = {key: reverse[key][int(output[key][index].argmax().item())] for key in HEADS}
                for key in HEADS:
                    correct[key] += int(expected[key] == predicted[key])
                correct["ask_total"] += int(expected["ask_reason"] != "none")
                correct["ask_correct"] += int(expected["ask_reason"] != "none" and predicted["ask_reason"] != "none")
                safe = expected["safe_to_send"] == "true"
                correct["negative_total"] += int(not safe)
                correct["negative_false_allow"] += int((not safe) and predicted["safe_to_send"] == "true")
    return {"rows": len(rows), "head_accuracy": {key: round(correct[key] / max(len(rows), 1), 6) for key in HEADS}, "ask_recall": round(correct["ask_correct"] / max(correct["ask_total"], 1), 6), "negative_false_allow": int(correct["negative_false_allow"]), "negative_total": int(correct["negative_total"])}


def _train_seed(train: Sequence[Mapping[str, Any]], holdout: Sequence[Mapping[str, Any]], vocab: Mapping[str, int], classes: Mapping[str, Mapping[str, int]], *, seed: int, config: CausalMoEConfig, epochs: int, microbatch: int, pooling: str) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device("cpu")
    model = LogicDecisionModel(vocab_size=len(vocab), config=config, classes=classes, pooling=pooling).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    order = list(range(len(train)))
    for epoch in range(max(1, epochs)):
        random.Random(seed + epoch).shuffle(order)
        model.train()
        for start in range(0, len(order), max(1, microbatch)):
            batch = [train[index] for index in order[start:start + max(1, microbatch)]]
            ids, mask = _pad(batch, vocab, device, config.max_length)
            output = model(ids, mask)
            labels = _label_tensors(batch, classes, device)
            loss = sum(F.cross_entropy(output[key], labels[key]) for key in HEADS)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return {"seed": seed, "train": _metrics(model, train, vocab, classes, device, config.max_length), "holdout": _metrics(model, holdout, vocab, classes, device, config.max_length)}


def run_candidate(*, dataset_path: Path = DEFAULT_DATASET, cpu_smoke: bool = False, epochs: int = 1, row_limit: int | None = None, d_model: int = 64, n_layers: int = 2, experts: int = 2, expert_hidden: int = 128, max_length: int = 96, microbatch: int = 16, pooling: str = "boundary", seeds: Sequence[int] = (38801, 38802, 38803)) -> dict[str, Any]:
    dataset = _load(dataset_path)
    train, holdout = _safe_rows(dataset)
    if row_limit is not None:
        limit = max(1, int(row_limit))
        train, holdout = train[:limit], holdout[:limit]
    vocab = _vocab(train)
    classes = _classes(train)
    gaps = _gaps(train, holdout, vocab, classes)
    report: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "status": "cpu_smoke_candidate_only" if cpu_smoke else "plan_only", "dataset": str(dataset_path), "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(), "train_rows": len(train), "holdout_rows": len(holdout), "train_only_vocabulary": {"size": len(vocab), "scope": "train_context_only"}, "gaps": gaps, "heads": list(HEADS), "model": {"backbone": "decoder_only_causal_moe", "d_model": d_model, "n_layers": n_layers, "experts": experts, "expert_hidden": expert_hidden, "max_length": max_length, "pooling": pooling}, "execution": {"optimizer_started": False, "device": "cpu", "gpu_touched": False, "docker_started": False, "network_contacted": False, "wire_created": False}, "training_eligible": 0, "capability_training_allowed": False, "logic_candidate_only": True, "promotion": dict(PROMOTION)}
    if gaps["blocked"]:
        report["status"] = "blocked_train_only_vocab_gap"
        report["blocking_before_optimizer"] = True
    elif cpu_smoke:
        config = CausalMoEConfig(d_model=d_model, n_heads=max(1, min(4, d_model // 16)), n_layers=n_layers, experts=experts, expert_hidden=expert_hidden, top_k=min(2, experts), dropout=0.05, max_length=max_length)
        report["execution"]["optimizer_started"] = True
        report["seeds"] = [_train_seed(train, holdout, vocab, classes, seed=int(seed), config=config, epochs=epochs, microbatch=microbatch, pooling=pooling) for seed in seeds]
    else:
        report["planned_seeds"] = list(seeds)
    report["report_sha256"] = _sha({key: value for key, value in report.items() if key != "report_sha256"})
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--row-limit", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--experts", type=int, default=2)
    parser.add_argument("--expert-hidden", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--pooling", choices=("boundary", "mean", "mean_boundary", "anchor_mean_boundary"), default="boundary")
    parser.add_argument("--output", default="research/pg388_logic_token_candidate_v1.json")
    args = parser.parse_args()
    report = run_candidate(dataset_path=Path(args.dataset), cpu_smoke=args.cpu_smoke, epochs=args.epochs, row_limit=args.row_limit, d_model=args.d_model, n_layers=args.layers, experts=args.experts, expert_hidden=args.expert_hidden, max_length=args.max_length, microbatch=args.microbatch, pooling=args.pooling)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"], "train_rows": report["train_rows"], "holdout_rows": report["holdout_rows"], "gaps": report["gaps"], "optimizer_started": report["execution"]["optimizer_started"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
