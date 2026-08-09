"""Token-model candidate for PG-387 CTF-like frontend contexts.

The model reads only projected JS-context tokens and predicts abstract
Rule-IR decisions.  It never loads source text, concrete canaries, URLs,
response bodies or evaluator answers.  Default mode is plan-only; the bounded
CPU smoke is a representation/decision wiring diagnostic and keeps all
promotion flags closed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg293_failure_next_action import PAD, UNK  # noqa: E402
from app.pg295_causal_moe import CausalMoEConfig, CausalMoELanguageModel  # noqa: E402


SCHEMA_VERSION = "pg387-ctf-context-token-candidate-v1"
DEFAULT_DATASET = ROOT / "research" / "pg387_ctf_frontend_context_dataset_v1.json"
HEADS = ("next_action", "repair_action", "probe_variant_ref", "ask_reason", "safe_to_send")
PROMOTION = {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}
FORBIDDEN = ("http://", "https://", "javascript:", "<script", "wire=", "payload=", "response_body=", "oracle_answer=")


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("PG-387 dataset root must be object")
    return value


def _target_values(tokens: Sequence[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for token in tokens:
        text = str(token)
        if "=" in text:
            key, value = text.split("=", 1)
            if key in HEADS:
                values[key] = value
    missing = [key for key in HEADS if key not in values]
    if missing:
        raise ValueError(f"PG-387 target missing {','.join(missing)}")
    return values


def _safe_rows(dataset: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if dataset.get("status") != "abstract_ctf_candidate_only":
        raise ValueError("PG-387 dataset status mismatch")
    rows: list[dict[str, Any]] = []
    for raw in dataset.get("rows", []):
        if not isinstance(raw, Mapping):
            raise ValueError("PG-387 row must be object")
        context = raw.get("context_tokens")
        target = raw.get("target_tokens")
        if not isinstance(context, list) or not isinstance(target, list):
            raise ValueError("PG-387 context/target must be lists")
        if raw.get("source_text_stored") is not False or raw.get("typed_evaluator_observed") is not False:
            raise ValueError("PG-387 raw/evaluator gate open")
        context_text = [str(item) for item in context]
        target_text = [str(item) for item in target]
        if any(any(fragment in token.casefold() for fragment in FORBIDDEN) for token in [*context_text, *target_text]):
            raise ValueError("PG-387 forbidden token reached loader")
        _target_values(target_text)
        rows.append({"context_tokens": context_text, "target_tokens": target_text, "split": str(raw.get("split", ""))})
    train = [row for row in rows if row["split"] == "train"]
    holdout = [row for row in rows if row["split"] == "implementation_holdout"]
    if not train or not holdout:
        raise ValueError("PG-387 train/holdout split empty")
    return train, holdout


def _build_train_vocab(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    tokens = {PAD, UNK}
    for row in rows:
        tokens.update(str(token) for token in row["context_tokens"])
    ordered = [PAD, UNK, *sorted(tokens - {PAD, UNK})]
    return {token: index for index, token in enumerate(ordered)}


def _classes(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    values = {key: set() for key in HEADS}
    for row in rows:
        labels = _target_values(row["target_tokens"])
        for key in HEADS:
            values[key].add(labels[key])
    return {key: {value: index for index, value in enumerate(sorted(items))} for key, items in values.items()}


def _gaps(train: Sequence[Mapping[str, Any]], holdout: Sequence[Mapping[str, Any]], vocab: Mapping[str, int], classes: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    unknown_tokens = sorted({token for row in holdout for token in row["context_tokens"] if token not in vocab})
    unknown_heads = {key: sorted({_target_values(row["target_tokens"])[key] for row in holdout if _target_values(row["target_tokens"])[key] not in classes[key]}) for key in HEADS}
    unknown_heads = {key: value for key, value in unknown_heads.items() if value}
    return {"unknown_context_count": len(unknown_tokens), "unknown_context_sha256": _sha(unknown_tokens), "unknown_head_value_count": sum(len(value) for value in unknown_heads.values()), "unknown_head_values_sha256": {key: _sha(value) for key, value in unknown_heads.items()}, "blocked": bool(unknown_tokens or unknown_heads)}


class ContextDecisionModel(nn.Module):
    def __init__(self, *, vocab_size: int, config: CausalMoEConfig, classes: Mapping[str, Mapping[str, int]]) -> None:
        super().__init__()
        self.backbone = CausalMoELanguageModel(vocab_size=vocab_size, config=config)
        self.heads = nn.ModuleDict({key: nn.Linear(config.d_model, len(values)) for key, values in classes.items()})

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden, _balance = self.backbone.forward_hidden(ids, valid_mask=mask)
        lengths = mask.long().sum(dim=1).clamp_min(1) - 1
        boundary = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
        return {key: head(boundary) for key, head in self.heads.items()}


def _pad(rows: Sequence[Mapping[str, Any]], vocab: Mapping[str, int], device: torch.device, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = [[int(vocab.get(token, vocab[UNK])) for token in row["context_tokens"]][:max_length] for row in rows]
    width = max((len(sequence) for sequence in sequences), default=1)
    ids = torch.full((len(sequences), width), int(vocab[PAD]), dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), width), dtype=torch.bool, device=device)
    for index, sequence in enumerate(sequences):
        ids[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[index, :len(sequence)] = True
    return ids, mask


def _labels(rows: Sequence[Mapping[str, Any]], classes: Mapping[str, Mapping[str, int]], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.tensor([classes[key][_target_values(row["target_tokens"])[key]] for row in rows], dtype=torch.long, device=device) for key in HEADS}


def _predict(model: ContextDecisionModel, rows: Sequence[Mapping[str, Any]], vocab: Mapping[str, int], classes: Mapping[str, Mapping[str, int]], device: torch.device, max_length: int) -> list[dict[str, str]]:
    reverse = {key: {index: value for value, index in mapping.items()} for key, mapping in classes.items()}
    predictions: list[dict[str, str]] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), 64):
            batch = rows[start:start + 64]
            ids, mask = _pad(batch, vocab, device, max_length)
            output = model(ids, mask)
            for index in range(len(batch)):
                predictions.append({key: reverse[key][int(output[key][index].argmax().item())] for key in HEADS})
    return predictions


def _metrics(rows: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    totals = Counter()
    for row, prediction in zip(rows, predictions):
        expected = _target_values(row["target_tokens"])
        for key in HEADS:
            totals[f"{key}_total"] += 1
            totals[f"{key}_correct"] += int(expected[key] == prediction[key])
        ask = expected["ask_reason"] != "none"
        totals["ask_total"] += int(ask)
        totals["ask_correct"] += int(ask and prediction["ask_reason"] != "none")
        safe = expected["safe_to_send"] == "1"
        predicted_safe = prediction["safe_to_send"] == "1"
        totals["safe_total"] += int(safe)
        totals["safe_correct"] += int(safe and predicted_safe)
        totals["negative_total"] += int(not safe)
        totals["negative_false_allow"] += int((not safe) and predicted_safe)
    return {
        "rows": len(rows),
        "next_action_accuracy": round(totals["next_action_correct"] / max(totals["next_action_total"], 1), 6),
        "repair_action_accuracy": round(totals["repair_action_correct"] / max(totals["repair_action_total"], 1), 6),
        "probe_variant_accuracy": round(totals["probe_variant_ref_correct"] / max(totals["probe_variant_ref_total"], 1), 6),
        "ask_recall": round(totals["ask_correct"] / max(totals["ask_total"], 1), 6) if totals["ask_total"] else None,
        "safe_recall": round(totals["safe_correct"] / max(totals["safe_total"], 1), 6) if totals["safe_total"] else None,
        "negative_false_allow": int(totals["negative_false_allow"]),
        "negative_total": int(totals["negative_total"]),
    }


def _train_seed(train: Sequence[Mapping[str, Any]], holdout: Sequence[Mapping[str, Any]], vocab: Mapping[str, int], classes: Mapping[str, Mapping[str, int]], *, seed: int, config: CausalMoEConfig, epochs: int, microbatch: int, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = ContextDecisionModel(vocab_size=len(vocab), config=config, classes=classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    order = list(range(len(train)))
    for epoch in range(max(1, epochs)):
        random.Random(seed + epoch).shuffle(order)
        model.train()
        for start in range(0, len(order), max(1, microbatch)):
            batch = [train[index] for index in order[start:start + max(1, microbatch)]]
            ids, mask = _pad(batch, vocab, device, config.max_length)
            output = model(ids, mask)
            labels = _labels(batch, classes, device)
            loss = sum(F.cross_entropy(output[key], labels[key]) for key in HEADS)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return {"seed": seed, "train": _metrics(train, _predict(model, train, vocab, classes, device, config.max_length)), "holdout": _metrics(holdout, _predict(model, holdout, vocab, classes, device, config.max_length))}


def run_candidate(*, dataset_path: Path = DEFAULT_DATASET, cpu_smoke: bool = False, epochs: int = 1, row_limit: int | None = None, d_model: int = 64, n_layers: int = 2, experts: int = 2, expert_hidden: int = 128, max_length: int = 96, microbatch: int = 16, seeds: Sequence[int] = (38701, 38702, 38703)) -> dict[str, Any]:
    dataset = _load(dataset_path)
    train, holdout = _safe_rows(dataset)
    if row_limit is not None:
        limit = max(1, int(row_limit))
        train, holdout = train[:limit], holdout[:limit]
    vocab = _build_train_vocab(train)
    classes = _classes(train)
    gaps = _gaps(train, holdout, vocab, classes)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "cpu_smoke_candidate_only" if cpu_smoke else "plan_only",
        "dataset": str(dataset_path),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "train_only_vocabulary": {"size": len(vocab), "scope": "train_context_only"},
        "gaps": gaps,
        "model": {"backbone": "decoder_only_causal_moe", "d_model": d_model, "n_layers": n_layers, "experts": experts, "expert_hidden": expert_hidden, "max_length": max_length},
        "execution": {"optimizer_started": False, "device": "cpu", "gpu_touched": False, "docker_started": False, "network_contacted": False},
        "training_eligible": 0,
        "capability_training_allowed": False,
        "representation_candidate_only": True,
        "promotion": dict(PROMOTION),
    }
    if gaps["blocked"]:
        report["status"] = "blocked_train_only_vocab_gap"
        report["blocking_before_optimizer"] = True
        report["report_sha256"] = _sha({key: value for key, value in report.items() if key != "report_sha256"})
        return report
    if not cpu_smoke:
        report["planned_seeds"] = list(seeds)
        report["report_sha256"] = _sha({key: value for key, value in report.items() if key != "report_sha256"})
        return report
    device = torch.device("cpu")
    config = CausalMoEConfig(d_model=d_model, n_heads=max(1, min(4, d_model // 16)), n_layers=n_layers, experts=experts, expert_hidden=expert_hidden, top_k=min(2, experts), dropout=0.05, max_length=max_length)
    report["execution"]["optimizer_started"] = True
    report["seeds"] = [_train_seed(train, holdout, vocab, classes, seed=int(seed), config=config, epochs=epochs, microbatch=microbatch, device=device) for seed in seeds]
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
    parser.add_argument("--output", default="research/pg387_ctf_context_token_candidate_v1.json")
    args = parser.parse_args()
    report = run_candidate(dataset_path=Path(args.dataset), cpu_smoke=args.cpu_smoke, epochs=args.epochs, row_limit=args.row_limit, d_model=args.d_model, n_layers=args.layers, experts=args.experts, expert_hidden=args.expert_hidden, max_length=args.max_length, microbatch=args.microbatch)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"], "train_rows": report["train_rows"], "holdout_rows": report["holdout_rows"], "gaps": report["gaps"], "optimizer_started": report["execution"]["optimizer_started"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
