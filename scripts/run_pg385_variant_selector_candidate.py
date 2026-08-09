"""Train/evaluate an abstract PG-385 repair-variant selector.

This is the missing model-facing step between the filter-feedback demo and the
reviewed evaluator binder.  The decoder-only MoE backbone sees only abstract
context tokens.  Small heads select ``encoding_ref``, ``probe_variant_ref``,
``repair_action``, ASK and safe/negative state.  No literal canary, URL,
response body, or evaluator answer is loaded.  A passing candidate still only
authorizes an abstract variant reference; a separate local evaluator may bind
an inert, reviewed canary at the last hop.

Default mode is plan-only.  ``--cpu-smoke`` is a bounded wiring check.  The
remote lane requires the weekend/explicit-flag/GPU0/A800 gate and remains
candidate-only with all promotion flags false.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
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
from scripts.plan_pg369_multitask_moe_candidate import SLOTS, derive_multitask_labels  # noqa: E402


SCHEMA_VERSION = "pg385-abstract-variant-selector-candidate-v1"
DEFAULT_DATASET = ROOT / "research/pg385_filter_repair_adversarial_dataset_v1.json"
SEEDS = (38501, 38502, 38503)
TARGET_HEADS = ("encoding_ref", "probe_variant_ref", "repair_action", "next_action", "question", "safe_to_send")
PROMOTION = {
    "training_allowed": False,
    "memory_promotion_allowed": False,
    "payload_catalog_promotion_allowed": False,
    "vulnerability_claim_allowed": False,
}
FORBIDDEN = ("http://", "https://", "javascript:", "<script", "wire=", "payload=", "response_body=", "oracle_answer=")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("PG-385 dataset root must be an object")
    return value


def _safe_rows(dataset: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if dataset.get("status") != "abstract_adversarial_candidate_only":
        raise ValueError("PG-385 dataset is not candidate-only")
    if dataset.get("safety", {}).get("raw_payload_in_context") is not False:
        raise ValueError("PG-385 raw payload gate is open")
    rows: list[dict[str, Any]] = []
    for raw in dataset.get("records", []):
        if not isinstance(raw, Mapping):
            raise ValueError("PG-385 record must be an object")
        if raw.get("raw_payload_stored") is not False or raw.get("raw_response_body_stored") is not False or raw.get("oracle_answer_in_context") is not False:
            raise ValueError("PG-385 raw/evaluator field reached model loader")
        if raw.get("context_firewall") != {"forbidden_token_count": 0, "sidecars_off_context": True}:
            raise ValueError("PG-385 context firewall is incomplete")
        context = raw.get("context_tokens")
        target = raw.get("target_tokens")
        if not isinstance(context, list) or not isinstance(target, list):
            raise ValueError("PG-385 context/target must be lists")
        target_text = [str(token) for token in target]
        if derive_multitask_labels(target_text) is None:
            raise ValueError("PG-385 target does not satisfy 13-slot contract")
        all_tokens = [str(token) for token in [*context, *target]]
        if any(any(fragment in token.casefold() for fragment in FORBIDDEN) for token in all_tokens):
            raise ValueError("PG-385 raw/evaluator marker reached model row")
        rows.append({
            "context_tokens": [str(token) for token in context],
            "target_tokens": target_text,
            "split": str(raw.get("split", "")),
        })
    train = [row for row in rows if row["split"] == "train"]
    holdout = [row for row in rows if row["split"] == "implementation_holdout"]
    if not train or not holdout:
        raise ValueError("PG-385 train/implementation_holdout split is empty")
    return train, holdout


def _target_values(tokens: Sequence[str]) -> dict[str, str]:
    labels = derive_multitask_labels(tokens)
    if labels is None:
        raise ValueError("target does not satisfy Rule-IR slots")
    return dict(labels["next_token"])


def _labels(row: Mapping[str, Any]) -> dict[str, str]:
    values = _target_values(row["target_tokens"])
    # Keep the head set intentionally small; the full 13-slot Rule-IR remains
    # the source contract, while this adapter learns the repair decision first.
    return {key: values[key] for key in TARGET_HEADS}


def _build_train_vocab(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    tokens = {PAD, UNK}
    for row in rows:
        tokens.update(str(token) for token in row["context_tokens"])
    return {token: index for index, token in enumerate([PAD, UNK, *sorted(tokens - {PAD, UNK})])}


def _classes(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    values: dict[str, set[str]] = {key: set() for key in TARGET_HEADS}
    for row in rows:
        labels = _labels(row)
        for key in TARGET_HEADS:
            values[key].add(labels[key])
    return {key: {item: index for index, item in enumerate(sorted(items))} for key, items in values.items()}


def _gaps(train: Sequence[Mapping[str, Any]], holdout: Sequence[Mapping[str, Any]], vocab: Mapping[str, int], classes: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    unknown_tokens = sorted({str(token) for row in holdout for token in row["context_tokens"] if str(token) not in vocab})
    unknown_heads: dict[str, list[str]] = {}
    for key in TARGET_HEADS:
        values = sorted({_labels(row)[key] for row in holdout if _labels(row)[key] not in classes[key]})
        if values:
            unknown_heads[key] = values
    return {
        "unknown_context_count": len(unknown_tokens),
        "unknown_context_sha256": _sha_json(unknown_tokens),
        "unknown_head_value_count": sum(len(items) for items in unknown_heads.values()),
        "unknown_head_values": {key: _sha_json(values) for key, values in sorted(unknown_heads.items())},
        "blocked": bool(unknown_tokens or unknown_heads),
    }


def _gate(device: str) -> dict[str, Any]:
    now = dt.datetime.now(ZoneInfo("Asia/Shanghai"))
    weekend = now.weekday() >= 5
    explicit = os.environ.get("BLACKBOX_REMOTE_A800_TRAIN") == "1"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return {
        "timestamp": now.isoformat(),
        "weekend": weekend,
        "explicit_remote_flag": explicit,
        "cuda_visible_devices": visible,
        "device": device,
        "passed": device == "cpu" or (weekend and explicit and visible == "0"),
    }


class VariantSelector(nn.Module):
    """Decoder-only backbone with abstract repair-decision heads."""

    def __init__(self, *, vocab_size: int, config: CausalMoEConfig, classes: Mapping[str, Mapping[str, int]]) -> None:
        super().__init__()
        self.backbone = CausalMoELanguageModel(vocab_size=vocab_size, config=config)
        self.heads = nn.ModuleDict({key: nn.Linear(config.d_model, len(values)) for key, values in classes.items()})

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden, _balance = self.backbone.forward_hidden(ids, valid_mask=mask)
        lengths = mask.long().sum(dim=1).clamp_min(1) - 1
        boundary = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
        return {key: head(boundary) for key, head in self.heads.items()}


def _pad(rows: Sequence[Mapping[str, Any]], vocab: Mapping[str, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = [[int(vocab.get(token, vocab[UNK])) for token in row["context_tokens"]] for row in rows]
    width = max((len(item) for item in sequences), default=1)
    ids = torch.full((len(sequences), width), int(vocab[PAD]), dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), width), dtype=torch.bool, device=device)
    for index, sequence in enumerate(sequences):
        ids[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[index, :len(sequence)] = True
    return ids, mask


def _batch_targets(rows: Sequence[Mapping[str, Any]], classes: Mapping[str, Mapping[str, int]], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.tensor([classes[key][_labels(row)[key]] for row in rows], dtype=torch.long, device=device) for key in TARGET_HEADS}


def _predict(model: VariantSelector, rows: Sequence[Mapping[str, Any]], vocab: Mapping[str, int], classes: Mapping[str, Mapping[str, int]], device: torch.device) -> list[dict[str, str]]:
    reverse = {key: {index: value for value, index in items.items()} for key, items in classes.items()}
    model.eval()
    predictions: list[dict[str, str]] = []
    with torch.inference_mode():
        for start in range(0, len(rows), 64):
            batch = rows[start:start + 64]
            ids, mask = _pad(batch, vocab, device)
            output = model(ids, mask)
            for index in range(len(batch)):
                predictions.append({key: reverse[key][int(output[key][index].argmax().item())] for key in TARGET_HEADS})
    return predictions


def _metrics(rows: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    totals = Counter()
    for row, prediction in zip(rows, predictions):
        expected = _labels(row)
        for key in TARGET_HEADS:
            totals[f"{key}_total"] += 1
            totals[f"{key}_correct"] += int(prediction[key] == expected[key])
        expected_ask = expected["question"].startswith("ask_")
        predicted_ask = prediction["question"].startswith("ask_")
        totals["ask_total"] += int(expected_ask)
        totals["ask_correct"] += int(expected_ask and predicted_ask)
        expected_repair = expected["next_action"] == "repair"
        predicted_repair = prediction["next_action"] == "repair"
        totals["repair_total"] += int(expected_repair)
        totals["repair_correct"] += int(expected_repair and predicted_repair)
        expected_safe = expected["safe_to_send"] == "1"
        predicted_safe = prediction["safe_to_send"] == "1"
        totals["positive_total"] += int(expected_safe)
        totals["positive_correct"] += int(expected_safe and predicted_safe)
        totals["negative_total"] += int(not expected_safe)
        totals["negative_false_allow"] += int((not expected_safe) and predicted_safe)
        totals["variant_exact_total"] += 1
        totals["variant_exact"] += int(prediction["probe_variant_ref"] == expected["probe_variant_ref"] and prediction["encoding_ref"] == expected["encoding_ref"])
    result = {
        "rows": len(rows),
        "variant_exact": round(totals["variant_exact"] / max(totals["variant_exact_total"], 1), 6),
        "encoding_exact": round(totals["encoding_ref_correct"] / max(totals["encoding_ref_total"], 1), 6),
        "probe_variant_exact": round(totals["probe_variant_ref_correct"] / max(totals["probe_variant_ref_total"], 1), 6),
        "repair_action_accuracy": round(totals["repair_action_correct"] / max(totals["repair_action_total"], 1), 6),
        "next_action_accuracy": round(totals["next_action_correct"] / max(totals["next_action_total"], 1), 6),
        "ask_recall": round(totals["ask_correct"] / max(totals["ask_total"], 1), 6) if totals["ask_total"] else None,
        "positive_recall": round(totals["positive_correct"] / max(totals["positive_total"], 1), 6) if totals["positive_total"] else None,
        "negative_false_allow": int(totals["negative_false_allow"]),
        "negative_total": int(totals["negative_total"]),
    }
    return result


def _train_seed(train: Sequence[Mapping[str, Any]], holdout: Sequence[Mapping[str, Any]], vocab: Mapping[str, int], classes: Mapping[str, Mapping[str, int]], *, seed: int, config: CausalMoEConfig, epochs: int, microbatch: int, device: torch.device, checkpoint: Path | None) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    model = VariantSelector(vocab_size=len(vocab), config=config, classes=classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    order = list(range(len(train)))
    for epoch in range(max(1, int(epochs))):
        random.Random(int(seed) + epoch).shuffle(order)
        model.train()
        for start in range(0, len(order), max(1, int(microbatch))):
            batch = [train[index] for index in order[start:start + max(1, int(microbatch))]]
            ids, mask = _pad(batch, vocab, device)
            output = model(ids, mask)
            labels = _batch_targets(batch, classes, device)
            loss = (
                3.0 * F.cross_entropy(output["encoding_ref"], labels["encoding_ref"])
                + 3.0 * F.cross_entropy(output["probe_variant_ref"], labels["probe_variant_ref"])
                + 2.0 * F.cross_entropy(output["repair_action"], labels["repair_action"])
                + 2.0 * F.cross_entropy(output["next_action"], labels["next_action"])
                + 2.0 * F.cross_entropy(output["question"], labels["question"])
                + 2.0 * F.cross_entropy(output["safe_to_send"], labels["safe_to_send"])
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    metrics = _metrics(holdout, _predict(model, holdout, vocab, classes, device))
    train_metrics = _metrics(train, _predict(model, train, vocab, classes, device))
    checkpoint_sha = None
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema_version": SCHEMA_VERSION, "seed": int(seed), "config": dict(config.__dict__), "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "vocabulary": dict(vocab), "classes": {key: dict(value) for key, value in classes.items()}}, checkpoint)
        checkpoint_sha = _sha_file(checkpoint)
    return {"seed": int(seed), "train": train_metrics, "holdout": metrics, "checkpoint": {"path": str(checkpoint) if checkpoint else None, "sha256": checkpoint_sha}}


def run_candidate(*, dataset_path: Path = DEFAULT_DATASET, device: str = "cpu", epochs: int = 1, microbatch: int = 8, d_model: int = 128, n_layers: int = 3, experts: int = 4, expert_hidden: int = 512, max_length: int = 128, row_limit: int | None = None, checkpoint_dir: Path | None = None) -> dict[str, Any]:
    dataset = _load(dataset_path)
    train, holdout = _safe_rows(dataset)
    if row_limit is not None:
        limit = max(1, int(row_limit))
        train, holdout = train[:limit], holdout[:limit]
    vocab = _build_train_vocab(train)
    classes = _classes(train)
    gaps = _gaps(train, holdout, vocab, classes)
    if gaps["blocked"]:
        raise ValueError("strict train-only variant vocabulary is not closed")
    gate = _gate(device)
    if device != "cpu":
        if not gate["passed"]:
            raise RuntimeError("remote PG-385 variant lane gate failed")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "A800" not in torch.cuda.get_device_name(0):
            raise RuntimeError("remote PG-385 variant lane requires one visible A800")
        torch.cuda.set_device(0)
    torch_device = torch.device(device)
    config = CausalMoEConfig(d_model=int(d_model), n_heads=4 if int(d_model) < 256 else 8, n_layers=int(n_layers), experts=int(experts), expert_hidden=int(expert_hidden), max_length=int(max_length), top_k=min(2, int(experts)))
    if int(max_length) < max((len(row["context_tokens"]) for row in [*train, *holdout]), default=1):
        raise ValueError("variant selector max_length is below context window")
    candidates = [_train_seed(train, holdout, vocab, classes, seed=int(seed), config=config, epochs=epochs, microbatch=microbatch, device=torch_device, checkpoint=(checkpoint_dir / f"pg385_variant_seed_{int(seed)}.pt" if checkpoint_dir else None)) for seed in SEEDS]
    worst = {
        "variant_exact_min": min(float(item["holdout"]["variant_exact"]) for item in candidates),
        "encoding_exact_min": min(float(item["holdout"]["encoding_exact"]) for item in candidates),
        "ask_recall_min": min(float(item["holdout"]["ask_recall"] or 0.0) for item in candidates),
        "positive_recall_min": min(float(item["holdout"]["positive_recall"] or 0.0) for item in candidates),
        "negative_false_allow_max": max(int(item["holdout"]["negative_false_allow"]) for item in candidates),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "abstract_variant_selector_candidate_only",
        "dataset": str(dataset_path),
        "dataset_sha256": _sha_file(dataset_path),
        "data": {"train_rows": len(train), "holdout_rows": len(holdout), "vocabulary_scope": "train_context_only", "unknown_gaps": gaps, "target_heads": list(TARGET_HEADS), "raw_rows_loaded": False},
        "execution_gate": gate,
        "training": {"device": device, "seeds": list(SEEDS), "epochs": int(epochs), "microbatch": int(microbatch), "config": dict(config.__dict__), "vocabulary_size": len(vocab), "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None},
        "candidates": candidates,
        "worst_seed": worst,
        "evaluator_binding": {"reviewed_local_canary_last_hop": True, "model_emits_raw_string": False, "model_emits_variant_reference": True, "candidate_wire_replay": False},
        "scientific_gate": {"variant_gate_passed": bool(worst["variant_exact_min"] >= 0.90 and worst["encoding_exact_min"] >= 0.90), "ask_gate_passed": bool(worst["ask_recall_min"] >= 0.95), "negative_gate_passed": bool(worst["negative_false_allow_max"] == 0), "claim_allowed": False},
        "promotion": dict(PROMOTION),
        "interpretation": "仅证明抽象反馈到受控变体引用的候选能力；具体字符串仍由固定本地 evaluator 最后一跳绑定，不能迁移到任意目标。",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=ROOT / "research/pg385_variant_selector_candidate_v1.json")
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--remote-candidate", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--expert-hidden", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--row-limit", type=int)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "artifacts/pg385-variant-selector")
    args = parser.parse_args()
    if args.cpu_smoke and args.remote_candidate:
        parser.error("--cpu-smoke and --remote-candidate are mutually exclusive")
    if args.cpu_smoke:
        report = run_candidate(dataset_path=args.dataset, device="cpu", epochs=min(args.epochs, 4), microbatch=min(args.microbatch, 8), d_model=min(args.d_model, 64), n_layers=min(args.n_layers, 2), experts=min(args.experts, 2), expert_hidden=min(args.expert_hidden, 128), max_length=args.max_length, row_limit=args.row_limit or 80, checkpoint_dir=None)
        report["status"] = "cpu_smoke_candidate_only"
    elif args.remote_candidate:
        report = run_candidate(dataset_path=args.dataset, device="cuda:0", epochs=args.epochs, microbatch=args.microbatch, d_model=args.d_model, n_layers=args.n_layers, experts=args.experts, expert_hidden=args.expert_hidden, max_length=args.max_length, row_limit=args.row_limit, checkpoint_dir=args.checkpoint_dir)
    else:
        report = {"schema_version": SCHEMA_VERSION, "status": "plan_only", "dataset": str(args.dataset), "dataset_sha256": _sha_file(args.dataset), "execution": {"optimizer_started": False, "gpu_touched": False, "docker_started": False, "network_used": False}, "promotion": dict(PROMOTION)}
    report["report_sha256"] = _sha_json(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "worst_seed": report.get("worst_seed"), "promotion": report.get("promotion")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["VariantSelector", "run_candidate"]
