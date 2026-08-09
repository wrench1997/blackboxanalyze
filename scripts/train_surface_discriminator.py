#!/usr/bin/env python3
"""Train the family-specific sanitized shadow-surface discriminator."""

from __future__ import annotations

import copy
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.rule_ir_decoder import DECODER_FAMILIES, calibrate_abstention_threshold, trace_feature_vector  # noqa: E402
from app.surface_discriminator import FEATURE_DIM, SurfaceDiscriminator  # noqa: E402
from train_rule_ir_decoder import _shadow_surface_records  # noqa: E402


SEED = 20260931
EVAL_SEEDS = (20260937, 20260941, 20260943)
OUTPUT_DIR = ROOT / "artifacts/surface-discriminator-loop-12-20260931"
CHECKPOINT = OUTPUT_DIR / "surface_discriminator.pt"
REPORT = OUTPUT_DIR / "report.json"
PROTOCOL = ROOT / "research/juice_shop_loop_12_surface_discriminator_protocol_v1.json"


def _features(rows: list[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor([trace_feature_vector(row["traces"]) for row in rows], dtype=torch.float32)


def _normalise(train: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train.mean(dim=0)
    std = train.std(dim=0, unbiased=False).clamp_min(1e-4)
    return (values - mean) / std, mean, std


@torch.inference_mode()
def _evaluate(model: SurfaceDiscriminator, features: torch.Tensor, labels: torch.Tensor, rows: list[dict[str, Any]], device: torch.device, threshold: float = 0.55) -> dict[str, Any]:
    model.eval()
    probabilities = torch.softmax(model(features.to(device)), dim=-1).cpu()
    predictions = probabilities.argmax(dim=-1)
    confidence = probabilities.max(dim=-1).values
    correct = predictions.eq(labels)
    accepted = confidence >= threshold
    by_family: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row, matched in zip(rows, correct):
        stats = by_family[row["family"]]
        stats[0] += int(matched)
        stats[1] += 1
    return {
        "accuracy": round(float(correct.float().mean()), 6) if len(labels) else 0.0,
        "correct": int(correct.sum()),
        "total": len(labels),
        "coverage": round(float(accepted.float().mean()), 6) if len(labels) else 0.0,
        "abstain_rate": round(float((~accepted).float().mean()), 6) if len(labels) else 0.0,
        "accepted_accuracy": round(float(correct[accepted].float().mean()), 6) if bool(accepted.any()) else None,
        "by_family": {key: round(good / total, 6) for key, (good, total) in sorted(by_family.items())},
        "probabilities": probabilities.tolist(),
        "predictions": [DECODER_FAMILIES[int(value)] for value in predictions],
        "targets": [row["family"] for row in rows],
    }


def main() -> None:
    started = time.perf_counter()
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_rows = _shadow_surface_records(SEED, 600)
    validation_rows = _shadow_surface_records(SEED + 1, 100)
    train_raw = _features(train_rows)
    validation_raw = _features(validation_rows)
    train_features, mean, std = _normalise(train_raw, train_raw)
    validation_features = (validation_raw - mean) / std
    label_index = {family: index for index, family in enumerate(DECODER_FAMILIES)}
    train_labels = torch.tensor([label_index[row["family"]] for row in train_rows], dtype=torch.long)
    validation_labels = torch.tensor([label_index[row["family"]] for row in validation_rows], dtype=torch.long)
    loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    model = SurfaceDiscriminator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.01)
    best_state = None
    best_accuracy = -1.0
    history = []
    for epoch in range(1, 41):
        model.train()
        running = 0.0
        seen = 0
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features.to(device))
            loss = loss_fn(logits, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * len(labels)
            seen += len(labels)
        validation = _evaluate(model, validation_features, validation_labels, validation_rows, device)
        history.append({"epoch": epoch, "train_loss": round(running / max(seen, 1), 6), "validation_accuracy": validation["accuracy"]})
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.inference_mode():
        validation_probabilities = torch.softmax(model(validation_features.to(device)), dim=-1).cpu().tolist()
    calibration = calibrate_abstention_threshold(validation_probabilities, validation_labels.tolist(), minimum_precision=0.98)
    fresh = []
    for eval_seed in EVAL_SEEDS:
        rows = _shadow_surface_records(eval_seed, 100)
        raw = _features(rows)
        features = (raw - mean) / std
        labels = torch.tensor([label_index[row["family"]] for row in rows], dtype=torch.long)
        result = _evaluate(model, features, labels, rows, device, threshold=float(calibration["threshold"]))
        result["seed"] = eval_seed
        fresh.append(result)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": "sift-surface-discriminator-checkpoint-v1",
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "feature_dim": FEATURE_DIM,
        "families": list(DECODER_FAMILIES),
        "normalisation_mean": mean.tolist(),
        "normalisation_std": std.tolist(),
        "abstain_threshold": calibration["threshold"],
        "seed": SEED,
        "device_at_training": str(device),
    }, CHECKPOINT)
    report = {
        "schema_version": "sift-surface-discriminator-report-v1",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "status": "completed",
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "model": {"parameters": sum(parameter.numel() for parameter in model.parameters()), "feature_dim": FEATURE_DIM, "families": list(DECODER_FAMILIES)},
        "data": {"train_examples": len(train_rows), "validation_examples": len(validation_rows), "fresh_examples_per_seed": 700, "input_oracle_fields_excluded": ["family", "record_id"]},
        "calibration": calibration,
        "validation": _evaluate(model, validation_features, validation_labels, validation_rows, device, threshold=float(calibration["threshold"])),
        "fresh_seeds": fresh,
        "stability": {"mean_accuracy": round(sum(row["accuracy"] for row in fresh) / len(fresh), 6), "min_accuracy": min(row["accuracy"] for row in fresh), "max_abstain_rate": max(row["abstain_rate"] for row in fresh)},
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"device": str(device), "validation_accuracy": report["validation"]["accuracy"], "fresh_mean": report["stability"]["mean_accuracy"], "threshold": calibration["threshold"], "checkpoint": str(CHECKPOINT.relative_to(ROOT))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
