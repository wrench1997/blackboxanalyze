"""Train a generic, calibrated surface-role discriminator on CUDA when available."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rule_ir_decoder import calibrate_abstention_threshold  # noqa: E402
from app.surface_role_discriminator import (  # noqa: E402
    SURFACE_ROLE_FEATURE_DIM,
    SURFACE_ROLES,
    SurfaceRoleDiscriminator,
    surface_shape_feature_vector,
)


PROTOCOL_ID = "pg-pk-08-surface-role-calibration-v1"
SEED = 20260851
FRESH_SEEDS = (20260857, 20260861, 20260867)
OUTPUT_DIR = ROOT / "artifacts" / "surface-role-discriminator-pg-pk-08"
CHECKPOINT = OUTPUT_DIR / "surface_role_discriminator.pt"
REPORT = OUTPUT_DIR / "report.json"
PROTOCOL = ROOT / "research" / "pg_pk_08_surface_role_calibration_protocol_v1.json"


def _shape_rows(seed: int, per_role: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for role in SURFACE_ROLES:
        for index in range(per_role):
            if role == "reflected_attribute":
                shape = {
                    "content_type_class": "html",
                    "status_class": "2xx",
                    "html_tag_count": rng.randint(3, 16),
                    "html_attribute_count": rng.randint(1, 8),
                    "script_count": rng.randint(0, 2),
                    "json_field_count": 0,
                    "response_header_count": rng.randint(2, 6),
                    "body_length": rng.randint(120, 900),
                    "body_length_delta_abs": rng.randint(12, 180),
                }
            elif role == "reflected_text":
                shape = {
                    "content_type_class": "html",
                    "status_class": "2xx",
                    "html_tag_count": rng.randint(3, 16),
                    "html_attribute_count": rng.randint(0, 3),
                    "script_count": rng.randint(0, 2),
                    "json_field_count": 0,
                    "response_header_count": rng.randint(2, 6),
                    "body_length": rng.randint(100, 900),
                    "body_length_delta_abs": rng.randint(12, 180),
                }
            elif role == "json_echo":
                shape = {
                    "content_type_class": "json",
                    "status_class": "2xx",
                    "html_tag_count": 0,
                    "html_attribute_count": 0,
                    "script_count": 0,
                    "json_field_count": rng.randint(1, 8),
                    "response_header_count": rng.randint(2, 6),
                    "body_length": rng.randint(40, 500),
                    "body_length_delta_abs": rng.randint(4, 180),
                }
            elif role == "header_echo":
                shape = {
                    "content_type_class": "html",
                    "status_class": "2xx",
                    "html_tag_count": rng.randint(2, 12),
                    "html_attribute_count": rng.randint(0, 3),
                    "script_count": rng.randint(0, 1),
                    "json_field_count": 0,
                    "response_header_count": rng.randint(5, 10),
                    "body_length": rng.randint(80, 600),
                    "body_length_delta_abs": rng.randint(0, 24),
                }
            else:
                shape = {
                    "content_type_class": "html",
                    "status_class": "2xx",
                    "html_tag_count": rng.randint(2, 12),
                    "html_attribute_count": rng.randint(0, 3),
                    "script_count": rng.randint(0, 1),
                    "json_field_count": 0,
                    "response_header_count": rng.randint(2, 4),
                    "body_length": rng.randint(70, 550),
                    "body_length_delta_abs": rng.randint(0, 8),
                }
            rows.append({"surface_shape": shape, "role": role, "row_id": f"{seed}-{role}-{index}"})
    return rows


def _features(rows: list[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor([surface_shape_feature_vector(row) for row in rows], dtype=torch.float32)


def _normalise(train: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train.mean(dim=0)
    std = train.std(dim=0, unbiased=False).clamp_min(1e-4)
    return (values - mean) / std, mean, std


@torch.inference_mode()
def _evaluate(
    model: SurfaceRoleDiscriminator,
    features: torch.Tensor,
    labels: torch.Tensor,
    rows: list[dict[str, Any]],
    device: torch.device,
    threshold: float,
) -> dict[str, Any]:
    probabilities = torch.softmax(model(features.to(device)), dim=-1).cpu()
    predictions = probabilities.argmax(dim=-1)
    confidence = probabilities.max(dim=-1).values
    correct = predictions.eq(labels)
    accepted = confidence >= float(threshold)
    by_role: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0, "accepted": 0})
    for row, matched, is_accepted in zip(rows, correct, accepted):
        stats = by_role[str(row["role"])]
        stats["correct"] += int(matched)
        stats["total"] += 1
        stats["accepted"] += int(is_accepted)
    return {
        "accuracy": round(float(correct.float().mean()), 6) if len(labels) else 0.0,
        "total": len(labels),
        "coverage": round(float(accepted.float().mean()), 6) if len(labels) else 0.0,
        "abstain_rate": round(float((~accepted).float().mean()), 6) if len(labels) else 0.0,
        "accepted_accuracy": round(float(correct[accepted].float().mean()), 6) if bool(accepted.any()) else None,
        "by_role": {
            role: {
                "accuracy": round(values["correct"] / max(values["total"], 1), 6),
                "accepted": values["accepted"],
                "total": values["total"],
            }
            for role, values in sorted(by_role.items())
        },
        "predictions": [SURFACE_ROLES[int(value)] for value in predictions],
        "targets": [str(row["role"]) for row in rows],
    }


def main() -> None:
    started = time.perf_counter()
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_rows = _shape_rows(SEED, 500)
    validation_rows = _shape_rows(SEED + 1, 120)
    train_raw = _features(train_rows)
    validation_raw = _features(validation_rows)
    train_features, mean, std = _normalise(train_raw, train_raw)
    validation_features = (validation_raw - mean) / std
    label_index = {role: index for index, role in enumerate(SURFACE_ROLES)}
    train_labels = torch.tensor([label_index[row["role"]] for row in train_rows], dtype=torch.long)
    validation_labels = torch.tensor([label_index[row["role"]] for row in validation_rows], dtype=torch.long)
    model = SurfaceRoleDiscriminator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.02)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.02)
    features = train_features.to(device)
    labels = train_labels.to(device)
    best_state: dict[str, torch.Tensor] | None = None
    best_accuracy = -1.0
    history: list[dict[str, Any]] = []
    for epoch in range(1, 81):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(features), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            accuracy = float(model(validation_features.to(device)).argmax(dim=-1).eq(validation_labels.to(device)).float().mean())
        history.append({"epoch": epoch, "loss": round(float(loss.detach().cpu()), 6), "validation_accuracy": round(accuracy, 6)})
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.inference_mode():
        validation_probabilities = torch.softmax(model(validation_features.to(device)), dim=-1).cpu().tolist()
    calibration = calibrate_abstention_threshold(validation_probabilities, validation_labels.tolist(), minimum_precision=0.99)
    threshold = float(calibration["threshold"])
    fresh_results: list[dict[str, Any]] = []
    for seed in FRESH_SEEDS:
        rows = _shape_rows(seed, 120)
        raw = _features(rows)
        normalized = (raw - mean) / std
        labels_for_seed = torch.tensor([label_index[row["role"]] for row in rows], dtype=torch.long)
        result = _evaluate(model, normalized, labels_for_seed, rows, device, threshold)
        result["seed"] = seed
        fresh_results.append(result)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": "sift-surface-role-discriminator-checkpoint-v1",
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "feature_dim": SURFACE_ROLE_FEATURE_DIM,
        "roles": list(SURFACE_ROLES),
        "normalisation_mean": mean.tolist(),
        "normalisation_std": std.tolist(),
        "abstain_threshold": threshold,
        "seed": SEED,
        "device_at_training": str(device),
    }, CHECKPOINT)
    validation = _evaluate(model, validation_features, validation_labels, validation_rows, device, threshold)
    fresh_mean = sum(row["accuracy"] for row in fresh_results) / len(fresh_results)
    fresh_min = min(row["accuracy"] for row in fresh_results)
    acceptance = {
        "fresh_min_accuracy_required": 0.90,
        "fresh_min_accuracy": fresh_min,
        "minimum_precision_required": 0.99,
        "validation_precision": calibration.get("precision"),
        "passed": bool(fresh_min >= 0.90 and float(calibration.get("precision", 0.0)) >= 0.99),
        "failure_action": "diagnostic_only_and_abstain",
    }
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "sift-surface-role-discriminator-report-v1",
        "status": "accepted_for_gate" if acceptance["passed"] else "diagnostic_only",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() else 0,
        "model": {"parameters": sum(parameter.numel() for parameter in model.parameters()), "feature_dim": SURFACE_ROLE_FEATURE_DIM, "roles": list(SURFACE_ROLES)},
        "data": {"train_examples": len(train_rows), "validation_examples": len(validation_rows), "fresh_examples_per_seed": len(_shape_rows(FRESH_SEEDS[0], 120)), "oracle_fields_excluded": True},
        "calibration": calibration,
        "validation": validation,
        "fresh_seeds": fresh_results,
        "stability": {
            "mean_accuracy": round(fresh_mean, 6),
            "min_accuracy": fresh_min,
            "max_abstain_rate": max(row["abstain_rate"] for row in fresh_results),
        },
        "acceptance": acceptance,
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "history_tail": history[-5:],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "status": report["status"],
        "device": report["device"],
        "validation_accuracy": report["validation"]["accuracy"],
        "fresh_mean_accuracy": report["stability"]["mean_accuracy"],
        "threshold": threshold,
        "checkpoint": report["checkpoint"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
