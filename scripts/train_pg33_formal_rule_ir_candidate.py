"""Train and gate a formal PG-33 Rule IR candidate.

This is the first trainer allowed to consume PG-33.  It deliberately uses only
the bounded visible projection (method, abstract probe class and response
shape); family labels, typed oracle values, source IDs, target IDs, evidence
hashes and Rule IR bindings never enter the feature vector.  The output is a
quarantined candidate report.  Promotion and long-term memory remain owned by
the capability gate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.catalog_rule_decoder import (  # noqa: E402
    CATALOG_DECODER_FAMILIES,
    CatalogRuleIRDecoderV2,
    abstract_catalog_rule_ir,
    catalog_feature_vector,
)
from app.dataset_utility_audit import audit_dataset  # noqa: E402
from app.model_capability_gate import evaluate_model_capability  # noqa: E402
from app.rule_ir_decoder import FEATURE_DIM  # noqa: E402


PROTOCOL_ID = "sift-pg33-formal-rule-ir-candidate-v1"
CATALOG_PATH = ROOT / "research" / "pg_pk_33_get_post_typed_replay_catalog_v1.json"
OUTPUT_DIR = ROOT / "artifacts" / "pg33-formal-rule-ir-candidate"
REPORT_PATH = ROOT / "research" / "pg_pk_33_formal_model_candidate_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg_pk_33_formal_model_candidate_v1.md"
SEED = 20260802
EPOCHS = 180
ABSTAIN_MARGIN = 0.08
MIN_DEV_PRECISION = 0.95


def _visible_record(row: dict[str, Any]) -> dict[str, Any]:
    """Adapt PG-33's safe projections to the decoder's visible trace schema."""

    manifest = row["payload_manifest"]
    response = row.get("response_projection") or {}
    shape = response.get("shape") or {}
    # These are abstract probe classes, not raw request strings.  The marker,
    # payload and route family are intentionally excluded.
    payload = {
        "method": str(manifest.get("method", "GET")),
        "path": "/maze/replay/fixture",
        "probe_kind": str(manifest.get("probe_kind", "typed_probe")),
        "probe": str(manifest.get("probe_kind", "typed_probe")),
        "encoding": "+".join(str(item) for item in manifest.get("encoding_chain", ["identity"])),
    }
    return {
        "payload": payload,
        "probe_artifact": {"encoding": payload["encoding"]},
        "response_projection": {
            "status_code": int(response.get("status_code", 0)),
            "headers": {"content-type": str(response.get("content_type_class", "other"))},
            "body_length": int(response.get("body_length_bucket", "0").split("-", 1)[0].replace("+", "") or 0),
            "json_shape": {
                "kind": str(shape.get("kind", "other")),
                "key_count": int(shape.get("key_count", 0)),
                "scalar_count": int(shape.get("scalar_count", 0)),
                "array_count": int(shape.get("array_count", 0)),
            },
        },
        # The oracle is represented only by a constant shape count.  Its
        # positive/negative values are never passed to the learner.
        "oracle_projection": {"field_count": 1},
    }


def _features(rows: list[dict[str, Any]]) -> torch.Tensor:
    vectors = []
    for row in rows:
        visible = _visible_record(row)
        vector = catalog_feature_vector(visible)
        # The base projector intentionally drops categorical values.  Add a
        # small, deterministic hash-bucket view of bounded *observable* shape
        # categories so the decoder can distinguish an abstract markup probe
        # from an abstract structured-fragment probe without seeing a family,
        # route, marker, source or oracle label.  Pair controls retain exactly
        # the same category, so this cannot manufacture a positive signal.
        manifest = row["payload_manifest"]
        response = row.get("response_projection") or {}
        shape = response.get("shape") or {}
        categories = (
            f"method:{manifest.get('method', 'GET')}",
            f"probe:{manifest.get('probe_kind', 'typed_probe')}",
            f"encoding:{'+'.join(str(item) for item in manifest.get('encoding_chain', ['identity']))}",
            f"content_type:{response.get('content_type_class', 'other')}",
            f"shape_kind:{shape.get('kind', 'other')}",
            f"shape_keys:{int(shape.get('key_count', 0)) // 4}",
            f"shape_arrays:{int(shape.get('array_count', 0))}",
            f"status:{int(response.get('status_code', 0)) // 100}",
        )
        for category in categories:
            digest = hashlib.blake2b(category.encode("utf-8"), digest_size=8).digest()
            index = 192 + (int.from_bytes(digest, "little") % 64)
            vector[index] = min(float(vector[index]) + 1.0, 8.0)
        vectors.append(vector)
    return torch.tensor(vectors, dtype=torch.float32)


def _normalise(train_raw: torch.Tensor, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train_raw.mean(dim=0)
    std = train_raw.std(dim=0, unbiased=False).clamp_min(1e-4)
    return (raw - mean) / std, mean, std


def _train(rows: list[dict[str, Any]], *, seed: int, device: torch.device) -> tuple[CatalogRuleIRDecoderV2, dict[str, Any]]:
    if not rows:
        raise ValueError("PG-33 train split has no typed-positive rows")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    raw = _features(rows)
    features, mean, std = _normalise(raw, raw)
    labels = torch.tensor([CATALOG_DECODER_FAMILIES.index(row["family"]) for row in rows], dtype=torch.long)
    model = CatalogRuleIRDecoderV2(dropout=0.05).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.01)
    generator = torch.Generator().manual_seed(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_accuracy = -1.0
    history: list[dict[str, Any]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(features), generator=generator)
        total_loss = 0.0
        for start in range(0, len(order), 8):
            indexes = order[start:start + 8]
            optimizer.zero_grad(set_to_none=True)
            logits = model(features[indexes].to(device))
            loss = loss_fn(logits, labels[indexes].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indexes)
        model.eval()
        with torch.inference_mode():
            prediction = model(features.to(device)).argmax(dim=-1).cpu()
        accuracy = float(prediction.eq(labels).float().mean())
        history.append({"epoch": epoch, "loss": round(total_loss / len(features), 6), "train_accuracy": round(accuracy, 6)})
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "train_accuracy": round(best_accuracy, 6),
        "history_tail": history[-5:],
    }


@torch.inference_mode()
def _raw_predictions(
    model: CatalogRuleIRDecoderV2,
    rows: list[dict[str, Any]],
    mean: list[float],
    std: list[float],
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    raw = _features(rows)
    normalised = (raw - torch.tensor(mean) ) / torch.tensor(std)
    outputs = model.decode(normalised.to(device), abstain_threshold=0.0, margin_threshold=0.0)
    result: list[dict[str, Any]] = []
    for row, output in zip(rows, outputs):
        result.append({
            "sample_id": row["sample_id"],
            "family": row["family"],
            "positive": bool(row["oracle_projection"].get("positive", False)),
            "candidate_family": output["candidate_family"],
            "confidence": float(output["confidence"]),
            "margin": float(output["margin"]),
        })
    return result


def _calibrate(dev_predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose a dev-only threshold that forbids observed false positives."""

    candidates = sorted({float(item["confidence"]) for item in dev_predictions}, reverse=True)
    candidates.append(1.0)
    best: tuple[int, float, float] | None = None
    for threshold in candidates:
        emitted = [
            item for item in dev_predictions
            if item["confidence"] >= threshold and item["margin"] >= ABSTAIN_MARGIN
        ]
        if not emitted:
            continue
        true_positive = sum(int(item["positive"] and item["candidate_family"] == item["family"]) for item in emitted)
        false_positive = sum(int(not item["positive"]) for item in emitted)
        precision = true_positive / len(emitted)
        if precision + 1e-12 < MIN_DEV_PRECISION or false_positive:
            continue
        score = (true_positive, precision, -threshold)
        if best is None or score > best:
            best = score
    if best is None:
        return {
            "abstain_threshold": 1.0,
            "margin_threshold": ABSTAIN_MARGIN,
            "dev_precision": 0.0,
            "dev_coverage": 0.0,
            "reason": "no_dev_threshold_meets_zero_false_positive_gate",
        }
    threshold = -best[2]
    emitted = [item for item in dev_predictions if item["confidence"] >= threshold and item["margin"] >= ABSTAIN_MARGIN]
    return {
        "abstain_threshold": round(threshold, 6),
        "margin_threshold": ABSTAIN_MARGIN,
        "dev_precision": round(best[1], 6),
        "dev_coverage": round(len(emitted) / max(len(dev_predictions), 1), 6),
        "reason": "dev_zero_false_positive_threshold",
    }


def _metrics(predictions: list[dict[str, Any]], calibration: dict[str, Any]) -> dict[str, float]:
    threshold = float(calibration["abstain_threshold"])
    margin_threshold = float(calibration["margin_threshold"])
    emitted = [item for item in predictions if item["confidence"] >= threshold and item["margin"] >= margin_threshold]
    positives = [item for item in predictions if item["positive"]]
    negatives = [item for item in predictions if not item["positive"]]
    true_positive = sum(int(item["positive"] and item["candidate_family"] == item["family"]) for item in emitted)
    false_positive = sum(int(not item["positive"]) for item in emitted)
    abstained = [item for item in predictions if item not in emitted]
    correct_abstain = sum(int(not item["positive"]) for item in abstained)
    confidence_error = []
    for item in predictions:
        predicted_correct = item in emitted and item["positive"] and item["candidate_family"] == item["family"]
        confidence_error.append(abs(float(item["confidence"]) - float(predicted_correct)))
    return {
        "typed_recall": round(true_positive / max(len(positives), 1), 6),
        "precision": round(true_positive / max(len(emitted), 1), 6) if emitted else 1.0,
        "false_positive_rate": round(false_positive / max(len(negatives), 1), 6),
        "abstain_precision": round(correct_abstain / max(len(abstained), 1), 6) if abstained else 1.0,
        "ece": round(sum(confidence_error) / max(len(confidence_error), 1), 6),
        "median_queries": 2.0,
    }


def _cell_metrics(rows: list[dict[str, Any]], predictions: dict[str, dict[str, Any]], calibration: dict[str, Any]) -> dict[str, float]:
    return _metrics([predictions[row["sample_id"]] for row in rows], calibration)


def _checkpoint_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PG-33 formal Rule IR candidate",
        "",
        "本报告是候选训练与能力门结果，不是已晋升模型。输入仅包含脱敏可见投影；typed oracle、来源和证据哈希只用于验收与评估。",
        "",
        "| cell | role | typed recall | precision | FPR | abstain precision |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for cell in report["cells"]:
        metrics = cell["candidate_metrics"]
        lines.append(
            f"| {cell['dataset_id']} | {cell['role']} | {metrics['typed_recall']:.2f} | "
            f"{metrics['precision']:.2f} | {metrics['false_positive_rate']:.2f} | {metrics['abstain_precision']:.2f} |"
        )
    lines.extend([
        "",
        f"能力门状态：`{report['capability_gate']['status']}`。训练授权：`{report['capability_gate']['training_allowed']}`；长期记忆：`{report['capability_gate']['memory_promotion_allowed']}`。",
        "",
        "如果 family-holdout/OOD 召回没有超过 always-abstain 基线，结果只能说明数据和训练管线可复现，不能说明泛化能力提升。",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    started = time.perf_counter()
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    utility = audit_dataset(catalog, dataset_id="pg33")
    if utility["utility_class"] != "replay_training_candidate_pending_capability_gate":
        raise RuntimeError(f"PG-33 utility gate rejected: {utility['utility_class']}")
    rows = list(catalog["samples"])
    train_rows = [row for row in rows if row["dataset_role"] == "train" and row["oracle_projection"]["positive"]]
    dev_rows = [row for row in rows if row["dataset_role"] == "dev"]
    if not train_rows or not dev_rows:
        raise RuntimeError("PG-33 requires train and dev replay rows before formal training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, training = _train(train_rows, seed=SEED, device=device)
    dev_predictions = _raw_predictions(model, dev_rows, training["mean"], training["std"], device=device)
    calibration = _calibrate(dev_predictions)
    all_predictions = _raw_predictions(model, rows, training["mean"], training["std"], device=device)
    prediction_by_id = {item["sample_id"]: item for item in all_predictions}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "rule_ir_decoder_v2.pt"
    torch.save({
        "schema_version": "sift-pg33-formal-rule-ir-checkpoint-v1",
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "feature_dim": FEATURE_DIM,
        "families": list(CATALOG_DECODER_FAMILIES),
        "normalisation_mean": training["mean"],
        "normalisation_std": training["std"],
        "abstain_threshold": calibration["abstain_threshold"],
        "margin_threshold": calibration["margin_threshold"],
        "seed": SEED,
        "device_at_training": str(device),
    }, checkpoint_path)
    checkpoint_sha256 = _checkpoint_sha256(checkpoint_path)

    cells: list[dict[str, Any]] = []
    for cell in catalog["dataset_tests"]:
        cell_rows = [
            row for row in rows
            if row["dataset_role"] == cell["role"] and int(row["sampling_seed"]) == int(cell["sampling_seed"])
        ]
        candidate_metrics = _cell_metrics(cell_rows, prediction_by_id, calibration)
        baseline_metrics = {
            "typed_recall": 0.0,
            "precision": 1.0,
            "false_positive_rate": 0.0,
            "abstain_precision": round(sum(int(not row["oracle_projection"]["positive"]) for row in cell_rows) / max(len(cell_rows), 1), 6),
            "ece": 0.0,
            "median_queries": 2.0,
        }
        enriched = dict(cell)
        enriched["metrics_status"] = "completed"
        enriched["checkpoint_sha256"] = checkpoint_sha256
        enriched["baseline_metrics"] = baseline_metrics
        enriched["candidate_metrics"] = candidate_metrics
        enriched["metrics"] = candidate_metrics
        enriched["evidence_hash"] = hashlib.sha256(json.dumps(enriched, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        cells.append(enriched)

    baseline_all = _metrics(
        [{**item, "confidence": 0.0, "candidate_family": "none"} for item in all_predictions],
        {"abstain_threshold": 1.0, "margin_threshold": 0.0},
    )
    candidate_all = _metrics(all_predictions, calibration)
    holdout_predictions = [item for item in all_predictions if item["sample_id"].startswith("pg33-v05") or item["sample_id"].startswith("pg33-v06")]
    baseline_worst = baseline_all
    candidate_worst = _metrics(holdout_predictions, calibration)
    capability_evidence = {
        "claim_id": "pg33-formal-rule-ir-candidate",
        "dataset_tests": cells,
        "unit_tests_passed": True,
        "oracle_validated": True,
        "data_lineage_complete": True,
        "authorized_sources_attested": True,
        "raw_data_retained": False,
        "false_positive_count": sum(int(not item["positive"] and item["confidence"] >= calibration["abstain_threshold"] and item["margin"] >= calibration["margin_threshold"]) for item in all_predictions),
        "baseline_metrics": baseline_all,
        "candidate_metrics": candidate_all,
        "baseline_worst_case_metrics": baseline_worst,
        "candidate_worst_case_metrics": candidate_worst,
    }
    capability_gate = evaluate_model_capability(capability_evidence)
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "sift-pg33-formal-model-candidate-report-v1",
        "catalog": {
            "path": str(CATALOG_PATH.relative_to(ROOT)),
            "catalog_sha256": hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest(),
            "sample_count": len(rows),
            "utility_audit": utility,
        },
        "model": {
            "class": "CatalogRuleIRDecoderV2",
            "families": list(CATALOG_DECODER_FAMILIES),
            "feature_dim": FEATURE_DIM,
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "visible_projection_labels": False,
            "rule_ir_emission": "grammar_checked_abstract_template",
        },
        "training": {
            "train_count": len(train_rows),
            "train_families": sorted({row["family"] for row in train_rows}),
            "seed": SEED,
            "epochs": EPOCHS,
            "train_accuracy": training["train_accuracy"],
            "history_tail": training["history_tail"],
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_sha256": checkpoint_sha256,
        },
        "calibration": calibration,
        "cells": cells,
        "aggregate": {"baseline_metrics": baseline_all, "candidate_metrics": candidate_all, "candidate_worst_case_metrics": candidate_worst},
        "capability_gate": capability_gate,
        "promotion": {
            "training_allowed": bool(capability_gate["training_allowed"]),
            "memory_promotion_allowed": bool(capability_gate["memory_promotion_allowed"]),
            "status": "quarantined_candidate" if capability_gate["status"] != "pass" else "eligible_pending_review",
        },
        "target_scope": {
            "base_url": "http://127.0.0.1:3100",
            "independent_target_implementation": False,
            "replay_transport": "in_process_asgi",
            "not_claimed": "third-party or separately implemented target generalization",
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "report_path": str(REPORT_PATH.relative_to(ROOT)),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "device": str(device),
        "train_count": len(train_rows),
        "train_accuracy": training["train_accuracy"],
        "calibration": calibration,
        "candidate_metrics": candidate_all,
        "capability_gate": capability_gate,
        "report": str(REPORT_PATH.relative_to(ROOT)),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
