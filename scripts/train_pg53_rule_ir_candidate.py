"""Train and evaluate a quarantined PG-53 Rule IR candidate.

Training uses only PG-35 visible projections; PG-36 is an independent
implementation holdout.  Typed oracle fields are labels used outside the
feature builder and are never fed to the model.  The resulting checkpoint is
an experiment artifact, not a promoted model or long-term memory write.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg53_rule_ir_candidate import PG53RuleIRCandidate, PG53_MODEL_FAMILIES  # noqa: E402
from app.rule_ir_decoder import FEATURE_DIM  # noqa: E402


PROTOCOL_ID = "pg-pk-53-rule-ir-candidate-v1"
REPORT_PATH = ROOT / "research" / "pg53_rule_ir_candidate_report_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg53_rule_ir_candidate_report_v1.md"
SOURCE_REPORT_PATH = ROOT / "research" / "pg53_cross_source_typed_replay_report_v1.json"
FUNNEL_DATASET_PATH = ROOT / "research" / "pg53_web_feature_funnel_dataset_v1.json"
OUTPUT_DIR = ROOT / "artifacts" / "pg53-rule-ir-candidate"
CHECKPOINT_PATH = OUTPUT_DIR / "decoder.pt"
SEED = 20260803
EPOCHS = 180
MARGIN_THRESHOLD = 0.08


def _visible_row(row: dict[str, Any]) -> dict[str, Any]:
    response = row["candidate"]["response"]
    shape = response.get("shape") or {}
    manifest = row["payload_manifest"]
    # The learner sees neutral transport/probe descriptors and bounded shape;
    # family, source, stage, evidence and oracle values are excluded.
    return {
        "payload": {
            "method": row["method"],
            "path": "/independent/fixture/surface",
            "probe_kind": "abstract_channel_class",
            "probe": "abstract_probe",
            "encoding": "identity",
        },
        "probe_artifact": {"encoding": "identity"},
        "response_projection": {
            "status_code": int(response.get("status_code", 0)),
            "headers": {"content-type": str(response.get("content_type_class", "other"))},
            "json_shape": {
                "kind": str(shape.get("kind", "other")),
                "key_count": int(shape.get("key_count", 0)),
                "scalar_count": int(shape.get("scalar_count", 0)),
                "array_count": int(shape.get("array_count", 0)),
            },
            "body_length": int(str(response.get("body_length_bucket", "0")).split("-", 1)[0] or 0),
        },
        "oracle_projection": {"field_count": 1},
        "probe_manifest": {"method": manifest.get("method", row["method"]), "encoding_depth": 0},
    }


def _features(rows: list[dict[str, Any]], feature_map: dict[str, dict[str, Any]], selected_features: list[str]) -> torch.Tensor:
    """Encode only the feature names that passed the reviewed funnel."""

    vectors: list[list[float]] = []
    for row in rows:
        values = feature_map[str(row["sample_id"])]["model_features"]
        vector = [0.0] * FEATURE_DIM
        for offset, name in enumerate(selected_features):
            value = values.get(name, 0.0)
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = 0.0
            # Keep every selected observable bounded and source-invariant.
            vector[8 + offset] = max(-1.0, min(1.0, numeric / 32.0 if abs(numeric) > 1.0 else numeric))
        vectors.append(vector)
    return torch.tensor(vectors, dtype=torch.float32)


def _labels(rows: list[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor([PG53_MODEL_FAMILIES.index(str(row["family"])) for row in rows], dtype=torch.long)


def _normalise(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = raw.mean(dim=0)
    std = raw.std(dim=0, unbiased=False).clamp_min(1e-4)
    return (raw - mean) / std, mean, std


def _train(rows: list[dict[str, Any]], feature_map: dict[str, dict[str, Any]], selected_features: list[str], *, device: torch.device) -> tuple[PG53RuleIRCandidate, dict[str, Any]]:
    if not rows:
        raise ValueError("PG-53 training split is empty")
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    raw = _features(rows, feature_map, selected_features)
    features, mean, std = _normalise(raw)
    labels = _labels(rows)
    counts = torch.bincount(labels, minlength=len(PG53_MODEL_FAMILIES)).float()
    weights = torch.zeros_like(counts)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    model = PG53RuleIRCandidate().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.015)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.02)
    generator = torch.Generator().manual_seed(SEED)
    best_state: dict[str, torch.Tensor] | None = None
    best_accuracy = -1.0
    history: list[dict[str, Any]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(features), generator=generator)
        total_loss = 0.0
        for start in range(0, len(order), 32):
            indexes = order[start:start + 32]
            optimizer.zero_grad(set_to_none=True)
            logits = model(features[indexes].to(device))
            loss = loss_fn(logits, labels[indexes].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(indexes)
        model.eval()
        with torch.inference_mode():
            predicted = model(features.to(device)).argmax(dim=-1).cpu()
        accuracy = float(predicted.eq(labels).float().mean())
        history.append({"epoch": epoch, "loss": round(total_loss / len(features), 6), "train_accuracy": round(accuracy, 6)})
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "normalisation_mean": mean.tolist(),
        "normalisation_std": std.tolist(),
        "train_accuracy": round(best_accuracy, 6),
        "class_counts": {name: int(counts[index]) for index, name in enumerate(PG53_MODEL_FAMILIES) if counts[index] > 0},
        "history_tail": history[-5:],
    }


@torch.inference_mode()
def _raw_predictions(model: PG53RuleIRCandidate, rows: list[dict[str, Any]], feature_map: dict[str, dict[str, Any]], selected_features: list[str], fit: dict[str, Any], *, device: torch.device) -> list[dict[str, Any]]:
    if not rows:
        return []
    mean = torch.tensor(fit["normalisation_mean"], dtype=torch.float32)
    std = torch.tensor(fit["normalisation_std"], dtype=torch.float32).clamp_min(1e-4)
    features = (_features(rows, feature_map, selected_features) - mean) / std
    outputs = model.decode(features.to(device), abstain_threshold=0.0, margin_threshold=0.0)
    return [
        {
            "sample_id": row["sample_id"],
            "source_id": row["source_id"],
            "sampling_seed": row["sampling_seed"],
            "expected_family": row["family"],
            "positive": row["decision"] == "confirmed_positive",
            "candidate_family": output["candidate_family"],
            "confidence": output["confidence"],
            "margin": output["margin"],
        }
        for row, output in zip(rows, outputs)
    ]


def _metrics(predictions: list[dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    emitted = [item for item in predictions if item["candidate_family"] != "ordinary_response" and item["confidence"] >= threshold and item["margin"] >= MARGIN_THRESHOLD]
    true_positive = sum(int(item["positive"] and item["candidate_family"] == item["expected_family"]) for item in emitted)
    false_accept = sum(int(not (item["positive"] and item["candidate_family"] == item["expected_family"])) for item in emitted)
    positives = [item for item in predictions if item["positive"]]
    negatives = [item for item in predictions if not item["positive"]]
    abstained = [item for item in predictions if item not in emitted]
    return {
        "count": len(predictions),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "emitted_count": len(emitted),
        "typed_recall": round(true_positive / max(len(positives), 1), 6),
        "precision": round(true_positive / max(len(emitted), 1), 6) if emitted else 1.0,
        "false_accept_count": false_accept,
        "false_accept_rate": round(false_accept / max(len(predictions), 1), 6),
        "negative_false_positive_rate": round(sum(int(not item["positive"]) for item in emitted) / max(len(negatives), 1), 6),
        "abstain_rate": round(len(abstained) / max(len(predictions), 1), 6),
        "correct_abstain_rate": round(sum(int(not item["positive"]) for item in abstained) / max(len(abstained), 1), 6) if abstained else 1.0,
    }


def _calibrate(dev: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    candidates = sorted({float(item["confidence"]) for item in dev} | {1.0}, reverse=True)
    best: tuple[int, float] | None = None
    best_threshold = 1.0
    for threshold in candidates:
        metrics = _metrics(dev, threshold=threshold)
        if metrics["false_accept_count"] != 0:
            continue
        score = (metrics["emitted_count"], -threshold)
        if best is None or score > best:
            best = score
            best_threshold = threshold
    return best_threshold, _metrics(dev, threshold=best_threshold)


def _group(predictions: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in predictions:
        groups.setdefault(str(item[key]), []).append(item)
    return {name: _metrics(values, threshold=0.0) for name, values in sorted(groups.items())}


def _markdown(report: dict[str, Any]) -> str:
    calibrated = report["holdout"]["calibrated"]
    return "\n".join([
        "# PG-53 Rule IR 候选训练与族外实现留出",
        "",
        "候选模型只在 PG-35 的可见 response-shape 投影上训练，PG-36 的不同实现/布局作为盲测。typed oracle 只做标签与复核，不进入特征；ordinary_response 被映射为 abstain。",
        "",
        f"设备：`{report['training']['device']}`；训练准确率：`{report['training']['fit']['train_accuracy']:.3f}`。",
        "漏斗审核后特征：" + ", ".join(f"`{name}`" for name in report["training"]["selected_features"]) + "。",
        f"PG-36 校准后 typed recall：`{calibrated['typed_recall']:.3f}`；precision：`{calibrated['precision']:.3f}`；false accept：`{calibrated['false_accept_count']}`；abstain：`{calibrated['abstain_rate']:.3f}`。",
        "",
        "结果仍是 quarantined candidate：需要更多独立实现、族外数据和多种子能力门后，才能考虑训练晋升或长期记忆。",
        "",
        f"训练晋升：`{report['promotion']['training_allowed']}`；长期记忆：`{report['promotion']['memory_promotion_allowed']}`；正式能力声明：`{report['promotion']['formal_claim_allowed']}`。",
    ]) + "\n"


def main() -> int:
    source = json.loads(SOURCE_REPORT_PATH.read_text(encoding="utf-8"))
    funnel = json.loads(FUNNEL_DATASET_PATH.read_text(encoding="utf-8"))
    if funnel.get("review_decision") != "approved_for_downstream_ood_experiment":
        raise RuntimeError("PG-53 feature funnel was not approved by the independent reviewer")
    selected_features = [str(name) for name in funnel.get("accepted_features", [])]
    if not selected_features:
        raise RuntimeError("PG-53 feature funnel accepted no features")
    feature_map = {str(row["sample_id"]): row for row in funnel.get("rows", [])}
    rows = list(source["rows"])
    train_rows = [row for row in rows if row["implementation"] == "pg35" and int(row["sampling_seed"]) in {5301, 5307}]
    dev_rows = [row for row in rows if row["implementation"] == "pg35" and int(row["sampling_seed"]) == 5311]
    holdout_rows = [row for row in rows if row["implementation"] == "pg36"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, fit = _train(train_rows, feature_map, selected_features, device=device)
    dev_predictions = _raw_predictions(model, dev_rows, feature_map, selected_features, fit, device=device)
    threshold, dev_metrics = _calibrate(dev_predictions)
    holdout_predictions = _raw_predictions(model, holdout_rows, feature_map, selected_features, fit, device=device)
    raw_holdout = _metrics(holdout_predictions, threshold=0.0)
    calibrated_holdout = _metrics(holdout_predictions, threshold=threshold)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": "pg53-rule-ir-candidate-checkpoint-v1",
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "families": list(PG53_MODEL_FAMILIES),
        "feature_dim": FEATURE_DIM,
        "normalisation_mean": fit["normalisation_mean"],
        "normalisation_std": fit["normalisation_std"],
        "abstain_threshold": threshold,
        "margin_threshold": MARGIN_THRESHOLD,
        "seed": SEED,
        "device_at_training": str(device),
        "source_holdout": "pg35_train_pg36_holdout",
        "selected_features": selected_features,
        "feature_funnel_review_evidence_sha256": funnel.get("review_evidence_sha256", ""),
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    checkpoint_sha256 = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg53-rule-ir-candidate-report-v1",
        "source_report": str(SOURCE_REPORT_PATH.relative_to(ROOT)),
        "training": {
            "source": "pg35",
            "holdout_source": "pg36",
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "holdout_rows": len(holdout_rows),
            "selected_features": selected_features,
            "feature_funnel_review_evidence_sha256": funnel.get("review_evidence_sha256", ""),
            "seed": SEED,
            "device": str(device),
            "feature_labels_visible": False,
            "oracle_in_features": False,
            "fit": fit,
            "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
            "checkpoint_sha256": checkpoint_sha256,
        },
        "thresholds": {"calibrated": threshold, "margin": MARGIN_THRESHOLD},
        "dev": {"metrics": dev_metrics, "prediction_count": len(dev_predictions)},
        "holdout": {
            "raw": raw_holdout,
            "calibrated": calibrated_holdout,
            "by_source": _group(holdout_predictions, "source_id"),
            "by_seed": _group(holdout_predictions, "sampling_seed"),
            "predictions": holdout_predictions,
        },
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "formal_claim_allowed": False,
            "status": "quarantined_candidate",
            "reason": "one_train_implementation_and_one_holdout_implementation_are_not_enough_for_capability_promotion",
        },
    }
    report["report_sha256"] = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"protocol_id": PROTOCOL_ID, "device": str(device), "threshold": threshold, "dev": dev_metrics, "holdout": calibrated_holdout, "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
