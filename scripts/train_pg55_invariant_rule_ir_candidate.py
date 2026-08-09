"""Train/evaluate the quarantined PG-55 invariant Rule IR candidate.

Training uses PG-53 seeds 5301/5307 and PG-42 ledger/envelope seeds 401/409.
The dev split is PG-53 PG-35 seed 5311 plus PG-42 ledger/envelope seed 419.
PG-42 framed, including the unknown ``template_injection`` family, remains a
blind holdout.  The typed oracle is used only for evaluation labels; a density
abstain gate is calibrated on dev and never sees holdout labels.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg53_rule_ir_candidate import PG53_MODEL_FAMILIES, PG53RuleIRCandidate  # noqa: E402
from app.rule_ir_decoder import FEATURE_DIM  # noqa: E402
from app.web_feature_funnel import build_feature_row  # noqa: E402


PROTOCOL_ID = "pg-pk-55-invariant-rule-ir-candidate-v1"
PG53_REPORT_PATH = ROOT / "research" / "pg53_cross_source_typed_replay_report_v1.json"
PG54_TRACE_PATH = ROOT / "research" / "pg54_pg42_rule_ir_ood_trace_v1.json"
FUNNEL_DATASET_PATH = ROOT / "research" / "pg55_invariant_feature_funnel_dataset_v1.json"
REPORT_PATH = ROOT / "research" / "pg55_invariant_rule_ir_candidate_report_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg55_invariant_rule_ir_candidate_report_v1.md"
OUTPUT_DIR = ROOT / "artifacts" / "pg55-invariant-rule-ir-candidate"
CHECKPOINT_PATH = OUTPUT_DIR / "decoder.pt"
SEED = 20260803
EPOCHS = 180
MARGIN_THRESHOLD = 0.08
KNOWN_FAMILIES = set(PG53_MODEL_FAMILIES)
UNKNOWN_FAMILY = "template_injection"
NEGATIVE_FAMILY = "ordinary_response"


def _load_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pg53 = json.loads(PG53_REPORT_PATH.read_text(encoding="utf-8"))["rows"]
    pg54 = json.loads(PG54_TRACE_PATH.read_text(encoding="utf-8"))["rows"]
    train = [row for row in pg53 if int(row["sampling_seed"]) in {5301, 5307}]
    train += [
        row
        for row in pg54
        if row["variant"] in {"ledger", "envelope"}
        and int(row["sampling_seed"]) in {401, 409}
        and row["family"] in KNOWN_FAMILIES
    ]
    dev = [row for row in pg53 if row["implementation"] == "pg35" and int(row["sampling_seed"]) == 5311]
    dev += [
        row
        for row in pg54
        if row["variant"] in {"ledger", "envelope"}
        and int(row["sampling_seed"]) == 419
        and row["family"] in KNOWN_FAMILIES
    ]
    holdout = [row for row in pg54 if row["variant"] == "framed"]
    return train, dev, holdout


def _features(rows: list[dict[str, Any]], selected_features: list[str]) -> torch.Tensor:
    vectors: list[list[float]] = []
    for row in rows:
        model_features = build_feature_row(row)["model_features"]
        vector = [0.0] * FEATURE_DIM
        for offset, name in enumerate(selected_features):
            try:
                numeric = float(model_features.get(name, 0.0))
            except (TypeError, ValueError):
                numeric = 0.0
            vector[8 + offset] = max(-1.0, min(1.0, numeric / 32.0 if abs(numeric) > 1.0 else numeric))
        vectors.append(vector)
    return torch.tensor(vectors, dtype=torch.float32)


def _labels(rows: list[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor([PG53_MODEL_FAMILIES.index(str(row["family"])) for row in rows], dtype=torch.long)


def _train(rows: list[dict[str, Any]], selected_features: list[str], *, device: torch.device) -> tuple[PG53RuleIRCandidate, dict[str, Any], torch.Tensor, torch.Tensor]:
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    raw = _features(rows, selected_features)
    features_mean = raw.mean(dim=0)
    features_std = raw.std(dim=0, unbiased=False).clamp_min(1e-4)
    features = (raw - features_mean) / features_std
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
    fit = {
        "normalisation_mean": features_mean.tolist(),
        "normalisation_std": features_std.tolist(),
        "train_accuracy": round(best_accuracy, 6),
        "class_counts": {name: int(counts[index]) for index, name in enumerate(PG53_MODEL_FAMILIES) if counts[index] > 0},
        "history_tail": history[-5:],
    }
    return model, fit, features_mean, features_std


@torch.inference_mode()
def _predict(model: PG53RuleIRCandidate, rows: list[dict[str, Any]], selected_features: list[str], mean: torch.Tensor, std: torch.Tensor, *, device: torch.device) -> tuple[list[dict[str, Any]], torch.Tensor]:
    normalized = (_features(rows, selected_features) - mean) / std
    outputs = model.decode(normalized.to(device), abstain_threshold=0.0, margin_threshold=0.0)
    return outputs, normalized


def _base_emits(output: dict[str, Any], threshold: float) -> bool:
    return output["candidate_family"] != NEGATIVE_FAMILY and float(output["confidence"]) >= threshold and float(output["margin"]) >= MARGIN_THRESHOLD


def _metrics(rows: list[dict[str, Any]], outputs: list[dict[str, Any]], *, threshold: float, distances: torch.Tensor | None = None, density_threshold: float | None = None) -> dict[str, Any]:
    accepted: list[int] = []
    for index, output in enumerate(outputs):
        if not _base_emits(output, threshold):
            continue
        if distances is not None and density_threshold is not None and float(distances[index]) > density_threshold:
            continue
        accepted.append(index)
    known_positive = [index for index, row in enumerate(rows) if row["decision"] == "confirmed_positive" and row["family"] in KNOWN_FAMILIES and row["family"] != NEGATIVE_FAMILY]
    unknown_positive = [index for index, row in enumerate(rows) if row["decision"] == "confirmed_positive" and row["family"] == UNKNOWN_FAMILY]
    negatives = [index for index, row in enumerate(rows) if row["decision"] != "confirmed_positive"]
    hits = [index for index in accepted if rows[index]["family"] in KNOWN_FAMILIES and rows[index]["family"] != NEGATIVE_FAMILY and rows[index]["family"] == outputs[index]["candidate_family"]]
    return {
        "count": len(rows),
        "positive_count": sum(int(row["decision"] == "confirmed_positive") for row in rows),
        "negative_count": len(negatives),
        "emitted_count": len(accepted),
        "known_positive_count": len(known_positive),
        "known_family_recall": round(len(hits) / max(len(known_positive), 1), 6),
        "known_wrong_family_count": sum(int(index in accepted and index not in hits) for index in known_positive),
        "unknown_positive_count": len(unknown_positive),
        "unknown_misname_count": sum(int(index in accepted) for index in unknown_positive),
        "unknown_strict_abstain": all(index not in accepted for index in unknown_positive),
        "negative_effect_false_accept_count": sum(int(index in accepted) for index in negatives),
        "negative_effect_false_accept_rate": round(sum(int(index in accepted) for index in negatives) / max(len(negatives), 1), 6),
        "abstain_rate": round(1.0 - len(accepted) / max(len(rows), 1), 6),
    }


def _calibrate_threshold(rows: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    candidates = sorted({float(output["confidence"]) for output in outputs} | {1.0}, reverse=True)
    best_threshold = 1.0
    best_metrics: dict[str, Any] | None = None
    for threshold in candidates:
        metrics = _metrics(rows, outputs, threshold=threshold)
        if metrics["negative_effect_false_accept_count"] or metrics["unknown_misname_count"]:
            continue
        if best_metrics is None or metrics["emitted_count"] > best_metrics["emitted_count"]:
            best_threshold = threshold
            best_metrics = metrics
    return best_threshold, best_metrics or _metrics(rows, outputs, threshold=best_threshold)


def _calibrate_density(rows: list[dict[str, Any]], outputs: list[dict[str, Any]], distances: torch.Tensor, threshold: float) -> tuple[float, dict[str, Any]]:
    best_threshold = 0.0
    best_metrics: dict[str, Any] | None = None
    for candidate in sorted({float(value) for value in distances.tolist()}):
        metrics = _metrics(rows, outputs, threshold=threshold, distances=distances, density_threshold=candidate)
        if metrics["negative_effect_false_accept_count"] or metrics["unknown_misname_count"]:
            continue
        if best_metrics is None or metrics["known_family_recall"] > best_metrics["known_family_recall"] or (metrics["known_family_recall"] == best_metrics["known_family_recall"] and metrics["emitted_count"] > best_metrics["emitted_count"]):
            best_threshold = candidate
            best_metrics = metrics
    return best_threshold, best_metrics or _metrics(rows, outputs, threshold=threshold, distances=distances, density_threshold=best_threshold)


def _prediction_groups(rows: list[dict[str, Any]], outputs: list[dict[str, Any]], threshold: float, distances: torch.Tensor, density_threshold: float) -> dict[str, dict[str, Any]]:
    groups = {
        "all": list(range(len(rows))),
        "holdout_framed": [index for index, row in enumerate(rows) if row["variant"] == "framed"],
        "known_families": [index for index, row in enumerate(rows) if row["family"] in KNOWN_FAMILIES and row["family"] != NEGATIVE_FAMILY],
        "unknown_template_family": [index for index, row in enumerate(rows) if row["family"] == UNKNOWN_FAMILY],
        "negative_control": [index for index, row in enumerate(rows) if row["family"] == NEGATIVE_FAMILY],
    }
    return {
        name: _metrics([rows[index] for index in indexes], [outputs[index] for index in indexes], threshold=threshold, distances=distances[indexes], density_threshold=density_threshold)
        for name, indexes in groups.items()
    }


def main() -> int:
    funnel = json.loads(FUNNEL_DATASET_PATH.read_text(encoding="utf-8"))
    if funnel.get("review_decision") != "approved_for_downstream_ood_experiment":
        raise RuntimeError("PG-55 feature funnel review did not pass")
    selected_features = [str(name) for name in funnel["accepted_features"]]
    train_rows, dev_rows, holdout_rows = _load_rows()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, fit, mean, std = _train(train_rows, selected_features, device=device)
    dev_outputs, dev_vectors = _predict(model, dev_rows, selected_features, mean, std, device=device)
    train_vectors = (_features(train_rows, selected_features) - mean) / std
    dev_distances = torch.cdist(dev_vectors.cpu(), train_vectors.cpu()).min(dim=1).values
    threshold, dev_metrics = _calibrate_threshold(dev_rows, dev_outputs)
    density_threshold, density_dev_metrics = _calibrate_density(dev_rows, dev_outputs, dev_distances, threshold)
    holdout_outputs, holdout_vectors = _predict(model, holdout_rows, selected_features, mean, std, device=device)
    holdout_distances = torch.cdist(holdout_vectors.cpu(), train_vectors.cpu()).min(dim=1).values
    raw_holdout = _metrics(holdout_rows, holdout_outputs, threshold=0.0)
    calibrated_holdout = _metrics(holdout_rows, holdout_outputs, threshold=threshold)
    density_holdout = _metrics(holdout_rows, holdout_outputs, threshold=threshold, distances=holdout_distances, density_threshold=density_threshold)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": "pg55-invariant-rule-ir-candidate-checkpoint-v1",
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "families": list(PG53_MODEL_FAMILIES),
        "feature_dim": FEATURE_DIM,
        "normalisation_mean": fit["normalisation_mean"],
        "normalisation_std": fit["normalisation_std"],
        "abstain_threshold": threshold,
        "margin_threshold": MARGIN_THRESHOLD,
        "density_threshold": density_threshold,
        "seed": SEED,
        "device_at_training": str(device),
        "selected_features": selected_features,
        "feature_funnel_review_evidence_sha256": funnel.get("review_evidence_sha256", ""),
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    checkpoint_sha256 = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg55-invariant-rule-ir-candidate-report-v1",
        "training": {
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "holdout_rows": len(holdout_rows),
            "train_split": "PG-53 seeds 5301/5307 + PG-42 ledger/envelope seeds 401/409",
            "dev_split": "PG-53 PG-35 seed 5311 + PG-42 ledger/envelope seed 419",
            "holdout_split": "PG-42 framed all seeds",
            "selected_features": selected_features,
            "feature_funnel_review_evidence_sha256": funnel.get("review_evidence_sha256", ""),
            "device": str(device),
            "fit": fit,
            "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
            "checkpoint_sha256": checkpoint_sha256,
            "oracle_in_features": False,
            "family_label_in_features": False,
        },
        "thresholds": {
            "confidence": threshold,
            "margin": MARGIN_THRESHOLD,
            "density": density_threshold,
            "density_calibration_source": "PG-55 dev only",
        },
        "dev": {"metrics": dev_metrics, "density_metrics": density_dev_metrics},
        "holdout": {
            "raw": raw_holdout,
            "calibrated": calibrated_holdout,
            "density_gated": density_holdout,
            "groups": _prediction_groups(holdout_rows, holdout_outputs, threshold, holdout_distances, density_threshold),
        },
        "unknown_family_policy": {
            "family": UNKNOWN_FAMILY,
            "model_class_present": False,
            "must_abstain": True,
            "density_gated_misname_count": density_holdout["unknown_misname_count"],
            "density_gated_strict_abstain": density_holdout["unknown_strict_abstain"],
        },
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "formal_claim_allowed": False,
            "status": "quarantined_candidate",
            "reason": "PG-55 is a training-method experiment; framed and template-family holdout must pass independently before any promotion",
        },
    }
    report["report_sha256"] = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    gated = report["holdout"]["density_gated"]
    lines = [
        "# PG-55 不变性 Rule IR 候选",
        "",
        f"设备：`{device}`；训练/开发/盲测：`{len(train_rows)}/{len(dev_rows)}/{len(holdout_rows)}`。",
        "审核后的特征：" + ", ".join(f"`{name}`" for name in selected_features),
        f"盲测 density gate 后 known recall：`{gated['known_family_recall']:.3f}`；unknown misname：`{gated['unknown_misname_count']}`；negative false accept：`{gated['negative_effect_false_accept_count']}`；abstain：`{gated['abstain_rate']:.3f}`。",
        "",
        "该候选仍不训练晋升或写入长期记忆；密度门的安全 abstain 不等于泛化能力证明。",
    ]
    MARKDOWN_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "device": str(device),
        "train_rows": len(train_rows),
        "dev": dev_metrics,
        "holdout_raw": raw_holdout,
        "holdout_density_gated": density_holdout,
        "thresholds": report["thresholds"],
        "report": str(REPORT_PATH.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
