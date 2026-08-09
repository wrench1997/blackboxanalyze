"""Run PG-23: hard-negative, multi-task, abstention-aware training.

The runner consumes only the authorized local Pikachu catalogs already in the
repository.  It never starts a target and never emits raw probe strings in the
report.  Synthetic transport/baseline negatives are offline response-shape
augmentations; they are explicitly marked as such and are not treated as
real replay evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.catalog_rule_decoder import CATALOG_DECODER_FAMILIES  # noqa: E402
from app.payload_catalog import flatten_catalog, load_catalog  # noqa: E402
from app.legacy_experiment_gate import assert_legacy_training_blocked  # noqa: E402
from app.pg23_multitask_decoder import (  # noqa: E402
    PG23MultiTaskDecoder,
    PG23_EVIDENCE_DIM,
    PG23_SCHEMA,
    PG23_SURFACE_ROLES,
    assert_visible_trace_redacted,
    pair_consistency_loss,
    pg23_evidence_vector,
    pg23_feature_vector,
    pg23_labels,
)
from app.rule_ir_decoder import FEATURE_DIM, validate_abstract_rule_ir  # noqa: E402


PROTOCOL_ID = "pg-pk-23-multitask-evidence-bound-v2"
CATALOG_PATHS = (
    ROOT / "research" / "pikachu_counterfactual_catalog_v1.json",
    ROOT / "research" / "pikachu_payload_catalog_v1.json",
)
OUTPUT_DIR = ROOT / "artifacts" / "pg-pk-23-multitask-v2"
REPORT_PATH = ROOT / "research" / "pg_pk_23_multitask_v2.json"
MARKDOWN_PATH = ROOT / "research" / "pg_pk_23_multitask_v2.md"
PROTOCOL_PATH = ROOT / "research" / "pg_pk_23_multitask_protocol_v2.json"
SEEDS = (20260802, 20260803, 20260804)
EPOCHS = 120
SMOKE_EPOCHS = 35
PAIR_WEIGHT = 0.20
SURFACE_WEIGHT = 0.45
EMIT_WEIGHT = 1.20
GRAD_CLIP = 1.0


def _load_rows() -> list[dict[str, Any]]:
    """Load one copy of every authorized sample, preferring counterfactual data."""

    rows_by_id: dict[str, dict[str, Any]] = {}
    for path in CATALOG_PATHS:
        catalog = load_catalog(path)
        for row in flatten_catalog(catalog):
            sample_id = str(row.get("sample_id", ""))
            if sample_id and sample_id not in rows_by_id:
                rows_by_id[sample_id] = row
    rows = list(rows_by_id.values())
    for row in rows:
        assert_visible_trace_redacted(row)
    return rows


def _copy_as_offline_negative(row: dict[str, Any], kind: str) -> dict[str, Any]:
    """Create a response-shape negative without replaying a request."""

    output = copy.deepcopy(row)
    source_id = str(output.get("sample_id", "offline"))
    output["sample_id"] = f"{source_id}-pg23-{kind}"
    output["counterfactual"] = {
        "kind": "negative_control",
        "intervention": kind,
        "source_sample_id": source_id,
    }
    output["rule_ir_result"] = False
    response = dict(output.get("response_projection") or {})
    baseline = dict((row.get("evidence") or {}).get("baseline") or {})
    if kind == "transport_failure":
        response.update({
            "status_code": 0,
            "body_length": 0,
            "headers": {},
            "json_shape": {},
        })
    elif kind == "baseline_replay":
        response.update({
            "status_code": int(baseline.get("status_code", 200)),
            "body_length": int(baseline.get("body_length", 0)),
            "headers": dict(baseline.get("headers") or {}),
            "json_shape": {},
        })
    else:
        raise ValueError(f"unknown PG-23 offline negative: {kind}")
    output["response_projection"] = response
    output["oracle_projection"] = {}
    return output


def _augment_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    augmented = list(rows)
    counts: Counter[str] = Counter()
    # A baseline and a transport failure for each catalog row make the reject
    # head see both normal no-signal responses and invalid transport states.
    for row in rows:
        for kind in ("baseline_replay", "transport_failure"):
            augmented.append(_copy_as_offline_negative(row, kind))
            counts[kind] += 1
    for row in augmented:
        assert_visible_trace_redacted(row)
    return augmented, counts


def _surface(row: dict[str, Any]) -> str:
    value = str((row.get("semantic") or {}).get("surface", "unknown"))
    return value if value in PG23_SURFACE_ROLES else "unknown"


def _encoding(row: dict[str, Any]) -> str:
    value = str((row.get("probe_artifact") or {}).get("encoding", "plain")).casefold()
    if "double" in value and "html" in value:
        return "double_html_entity"
    if "html" in value or "entity" in value:
        return "html_entity"
    if "percent" in value or "url" in value:
        return "url_percent"
    return "plain"


def _source(row: dict[str, Any]) -> str:
    return str(row.get("source_id", "unknown"))


def _split(rows: list[dict[str, Any]], name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if name == "source_holdout":
        # The small staged catalog is a held-out source and includes a family
        # that is not present in the paired training source.
        test_sources = {"pikachu-local-container-pg-pk-01"}
        return ([row for row in rows if _source(row) not in test_sources],
                [row for row in rows if _source(row) in test_sources])
    if name == "hard_negative_holdout":
        def intervention(row: dict[str, Any]) -> str:
            return str((row.get("counterfactual") or {}).get("intervention", ""))
        # Keep baseline negatives in training/calibration.  Hold out only the
        # marker-substitution and transport mechanisms that must generalize.
        train = [
            row for row in rows
            if not bool(row.get("counterfactual")) or intervention(row) == "baseline_replay"
        ]
        test = [
            row for row in rows
            if intervention(row) in {"marker_substitution", "transport_failure"}
        ]
        return train, test
    if name == "encoding_holdout":
        train_variants = {"plain", "url_percent"}
        test_variants = {"html_entity", "double_html_entity"}
        return ([row for row in rows if _encoding(row) in train_variants],
                [row for row in rows if _encoding(row) in test_variants])
    if name == "surface_holdout":
        train_surfaces = {"xss_reflected_get", "sqli_str", "sqli_search"}
        return ([row for row in rows if _surface(row) in train_surfaces],
                [row for row in rows if _surface(row) not in train_surfaces])
    if name == "joint_holdout":
        train_variants = {"plain", "url_percent"}
        train_surfaces = {"xss_reflected_get", "sqli_str", "sqli_search"}
        return (
            [row for row in rows if _encoding(row) in train_variants and _surface(row) in train_surfaces],
            [row for row in rows if _encoding(row) in {"html_entity", "double_html_entity"} and _surface(row) not in train_surfaces],
        )
    raise ValueError(f"unknown PG-23 split: {name}")


def _calibration_split(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fit: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    for row in rows:
        digest = hashlib.sha256(str(row.get("sample_id", "")).encode("utf-8")).digest()
        if digest[0] % 5 == 0:
            calibration.append(row)
        else:
            fit.append(row)
    if not fit or not calibration:
        midpoint = max(1, len(rows) // 2)
        fit, calibration = rows[midpoint:], rows[:midpoint]
    return fit, calibration


def _features(rows: list[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor([pg23_feature_vector(row) for row in rows], dtype=torch.float32)


def _evidence_features(rows: list[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor([pg23_evidence_vector(row) for row in rows], dtype=torch.float32)


def _normalise(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = raw.mean(dim=0)
    std = raw.std(dim=0, unbiased=False).clamp_min(1e-4)
    return (raw - mean) / std, mean, std


def _labels(rows: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    families: list[int] = []
    surfaces: list[int] = []
    emits: list[float] = []
    for row in rows:
        family, surface, emit = pg23_labels(row)
        families.append(family)
        surfaces.append(surface)
        emits.append(emit)
    return (
        torch.tensor(families, dtype=torch.long),
        torch.tensor(surfaces, dtype=torch.long),
        torch.tensor(emits, dtype=torch.float32),
    )


def _class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = torch.ones_like(counts)
    present = counts > 0
    if present.any():
        weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return weights


def _train(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    device: torch.device,
    epochs: int,
) -> tuple[PG23MultiTaskDecoder, dict[str, Any]]:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    raw = _features(rows)
    evidence = _evidence_features(rows)
    features_cpu, mean, std = _normalise(raw)
    family_labels, surface_labels, emit_labels = _labels(rows)
    model = PG23MultiTaskDecoder(hidden_dim=128, embedding_dim=64, dropout=0.06).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.02)
    family_loss_fn = nn.CrossEntropyLoss(
        weight=_class_weights(family_labels, len(CATALOG_DECODER_FAMILIES)).to(device),
        label_smoothing=0.02,
    )
    surface_loss_fn = nn.CrossEntropyLoss(
        weight=_class_weights(surface_labels, len(PG23_SURFACE_ROLES)).to(device),
        label_smoothing=0.02,
    )
    positive_count = float(emit_labels.sum())
    negative_count = float(len(emit_labels) - positive_count)
    pos_weight = torch.tensor([max(1.0, negative_count / max(positive_count, 1.0))], device=device)
    emit_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    features = features_cpu.to(device)
    evidence_features = evidence.to(device)
    target_family = family_labels.to(device)
    target_surface = surface_labels.to(device)
    target_emit = emit_labels.to(device)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(features, evidence_features)
        family_loss = family_loss_fn(output["family_logits"], target_family)
        surface_loss = surface_loss_fn(output["surface_logits"], target_surface)
        emit_loss = emit_loss_fn(output["emit_logits"], target_emit)
        consistency = pair_consistency_loss(output["embedding"], rows)
        loss = family_loss + SURFACE_WEIGHT * surface_loss + EMIT_WEIGHT * emit_loss + PAIR_WEIGHT * consistency
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        history.append({
            "epoch": float(epoch),
            "loss": loss_value,
            "family_loss": float(family_loss.detach().cpu()),
            "surface_loss": float(surface_loss.detach().cpu()),
            "emit_loss": float(emit_loss.detach().cpu()),
            "pair_loss": float(consistency.detach().cpu()),
        })
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {
        "normalisation_mean": mean.tolist(),
        "normalisation_std": std.tolist(),
        "best_loss": round(best_loss, 6),
        "family_support": dict(Counter(int(value) for value in family_labels.tolist())),
        "surface_support": dict(Counter(int(value) for value in surface_labels.tolist())),
        "emit_positive_count": int(positive_count),
        "emit_negative_count": int(negative_count),
        "history_tail": history[-5:],
    }


def _raw_outputs(
    model: PG23MultiTaskDecoder,
    rows: list[dict[str, Any]],
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    features = ((_features(rows) - mean) / std.clamp_min(1e-4)).to(device)
    evidence_features = _evidence_features(rows).to(device)
    with torch.inference_mode():
        output = model(features, evidence_features)
        family_probabilities = torch.softmax(output["family_logits"], dim=-1).cpu()
        surface_probabilities = torch.softmax(output["surface_logits"], dim=-1).cpu()
        emit_probabilities = torch.sigmoid(output["emit_logits"]).cpu()
    values: list[dict[str, Any]] = []
    for row, family_row, surface_row, emit in zip(rows, family_probabilities, surface_probabilities, emit_probabilities):
        order = torch.sort(family_row, descending=True).values
        family_index = int(family_row.argmax())
        surface_index = int(surface_row.argmax())
        values.append({
            "row": row,
            "family_probabilities": family_row.tolist(),
            "surface_probabilities": surface_row.tolist(),
            "candidate_family": CATALOG_DECODER_FAMILIES[family_index],
            "candidate_surface": PG23_SURFACE_ROLES[surface_index],
            "family_confidence": float(family_row[family_index]),
            "family_margin": float(order[0] - order[1]),
            "surface_confidence": float(surface_row[surface_index]),
            "emit_probability": float(emit),
        })
    return values


def _choose_thresholds(outputs: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """Choose the highest-coverage threshold with zero calibration false accepts."""

    if not outputs:
        return {"family_threshold": 1.0, "emit_threshold": 1.0, "margin_threshold": 1.0, "accepted": 0}
    candidates = [round(value, 2) for value in torch.linspace(0.20, 0.98, 40).tolist()]
    family_candidates = (0.40, 0.50, 0.60, 0.70)
    margin_candidates = (0.00, 0.05, 0.10, 0.20)
    best: tuple[int, float, float, float] | None = None
    for family_threshold in family_candidates:
        for margin_threshold in margin_candidates:
            for emit_threshold in candidates:
                accepted = [
                    output for output in outputs
                    if output["family_confidence"] >= family_threshold
                    and output["family_margin"] >= margin_threshold
                    and output["emit_probability"] >= emit_threshold
                ]
                false_accepts = 0
                true_accepts = 0
                for output in accepted:
                    expected_family, _, expected_emit = pg23_labels(output["row"])
                    family_matches = output["candidate_family"] == CATALOG_DECODER_FAMILIES[expected_family]
                    if bool(expected_emit) and family_matches:
                        true_accepts += 1
                    else:
                        false_accepts += 1
                if false_accepts:
                    continue
                score = (true_accepts, -emit_threshold, -family_threshold, -margin_threshold)
                if best is None or score > best:
                    best = score
    if best is None:
        return {"family_threshold": 1.0, "emit_threshold": 1.0, "margin_threshold": 1.0, "accepted": 0}
    _, neg_emit, neg_family, neg_margin = best
    return {
        "family_threshold": round(-neg_family, 2),
        "emit_threshold": round(-neg_emit, 2),
        "margin_threshold": round(-neg_margin, 2),
        "accepted": int(best[0]),
    }


def _ece(outputs: list[dict[str, Any]]) -> float:
    if not outputs:
        return 0.0
    bins: list[list[dict[str, Any]]] = [[] for _ in range(10)]
    for output in outputs:
        index = min(9, int(float(output["emit_probability"]) * 10.0))
        bins[index].append(output)
    total = float(len(outputs))
    value = 0.0
    for bucket in bins:
        if not bucket:
            continue
        confidence = sum(float(item["emit_probability"]) for item in bucket) / len(bucket)
        accuracy = sum(bool(pg23_labels(item["row"])[2]) for item in bucket) / len(bucket)
        value += len(bucket) / total * abs(confidence - accuracy)
    return round(value, 6)


def _evaluate(
    model: PG23MultiTaskDecoder,
    rows: list[dict[str, Any]],
    fit: dict[str, Any],
    thresholds: dict[str, float | int | None],
    *,
    device: torch.device,
) -> dict[str, Any]:
    mean = torch.tensor(fit["normalisation_mean"], dtype=torch.float32)
    std = torch.tensor(fit["normalisation_std"], dtype=torch.float32)
    outputs = _raw_outputs(model, rows, mean, std, device=device)
    predictions: list[dict[str, Any]] = []
    for output in outputs:
        row = output["row"]
        expected_family, expected_surface, expected_emit = pg23_labels(row)
        accepted = (
            output["family_confidence"] >= float(thresholds["family_threshold"])
            and output["emit_probability"] >= float(thresholds["emit_threshold"])
            and output["family_margin"] >= float(thresholds["margin_threshold"])
        )
        expected_family_name = CATALOG_DECODER_FAMILIES[expected_family]
        predicted_family = output["candidate_family"] if accepted else None
        rule_ir = None
        if accepted:
            from app.catalog_rule_decoder import abstract_catalog_rule_ir
            rule_ir = abstract_catalog_rule_ir(output["candidate_family"])
            validate_abstract_rule_ir(rule_ir)
        predictions.append({
            "sample_id": str(row.get("sample_id", "")),
            "source_id": _source(row),
            "expected_family": expected_family_name,
            "predicted_family": predicted_family,
            "expected_surface": PG23_SURFACE_ROLES[expected_surface],
            "predicted_surface": output["candidate_surface"],
            "expected_emit": bool(expected_emit),
            "accepted": bool(accepted),
            "abstained": not bool(accepted),
            "family_confidence": round(output["family_confidence"], 6),
            "family_margin": round(output["family_margin"], 6),
            "emit_probability": round(output["emit_probability"], 6),
            "exit_found": bool(accepted and expected_emit and predicted_family == expected_family_name),
            "false_positive": bool(accepted and (not expected_emit or predicted_family != expected_family_name)),
            "family_error": bool(accepted and predicted_family != expected_family_name),
            "rule_ir_emitted": bool(rule_ir is not None),
        })
    total = len(predictions)
    positive = sum(row["expected_emit"] for row in predictions)
    negative = total - positive
    accepted = sum(row["accepted"] for row in predictions)
    return {
        "total": total,
        "positive_count": positive,
        "negative_count": negative,
        "accepted_count": accepted,
        "exit_found_rate": round(sum(row["exit_found"] for row in predictions) / max(total, 1), 6),
        "positive_recall": round(sum(row["exit_found"] for row in predictions) / max(positive, 1), 6),
        "false_positive_rate": round(sum(row["false_positive"] for row in predictions) / max(total, 1), 6),
        "false_accept_rate_negative": round(
            sum(row["false_positive"] and not row["expected_emit"] for row in predictions) / max(negative, 1),
            6,
        ),
        "unsafe_accept_rate": round(sum(row["false_positive"] for row in predictions) / max(total, 1), 6),
        "family_error_rate_accepted": round(sum(row["family_error"] for row in predictions) / max(accepted, 1), 6),
        "abstain_rate": round(sum(row["abstained"] for row in predictions) / max(total, 1), 6),
        "rule_ir_emission_rate": round(sum(row["rule_ir_emitted"] for row in predictions) / max(total, 1), 6),
        "emit_ece": _ece([{"emit_probability": row["emit_probability"], "row": next(item for item in rows if item.get("sample_id") == row["sample_id"])} for row in predictions]),
        "predictions": predictions,
    }


def _compact_training_summary(fit: dict[str, Any]) -> dict[str, Any]:
    return {
        "best_loss": fit["best_loss"],
        "emit_positive_count": fit["emit_positive_count"],
        "emit_negative_count": fit["emit_negative_count"],
        "history_tail": fit["history_tail"],
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PG-23 Pikachu 多任务硬负样本训练",
        "",
        "模型同时学习漏洞族、表面角色和是否允许发射抽象 Rule IR。输入投影会抹掉原始 marker、路径族名、counterfactual 名称和 oracle 值；训练数据只来自授权 loopback Catalog，离线负样本不代表真实回放。",
        "",
        "## 评估",
        "",
        "| split | seed | total | positive recall | false accept (negative) | abstain | Rule IR emission |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["evaluations"]:
        metrics = row["metrics"]
        lines.append(
            f"| {row['split']} | {row['seed']} | {metrics['total']} | {metrics['positive_recall']:.2f} | "
            f"{metrics['false_accept_rate_negative']:.2f} | {metrics['abstain_rate']:.2f} | {metrics['rule_ir_emission_rate']:.2f} |"
        )
    lines.extend([
        "",
        "## 当前结论",
        "",
        "PG-23 的安全目标是先消除硬负样本误报，再逐步提高有 bounded evidence 的接受率。任何 transport failure、无 evidence 或族外表面都只能 abstain；Rule IR 是受限模板，不是自由代码生成。",
        "",
        f"完整 JSON：`{report['report_path']}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    started = time.perf_counter()
    # The catalog is intentionally tiny.  Limiting BLAS threads avoids a
    # Windows/PyTorch thread explosion that can make a small full-batch run
    # consume more memory than the actual model.
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    smoke = os.environ.get("PG23_SMOKE", "") == "1"
    assert_legacy_training_blocked(CATALOG_PATHS)
    base_rows = _load_rows()
    rows, augmentation_counts = _augment_rows(base_rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluations: list[dict[str, Any]] = []
    checkpoint_paths: list[str] = []
    split_names = (("source_holdout",) if smoke else ("source_holdout", "hard_negative_holdout", "encoding_holdout", "surface_holdout", "joint_holdout"))
    seeds = (SEEDS[0],) if smoke else SEEDS
    epochs = SMOKE_EPOCHS if smoke else EPOCHS
    for split_name in split_names:
        train_rows, test_rows = _split(rows, split_name)
        if not train_rows or not test_rows:
            continue
        fit_rows, calibration_rows = _calibration_split(train_rows)
        for seed in seeds:
            model, fit = _train(fit_rows, seed=seed, device=device, epochs=epochs)
            mean = torch.tensor(fit["normalisation_mean"], dtype=torch.float32)
            std = torch.tensor(fit["normalisation_std"], dtype=torch.float32)
            calibration_outputs = _raw_outputs(model, calibration_rows, mean, std, device=device)
            thresholds = _choose_thresholds(calibration_outputs)
            metrics = _evaluate(model, test_rows, fit, thresholds, device=device)
            evaluations.append({
                "split": split_name,
                "seed": seed,
                "train_count": len(fit_rows),
                "calibration_count": len(calibration_rows),
                "test_count": len(test_rows),
                "thresholds": thresholds,
                "training": _compact_training_summary(fit),
                "metrics": metrics,
            })
            if split_name == "source_holdout" and seed == SEEDS[0]:
                checkpoint_dir = OUTPUT_DIR / split_name
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / "decoder.pt"
                torch.save({
                    "schema_version": PG23_SCHEMA,
                    "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "feature_dim": FEATURE_DIM,
                    "evidence_dim": PG23_EVIDENCE_DIM,
                    "families": list(CATALOG_DECODER_FAMILIES),
                    "surfaces": list(PG23_SURFACE_ROLES),
                    "normalisation_mean": fit["normalisation_mean"],
                    "normalisation_std": fit["normalisation_std"],
                    "thresholds": thresholds,
                    "seed": seed,
                    "device_at_training": str(device),
                }, checkpoint_path)
                checkpoint_paths.append(str(checkpoint_path.relative_to(ROOT)))
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "sift-pikachu-pg23-multitask-report-v1",
        "catalogs": [str(path.relative_to(ROOT)) for path in CATALOG_PATHS],
        "base_sample_count": len(base_rows),
        "augmented_sample_count": len(rows),
        "augmentation_counts": dict(augmentation_counts),
        "model": {
            "class": "PG23MultiTaskDecoder",
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "feature_dim": FEATURE_DIM,
            "evidence_dim": PG23_EVIDENCE_DIM,
            "families": list(CATALOG_DECODER_FAMILIES),
            "surfaces": list(PG23_SURFACE_ROLES),
            "free_form_payload_generation": False,
            "evidence_bound_rule_ir_head": True,
            "losses": ["family cross entropy", "surface cross entropy", "emit BCE", "same-pair embedding consistency"],
        },
        "split_design": {
            "source_holdout": "staged local source held out",
            "hard_negative_holdout": "counterfactual and offline response-shape negatives held out",
            "encoding_holdout": "plain/url_percent train; html_entity/double_html_entity test",
            "surface_holdout": "reflected_get/sqli_str/sqli_search train; remaining surfaces test",
            "joint_holdout": "encoding and surface held out together",
            "labels_hidden_from_visible_trace": True,
        },
        "target_scope": {
            "loopback_only": True,
            "external_network": False,
            "script_execution": False,
            "database_write": False,
            "raw_probe_strings_in_report": False,
        },
        "evaluations": evaluations,
        "checkpoints": checkpoint_paths,
        "report_path": str(REPORT_PATH.relative_to(ROOT)),
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "device": report["model"]["device"],
        "base_sample_count": report["base_sample_count"],
        "augmented_sample_count": report["augmented_sample_count"],
        "evaluations": [
            {
                "split": row["split"],
                "seed": row["seed"],
                **{key: row["metrics"][key] for key in ("total", "positive_recall", "false_accept_rate_negative", "abstain_rate", "rule_ir_emission_rate")},
            }
            for row in evaluations
        ],
        "report": report["report_path"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
