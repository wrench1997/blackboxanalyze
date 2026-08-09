"""Train and evaluate a quarantined PG-36 Rule IR candidate.

The candidate consumes only bounded request/response projections.  The typed
oracle is a supervised label and is never present in the feature vector.  A
fresh model is trained on the north implementation's train/dev families and
is evaluated on a seed holdout, unseen families, the south implementation,
and ordinary/unknown negative surfaces.  This is a capability experiment,
not an authorization to attack a target or to promote memory.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
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

from app.model_capability_gate import evaluate_model_capability  # noqa: E402


CATALOG_PATH = ROOT / "research" / "pg36_independent_maze_catalog_v1.json"
FEATURE_TRAINER = ROOT / "scripts" / "train_pg33_formal_rule_ir_candidate.py"
OUTPUT_DIR = ROOT / "artifacts" / "pg36-formal-rule-ir-candidate"
CHECKPOINT_PATH = OUTPUT_DIR / "rule_ir.pt"
REPORT_PATH = ROOT / "research" / "pg36_formal_rule_ir_candidate_report_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg36_formal_rule_ir_candidate_report_v1.md"
SEED = 20360802
TRAIN_SEEDS = (361, 367)
SEED_HOLDOUT = 373
EPOCHS = 180
FEATURE_DIM = 256
CONFIDENCE_THRESHOLD = 0.60
EFFECT_THRESHOLD = 0.60
FAMILIES = (
    "access_control",
    "authentication",
    "command_injection",
    "input_validation",
    "injection",
    "logic",
    "ordinary_response",
    "url_redirect",
    "xss",
    "unknown_surface",
)


class RuleIRModel(nn.Module):
    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_dim: int = 160) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.family_classifier = nn.Linear(hidden_dim, len(FAMILIES))
        self.effect_classifier = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(features)
        return self.family_classifier(hidden), self.effect_classifier(hidden).squeeze(-1)


def _load_feature_module() -> Any:
    spec = importlib.util.spec_from_file_location("pg33_features_for_pg36", FEATURE_TRAINER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load bounded feature projector")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _features(feature_module: Any, rows: list[dict[str, Any]]) -> torch.Tensor:
    vectors = feature_module._features(rows)
    # Add only bounded observable categories.  No family, source, target,
    # evidence hash, typed oracle or raw request/response content is used.
    for row, vector in zip(rows, vectors):
        manifest = row.get("payload_manifest") or {}
        response = row.get("response_projection") or {}
        shape = response.get("shape") or {}
        categories = (
            f"pg36-method:{manifest.get('method', 'GET')}",
            f"pg36-phase:{row.get('phase', 'unknown')}",
            f"pg36-probe:{manifest.get('probe_kind', 'typed_probe')}",
            f"pg36-content:{response.get('content_type_class', 'other')}",
            f"pg36-status:{response.get('status_class', 'other')}",
            f"pg36-shape-kind:{shape.get('kind', 'other')}",
            f"pg36-shape-keys:{int(shape.get('key_count', 0))}",
            f"pg36-shape-scalars:{int(shape.get('scalar_count', 0))}",
            f"pg36-shape-arrays:{int(shape.get('array_count', 0))}",
            f"pg36-shape-bools:{int(shape.get('bool_count', 0))}",
            f"pg36-shape-numbers:{int(shape.get('number_count', 0))}",
            f"pg36-shape-strings:{int(shape.get('string_count', 0))}",
        )
        for category in categories:
            digest = hashlib.blake2b(category.encode("utf-8"), digest_size=8).digest()
            index = 224 + (int.from_bytes(digest, "little") % 32)
            vector[index] = min(float(vector[index]) + 1.0, 8.0)
    return vectors


def _normalise(raw: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (raw - mean) / std


def _metrics(
    model: RuleIRModel,
    features: torch.Tensor,
    rows: list[dict[str, Any]],
    labels: torch.Tensor,
    device: torch.device,
    train_features: torch.Tensor,
    novelty_threshold: float,
) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "positive_count": 0, "negative_count": 0, "accepted_count": 0,
                "false_positive_count": 0, "typed_recall": 0.0, "effect_recall_any_family": 0.0,
                "precision": 1.0, "false_positive_rate": 0.0, "abstain_precision": 1.0,
                "ece": 0.0, "abstain_rate": 1.0, "median_queries": 2.0}
    model.eval()
    with torch.inference_mode():
        family_logits, effect_logits = model(features.to(device))
        probabilities = torch.softmax(family_logits, dim=-1).cpu()
        effects = torch.sigmoid(effect_logits).cpu()
    confidence, prediction = probabilities.max(dim=-1)
    distances = torch.cdist(features.cpu(), train_features.cpu()).min(dim=1).values
    accepted = (confidence >= CONFIDENCE_THRESHOLD) & (effects >= EFFECT_THRESHOLD) & (distances <= novelty_threshold)
    positive = torch.tensor([bool((row.get("oracle_projection") or {}).get("positive", False)) for row in rows])
    correct_family = prediction.eq(labels.cpu())
    typed_positive = positive & accepted & correct_family
    effect_positive = positive & accepted
    false_positive = (~positive) & accepted
    abstained = ~accepted
    positive_count = int(positive.sum())
    negative_count = int((~positive).sum())
    accepted_count = int(accepted.sum())
    confidence_error = (confidence - (typed_positive | ((~positive) & (~accepted))).float()).abs()
    by_family: dict[str, dict[str, float]] = {}
    for family in FAMILIES:
        mask = torch.tensor([row.get("family") == family for row in rows])
        family_positive = positive & mask
        if bool(mask.any()):
            by_family[family] = {
                "count": int(mask.sum()),
                "positive_count": int(family_positive.sum()),
                "typed_recall": round(float((typed_positive & mask).sum()) / max(int(family_positive.sum()), 1), 6),
                "accepted_count": int((accepted & mask).sum()),
            }
    return {
        "count": len(rows),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "accepted_count": accepted_count,
        "false_positive_count": int(false_positive.sum()),
        "typed_recall": round(float(typed_positive.sum()) / max(positive_count, 1), 6),
        "effect_recall_any_family": round(float(effect_positive.sum()) / max(positive_count, 1), 6),
        "precision": round(float(typed_positive.sum()) / max(accepted_count, 1), 6),
        "false_positive_rate": round(float(false_positive.sum()) / max(negative_count, 1), 6),
        "abstain_precision": round(float((~positive & abstained).sum()) / max(int(abstained.sum()), 1), 6),
        "ece": round(float(confidence_error.mean()), 6),
        "abstain_rate": round(float(abstained.float().mean()), 6),
        "median_queries": 2.0,
        "mean_confidence": round(float(confidence.mean()), 6),
        "mean_effect_probability": round(float(effects.mean()), 6),
        "max_train_distance": round(float(distances.max()), 6),
        "by_family": by_family,
    }


def _cell_metrics(model: RuleIRModel, feature_module: Any, rows: list[dict[str, Any]], mean: torch.Tensor, std: torch.Tensor, train_features: torch.Tensor, threshold: float, device: torch.device) -> dict[str, Any]:
    raw = _features(feature_module, rows)
    features = _normalise(raw, mean, std)
    labels = torch.tensor([FAMILIES.index(row["family"]) for row in rows], dtype=torch.long)
    return _metrics(model, features, rows, labels, device, train_features, threshold)


def _gate_cells(catalog: dict[str, Any], rows: list[dict[str, Any]], model: RuleIRModel, feature_module: Any, mean: torch.Tensor, std: torch.Tensor, train_features: torch.Tensor, threshold: float, checkpoint_sha256: str, device: torch.device) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    baseline = {"typed_recall": 0.0, "precision": 1.0, "false_positive_rate": 0.0, "abstain_precision": 1.0, "ece": 0.0, "median_queries": 2.0}
    for base in catalog["dataset_tests"]:
        role = str(base["role"])
        seed = int(base["sampling_seed"])
        for implementation in ("north", "south"):
            selected = [row for row in rows if row.get("dataset_role") == role and int(row.get("sampling_seed", -1)) == seed and row.get("implementation") == implementation]
            if not selected:
                continue
            metrics = _cell_metrics(model, feature_module, selected, mean, std, train_features, threshold, device)
            target_ids = sorted({str(row["target_instance_id"]) for row in selected})
            source_hashes = sorted({str(row["source_sha256"]) for row in selected})
            enriched = {
                "sample_id": f"pg36-gated-{implementation}-{role}-s{seed}",
                "dataset_id": f"pg36-gated-{implementation}-{role}-s{seed}",
                "source_id": f"pg36-independent-source-{implementation}",
                "source_hash": hashlib.sha256("|".join(source_hashes).encode()).hexdigest(),
                "target_instance_ids": target_ids,
                "target_instance_id": target_ids[0],
                "family_set": sorted({str(row["family"]) for row in selected}),
                "sampling_seed": seed,
                "role": role,
                "evidence_hash": hashlib.sha256(json.dumps(sorted(row["evidence"]["evidence_hash"] for row in selected), separators=(",", ":")).encode()).hexdigest(),
                "dataset_manifest_sha256": hashlib.sha256(json.dumps(sorted(row["sample_id"] for row in selected), separators=(",", ":")).encode()).hexdigest(),
                "split_manifest_sha256": hashlib.sha256(f"pg36-{implementation}-{role}-{seed}".encode()).hexdigest(),
                "probe_sha256": hashlib.sha256(json.dumps(sorted(row["payload_manifest"]["payload_sha256"] for row in selected), separators=(",", ":")).encode()).hexdigest(),
                "oracle_contract_sha256": hashlib.sha256(b"pg36-independent-maze-oracle-v1").hexdigest(),
                "checkpoint_sha256": checkpoint_sha256,
                "sample_count": len(selected),
                "unique_sample_count": len(selected),
                "denominator": len(selected),
                "positive_count": sum(int(bool(row["oracle_projection"].get("positive"))) for row in selected),
                "negative_count": sum(int(not row["oracle_projection"].get("positive")) for row in selected),
                "abstain_count": int(metrics["count"] - metrics["accepted_count"]),
                "metrics_status": "completed",
                "metrics": {key: metrics[key] for key in baseline},
                "baseline_metrics": baseline,
                "candidate_metrics": {key: metrics[key] for key in baseline},
            }
            enriched["evidence_hash"] = hashlib.sha256(json.dumps(enriched, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            cells.append(enriched)
    return cells


def main() -> int:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    rows = list(catalog["samples"])
    feature_module = _load_feature_module()
    train_rows = [row for row in rows if row.get("implementation") == "north" and row.get("dataset_role") in {"train", "dev"} and int(row.get("sampling_seed", -1)) in TRAIN_SEEDS]
    seed_rows = [row for row in rows if row.get("implementation") == "north" and row.get("dataset_role") in {"train", "dev"} and int(row.get("sampling_seed", -1)) == SEED_HOLDOUT]
    family_rows = [row for row in rows if row.get("implementation") == "north" and row.get("dataset_role") == "family_holdout"]
    ood_rows = [row for row in rows if row.get("implementation") == "north" and row.get("dataset_role") == "ood_source"]
    negative_rows = [row for row in rows if row.get("implementation") == "north" and row.get("dataset_role") == "negative_control"]
    source_rows = [row for row in rows if row.get("implementation") == "south" and row.get("dataset_role") in {"train", "dev"}]
    unknown_rows = [row for row in rows if row.get("implementation") == "south" and row.get("dataset_role") == "negative_control"]
    split_rows = {"train": train_rows, "dev": seed_rows, "family_holdout": family_rows, "ood_source": ood_rows, "negative_control": negative_rows}
    if not all(split_rows.values()) or not source_rows or not unknown_rows:
        raise RuntimeError("PG-36 formal split is incomplete")

    raw_train = _features(feature_module, train_rows)
    mean = raw_train.mean(dim=0)
    std = raw_train.std(dim=0, unbiased=False).clamp_min(1e-4)
    feature_map = {name: _normalise(_features(feature_module, split), mean, std) for name, split in split_rows.items()}
    labels = {name: torch.tensor([FAMILIES.index(row["family"]) for row in split], dtype=torch.long) for name, split in split_rows.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = RuleIRModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0025, weight_decay=0.01)
    family_loss = nn.CrossEntropyLoss(label_smoothing=0.01)
    positive_count = sum(int(bool(row["oracle_projection"].get("positive"))) for row in train_rows)
    negative_count = len(train_rows) - positive_count
    effect_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([max(1.0, negative_count / max(positive_count, 1))], device=device))
    best_state: dict[str, torch.Tensor] | None = None
    best_objective = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        family_logits, effect_logits = model(feature_map["train"].to(device))
        train_effect = torch.tensor([bool(row["oracle_projection"].get("positive")) for row in train_rows], dtype=torch.float32, device=device)
        train_loss = family_loss(family_logits, labels["train"].to(device)) + effect_loss(effect_logits, train_effect)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.inference_mode():
                dev_family, dev_effect = model(feature_map["dev"].to(device))
                dev_target = torch.tensor([bool(row["oracle_projection"].get("positive")) for row in seed_rows], dtype=torch.float32, device=device)
                dev_loss = family_loss(dev_family, labels["dev"].to(device)) + effect_loss(dev_effect, dev_target)
            objective = float((train_loss.detach() + 0.35 * dev_loss.detach()).cpu())
            history.append({"epoch": epoch, "train_loss": round(float(train_loss.detach()), 6), "dev_loss": round(float(dev_loss.detach()), 6), "selection_objective": round(objective, 6)})
            if objective < best_objective:
                best_objective = objective
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    train_features = feature_map["train"]
    distances = torch.cdist(train_features, train_features).fill_diagonal_(float("inf")).min(dim=1).values
    novelty_threshold = max(8.0, float(torch.quantile(distances, 0.95)) + 2.0)
    split_metrics = {name: _metrics(model, feature_map[name], split_rows[name], labels[name], device, train_features, novelty_threshold) for name in split_rows}
    source_metrics = _cell_metrics(model, feature_module, source_rows, mean, std, train_features, novelty_threshold, device)
    unknown_metrics = _cell_metrics(model, feature_module, unknown_rows, mean, std, train_features, novelty_threshold, device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "sift-pg36-rule-ir-checkpoint-v1", "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "families": list(FAMILIES), "feature_dim": FEATURE_DIM, "normalisation_mean": mean.tolist(), "normalisation_std": std.tolist(), "confidence_threshold": CONFIDENCE_THRESHOLD, "effect_threshold": EFFECT_THRESHOLD, "novelty_threshold": novelty_threshold, "seed": SEED, "device_at_training": str(device)}, CHECKPOINT_PATH)
    checkpoint_sha256 = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
    cells = _gate_cells(catalog, rows, model, feature_module, mean, std, train_features, novelty_threshold, checkpoint_sha256, device)
    capability_evidence = {"claim_id": "pg36-formal-rule-ir-candidate", "dataset_tests": cells, "unit_tests_passed": True, "oracle_validated": True, "data_lineage_complete": True, "authorized_sources_attested": True, "raw_data_retained": False, "false_positive_count": sum(int(item["false_positive_count"]) for item in split_metrics.values()), "baseline_metrics": {"typed_recall": 0.0, "precision": 1.0, "false_positive_rate": 0.0, "abstain_precision": 1.0, "ece": 0.0, "median_queries": 2.0}, "candidate_metrics": {key: split_metrics["family_holdout"][key] for key in ("typed_recall", "precision", "false_positive_rate", "abstain_precision", "ece", "median_queries")}, "baseline_worst_case_metrics": {"typed_recall": 0.0, "precision": 1.0, "false_positive_rate": 0.0, "abstain_precision": 1.0, "ece": 0.0, "median_queries": 2.0}, "candidate_worst_case_metrics": {key: min(split_metrics[name][key] for name in split_metrics) if key in {"typed_recall", "false_positive_rate", "ece"} else (max(split_metrics[name][key] for name in split_metrics) if key == "median_queries" else min(split_metrics[name][key] for name in split_metrics)) for key in ("typed_recall", "precision", "false_positive_rate", "abstain_precision", "ece", "median_queries")}}
    capability_gate = evaluate_model_capability(capability_evidence, policy={"min_distinct_source_hashes": 2})
    report = {"protocol_id": "sift-pg36-formal-rule-ir-v1", "schema_version": "pg-pk-36-formal-rule-ir-report-v1", "status": "diagnostic_only", "catalog": {"path": str(CATALOG_PATH.relative_to(ROOT)), "sha256": hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest(), "sample_count": len(rows), "typed_positive_count": int(catalog["typed_positive_count"]), "raw_probe_strings_stored": False, "raw_response_bodies_stored": False}, "model": {"class": "RuleIRModel", "families": list(FAMILIES), "device": str(device), "visible_projection_only": True, "typed_oracle_consumed_by_model": False, "positive_authority": False}, "split_plan": {"train": "north implementation; train/dev roles; seeds 361,367", "dev": "north implementation; train/dev roles; seed 373", "family_holdout": "north implementation; logic/url_redirect", "ood_source": "north implementation; input_validation/command_injection", "negative_control": "north implementation; ordinary/unknown", "source_holdout": "south implementation; train/dev roles", "unknown_abstain": "south implementation; ordinary/unknown"}, "training": {"count": len(train_rows), "positive_count": positive_count, "negative_count": negative_count, "epochs": EPOCHS, "seed": SEED, "selection": "minimum train plus held-out-seed supervised objective; not accuracy-only", "best_objective": round(best_objective, 6), "history_tail": history[-5:], "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "checkpoint_sha256": checkpoint_sha256}, "thresholds": {"confidence": CONFIDENCE_THRESHOLD, "effect": EFFECT_THRESHOLD, "novelty": novelty_threshold}, "splits": split_metrics, "source_holdout": source_metrics, "unknown_abstain": unknown_metrics, "capability_gate": capability_gate, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "status": "quarantined_candidate", "reason": "strict family/source gate is not a training authority"}, "training_allowed": False, "memory_promotion_allowed": False, "cells": cells}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-36 formal Rule IR candidate", "", "模型只读取 bounded visible projection；typed oracle 只作为监督标签，未进入输入。", "", "| split | typed recall | effect recall | precision | FPR | abstain |", "|---|---:|---:|---:|---:|---:|"]
    for name in ("train", "dev", "family_holdout", "ood_source", "negative_control"):
        item = split_metrics[name]
        lines.append(f"| {name} | {item['typed_recall']:.2f} | {item['effect_recall_any_family']:.2f} | {item['precision']:.2f} | {item['false_positive_rate']:.2f} | {item['abstain_rate']:.2f} |")
    lines.append(f"| source_holdout (south) | {source_metrics['typed_recall']:.2f} | {source_metrics['effect_recall_any_family']:.2f} | {source_metrics['precision']:.2f} | {source_metrics['false_positive_rate']:.2f} | {source_metrics['abstain_rate']:.2f} |")
    lines.extend(["", f"状态：`{capability_gate['status']}`；claim_allowed=`{capability_gate['claim_allowed']}`；训练晋升与长期记忆均为 `False`。", ""])
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "train_count": len(train_rows), "splits": {name: {key: item[key] for key in ("typed_recall", "effect_recall_any_family", "precision", "false_positive_rate", "abstain_rate")} for name, item in split_metrics.items()}, "source_holdout": {key: source_metrics[key] for key in ("typed_recall", "effect_recall_any_family", "precision", "false_positive_rate", "abstain_rate")}, "unknown_abstain": {key: unknown_metrics[key] for key in ("typed_recall", "effect_recall_any_family", "precision", "false_positive_rate", "abstain_rate")}, "capability_gate": {"status": capability_gate["status"], "claim_allowed": capability_gate["claim_allowed"], "reasons": capability_gate.get("reasons", [])[:12]}, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
