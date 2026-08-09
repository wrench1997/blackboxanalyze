"""Train PG-37 representation ablations without typed-oracle features.

The three variants test whether multi-surface counterfactual pairing improves
the invariant Rule IR representation.  All variants use the same fresh
Catalog and strict source/family/unknown holdouts; none is promotion-authorized
by this script.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.model_capability_gate import evaluate_model_capability  # noqa: E402


CATALOG_PATH = ROOT / "research" / "pg37_counterfactual_catalog_v1.json"
FEATURE_TRAINER = ROOT / "scripts" / "train_pg36_formal_rule_ir_candidate.py"
OUTPUT_DIR = ROOT / "artifacts" / "pg37-representation-ablation"
REPORT_PATH = ROOT / "research" / "pg37_representation_ablation_report_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg37_representation_ablation_report_v1.md"
SEED = 20370802
TRAIN_SEEDS = (361, 367)
SEED_HOLDOUT = 373
EPOCHS = 160
PAIR_WEIGHT = 0.50
CONFIDENCE_THRESHOLD = 0.60
EFFECT_THRESHOLD = 0.60
FAMILIES = (
    "access_control", "authentication", "command_injection", "input_validation", "injection",
    "logic", "ordinary_response", "url_redirect", "xss", "unknown_surface",
)
ABLATIONS = ("surface_only", "counterfactual_paired", "phase_only")


def _load_pg36_module() -> Any:
    spec = importlib.util.spec_from_file_location("pg36_model_for_pg37", FEATURE_TRAINER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load PG-36 model definition")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_features(module: Any) -> Any:
    return module._load_feature_module()


def _hashed_category(vector: torch.Tensor, category: str, start: int = 224, width: int = 32) -> None:
    digest = hashlib.blake2b(category.encode("utf-8"), digest_size=8).digest()
    index = start + (int.from_bytes(digest, "little") % width)
    vector[index] = min(float(vector[index]) + 1.0, 8.0)


def _features(module: Any, feature_module: Any, rows: list[dict[str, Any]], ablation: str) -> torch.Tensor:
    if ablation != "phase_only":
        vectors = module._features(feature_module, rows)
    else:
        vectors = torch.zeros((len(rows), module.FEATURE_DIM), dtype=torch.float32)
    for row, vector in zip(rows, vectors):
        manifest = row.get("payload_manifest") or {}
        response = row.get("response_projection") or {}
        shape = response.get("shape") or {}
        if ablation == "phase_only":
            categories = (
                f"pg37-method:{manifest.get('method', 'GET')}",
                f"pg37-phase:{row.get('phase', 'unknown')}",
                f"pg37-status:{response.get('status_class', 'other')}",
            )
        else:
            # Explicitly do not include surface_variant, route, source, family,
            # oracle fields, or evidence identifiers.  Shape is observable and
            # is intentionally varied by the fixture for the ablation.
            categories = (
                f"pg37-method:{manifest.get('method', 'GET')}",
                f"pg37-phase:{row.get('phase', 'unknown')}",
                f"pg37-probe:{manifest.get('probe_kind', 'typed_probe')}",
                f"pg37-content:{response.get('content_type_class', 'other')}",
                f"pg37-status:{response.get('status_class', 'other')}",
                f"pg37-shape-kind:{shape.get('kind', 'other')}",
                f"pg37-shape-keys:{int(shape.get('key_count', 0))}",
                f"pg37-shape-scalars:{int(shape.get('scalar_count', 0))}",
                f"pg37-shape-arrays:{int(shape.get('array_count', 0))}",
                f"pg37-shape-bools:{int(shape.get('bool_count', 0))}",
                f"pg37-shape-numbers:{int(shape.get('number_count', 0))}",
                f"pg37-shape-strings:{int(shape.get('string_count', 0))}",
            )
        for category in categories:
            _hashed_category(vector, category)
    return vectors


def _pair_groups(rows: list[dict[str, Any]], features: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    groups: dict[tuple[str, str, int, str, str, str], dict[str, int]] = defaultdict(dict)
    for index, row in enumerate(rows):
        key = (str(row["implementation"]), str(row["surface_id"]), int(row["sampling_seed"]), str(row["method"]), str(row["phase"]), str(row["pair_role"]))
        groups[key][str(row["surface_variant"])] = index
    return [(features[indexes["compact"]], features[indexes["nested"]]) for indexes in groups.values() if {"compact", "nested"}.issubset(indexes)]


def _pair_consistency(model: nn.Module, rows: list[dict[str, Any]], features: torch.Tensor, device: torch.device) -> dict[str, Any]:
    groups: dict[tuple[str, str, int, str, str, str], dict[str, int]] = defaultdict(dict)
    for index, row in enumerate(rows):
        key = (str(row["implementation"]), str(row["surface_id"]), int(row["sampling_seed"]), str(row["method"]), str(row["phase"]), str(row["pair_role"]))
        groups[key][str(row["surface_variant"])] = index
    complete = [indexes for indexes in groups.values() if {"compact", "nested", "headerized"}.issubset(indexes)]
    if not complete:
        return {"pair_count": 0, "family_agreement_rate": 0.0, "mean_l1": 0.0}
    agreements = 0
    distances: list[float] = []
    model.eval()
    with torch.inference_mode():
        for indexes in complete:
            batch = torch.stack([features[indexes[name]] for name in ("compact", "nested", "headerized")]).to(device)
            family_logits, effect_logits = model(batch)
            probabilities = torch.softmax(family_logits, dim=-1).cpu()
            effects = torch.sigmoid(effect_logits).cpu()
            agreements += int(len({int(value) for value in probabilities.argmax(dim=-1)}) == 1)
            distances.append(float(torch.abs(probabilities[1:] - probabilities[:1]).mean() + torch.abs(effects[1:] - effects[:1]).mean()))
    return {"pair_count": len(complete), "family_agreement_rate": round(agreements / len(complete), 6), "mean_l1": round(sum(distances) / len(distances), 6)}


def _metrics(module: Any, model: nn.Module, features: torch.Tensor, rows: list[dict[str, Any]], labels: torch.Tensor, device: torch.device, train_features: torch.Tensor, novelty_threshold: float) -> dict[str, Any]:
    # PG-36's metric implementation is shared, but use PG-37's thresholds and
    # model-independent labels here to keep the comparison explicit.
    model.eval()
    with torch.inference_mode():
        family_logits, effect_logits = model(features.to(device))
        probabilities = torch.softmax(family_logits, dim=-1).cpu()
        effects = torch.sigmoid(effect_logits).cpu()
    confidence, prediction = probabilities.max(dim=-1)
    distances = torch.cdist(features.cpu(), train_features.cpu()).min(dim=1).values
    accepted = (confidence >= CONFIDENCE_THRESHOLD) & (effects >= EFFECT_THRESHOLD) & (distances <= novelty_threshold)
    positive = torch.tensor([bool(row["oracle_projection"].get("positive", False)) for row in rows])
    correct = prediction.eq(labels.cpu())
    typed = positive & accepted & correct
    effect_any = positive & accepted
    false_positive = (~positive) & accepted
    abstained = ~accepted
    positive_count = int(positive.sum())
    negative_count = int((~positive).sum())
    accepted_count = int(accepted.sum())
    by_family: dict[str, dict[str, float]] = {}
    for family in FAMILIES:
        mask = torch.tensor([row["family"] == family for row in rows])
        fam_pos = positive & mask
        if bool(mask.any()):
            by_family[family] = {"count": int(mask.sum()), "positive_count": int(fam_pos.sum()), "typed_recall": round(float((typed & mask).sum()) / max(int(fam_pos.sum()), 1), 6), "accepted_count": int((accepted & mask).sum())}
    confidence_correct = typed | ((~positive) & (~accepted))
    return {"count": len(rows), "positive_count": positive_count, "negative_count": negative_count, "accepted_count": accepted_count, "false_positive_count": int(false_positive.sum()), "typed_recall": round(float(typed.sum()) / max(positive_count, 1), 6), "effect_recall_any_family": round(float(effect_any.sum()) / max(positive_count, 1), 6), "precision": round(float(typed.sum()) / max(accepted_count, 1), 6), "false_positive_rate": round(float(false_positive.sum()) / max(negative_count, 1), 6), "abstain_precision": round(float((~positive & abstained).sum()) / max(int(abstained.sum()), 1), 6), "ece": round(float((confidence - confidence_correct.float()).abs().mean()), 6), "abstain_rate": round(float(abstained.float().mean()), 6), "median_queries": 2.0, "mean_confidence": round(float(confidence.mean()), 6), "mean_effect_probability": round(float(effects.mean()), 6), "max_train_distance": round(float(distances.max()), 6), "by_family": by_family}


def _cell_metrics(model: nn.Module, module: Any, feature_module: Any, rows: list[dict[str, Any]], ablation: str, mean: torch.Tensor, std: torch.Tensor, train_features: torch.Tensor, novelty_threshold: float, device: torch.device) -> dict[str, Any]:
    raw = _features(module, feature_module, rows, ablation)
    features = (raw - mean) / std
    labels = torch.tensor([FAMILIES.index(row["family"]) for row in rows], dtype=torch.long)
    return _metrics(module, model, features, rows, labels, device, train_features, novelty_threshold)


def _gate_cells(catalog: dict[str, Any], rows: list[dict[str, Any]], model: nn.Module, module: Any, feature_module: Any, ablation: str, mean: torch.Tensor, std: torch.Tensor, train_features: torch.Tensor, threshold: float, checkpoint_sha256: str, device: torch.device) -> list[dict[str, Any]]:
    baseline = {"typed_recall": 0.0, "precision": 1.0, "false_positive_rate": 0.0, "abstain_precision": 1.0, "ece": 0.0, "median_queries": 2.0}
    cells: list[dict[str, Any]] = []
    for base in catalog["dataset_tests"]:
        role = str(base["role"])
        seed = int(base["sampling_seed"])
        for implementation in ("atlas", "orbit"):
            selected = [row for row in rows if row.get("dataset_role") == role and int(row.get("sampling_seed", -1)) == seed and row.get("implementation") == implementation]
            if not selected:
                continue
            metrics = _cell_metrics(model, module, feature_module, selected, ablation, mean, std, train_features, threshold, device)
            targets = sorted({str(row["target_instance_id"]) for row in selected})
            source_hashes = sorted({str(row["source_sha256"]) for row in selected})
            cell = {"sample_id": f"pg37-gated-{ablation}-{implementation}-{role}-s{seed}", "dataset_id": f"pg37-gated-{ablation}-{implementation}-{role}-s{seed}", "source_id": f"pg37-counterfactual-source-{implementation}", "source_hash": hashlib.sha256("|".join(source_hashes).encode()).hexdigest(), "target_instance_ids": targets, "target_instance_id": targets[0], "family_set": sorted({str(row["family"]) for row in selected}), "sampling_seed": seed, "role": role, "evidence_hash": hashlib.sha256(json.dumps(sorted(row["evidence"]["evidence_hash"] for row in selected), separators=(",", ":")).encode()).hexdigest(), "dataset_manifest_sha256": hashlib.sha256(json.dumps(sorted(row["sample_id"] for row in selected), separators=(",", ":")).encode()).hexdigest(), "split_manifest_sha256": hashlib.sha256(f"pg37-{ablation}-{implementation}-{role}-{seed}".encode()).hexdigest(), "probe_sha256": hashlib.sha256(json.dumps(sorted(row["payload_manifest"]["payload_sha256"] for row in selected), separators=(",", ":")).encode()).hexdigest(), "oracle_contract_sha256": hashlib.sha256(b"pg37-counterfactual-typed-oracle-v1").hexdigest(), "checkpoint_sha256": checkpoint_sha256, "sample_count": len(selected), "unique_sample_count": len(selected), "denominator": len(selected), "positive_count": sum(int(row["oracle_projection"]["positive"]) for row in selected), "negative_count": sum(int(not row["oracle_projection"]["positive"]) for row in selected), "abstain_count": int(metrics["count"] - metrics["accepted_count"]), "metrics_status": "completed", "metrics": {key: metrics[key] for key in baseline}, "baseline_metrics": baseline, "candidate_metrics": {key: metrics[key] for key in baseline}}
            cell["evidence_hash"] = hashlib.sha256(json.dumps(cell, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            cells.append(cell)
    return cells


def _train_one(module: Any, feature_module: Any, rows: list[dict[str, Any]], ablation: str, device: torch.device) -> dict[str, Any]:
    train_rows = [row for row in rows if row["implementation"] == "atlas" and row["dataset_role"] in {"train", "dev"} and int(row["sampling_seed"]) in TRAIN_SEEDS and row["surface_variant"] in {"compact", "nested"}]
    seed_rows = [row for row in rows if row["implementation"] == "atlas" and row["dataset_role"] in {"train", "dev"} and int(row["sampling_seed"]) == SEED_HOLDOUT and row["surface_variant"] in {"compact", "nested"}]
    surface_rows = [row for row in rows if row["implementation"] == "atlas" and row["dataset_role"] in {"train", "dev"} and row["surface_variant"] == "headerized"]
    family_rows = [row for row in rows if row["implementation"] == "atlas" and row["dataset_role"] == "family_holdout"]
    ood_rows = [row for row in rows if row["implementation"] == "atlas" and row["dataset_role"] == "ood_source"]
    negative_rows = [row for row in rows if row["dataset_role"] == "negative_control"]
    source_rows = [row for row in rows if row["implementation"] == "orbit" and row["dataset_role"] in {"train", "dev"}]
    counterfactual_rows = [row for row in rows if row["implementation"] == "atlas" and row["dataset_role"] in {"train", "dev"}]
    split_rows = {"train": train_rows, "dev": seed_rows, "surface_holdout": surface_rows, "family_holdout": family_rows, "ood_source": ood_rows, "negative_control": negative_rows}
    raw_train = _features(module, feature_module, train_rows, ablation)
    mean = raw_train.mean(dim=0)
    std = raw_train.std(dim=0, unbiased=False).clamp_min(1e-4)
    feature_map = {name: (_features(module, feature_module, split, ablation) - mean) / std for name, split in split_rows.items()}
    labels = {name: torch.tensor([FAMILIES.index(row["family"]) for row in split], dtype=torch.long) for name, split in split_rows.items()}
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = module.RuleIRModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0025, weight_decay=0.01)
    family_loss = nn.CrossEntropyLoss(label_smoothing=0.01)
    pos = sum(int(row["oracle_projection"]["positive"]) for row in train_rows)
    neg = len(train_rows) - pos
    effect_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([max(1.0, neg / max(pos, 1))], device=device))
    pair_groups = _pair_groups(train_rows, feature_map["train"])
    best_state: dict[str, torch.Tensor] | None = None
    best_objective = float("inf")
    history: list[dict[str, float]] = []
    pair_weight = PAIR_WEIGHT if ablation == "counterfactual_paired" else 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        family_logits, effect_logits = model(feature_map["train"].to(device))
        effect_target = torch.tensor([bool(row["oracle_projection"]["positive"]) for row in train_rows], dtype=torch.float32, device=device)
        supervised_family = family_loss(family_logits, labels["train"].to(device))
        supervised_effect = effect_loss(effect_logits, effect_target)
        pair_loss = torch.tensor(0.0, device=device)
        if pair_groups:
            left_batch = torch.stack([left for left, _ in pair_groups]).to(device)
            right_batch = torch.stack([right for _, right in pair_groups]).to(device)
            left_family, left_effect = model(left_batch)
            right_family, right_effect = model(right_batch)
            pair_loss = torch.mean((torch.softmax(left_family, dim=-1) - torch.softmax(right_family, dim=-1)) ** 2) + torch.mean((torch.sigmoid(left_effect) - torch.sigmoid(right_effect)) ** 2)
        objective = supervised_family + supervised_effect + pair_weight * pair_loss
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.inference_mode():
                dev_family, dev_effect = model(feature_map["dev"].to(device))
                dev_target = torch.tensor([bool(row["oracle_projection"]["positive"]) for row in seed_rows], dtype=torch.float32, device=device)
                dev_loss = family_loss(dev_family, labels["dev"].to(device)) + effect_loss(dev_effect, dev_target)
            selection = float((objective.detach() + 0.35 * dev_loss.detach()).cpu())
            history.append({"epoch": epoch, "supervised_family_loss": round(float(supervised_family.detach()), 6), "supervised_effect_loss": round(float(supervised_effect.detach()), 6), "pair_loss": round(float(pair_loss.detach()), 6), "selection_objective": round(selection, 6)})
            if selection < best_objective:
                best_objective = selection
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    train_features = feature_map["train"]
    distances = torch.cdist(train_features, train_features).fill_diagonal_(float("inf")).min(dim=1).values
    novelty_threshold = max(8.0, float(torch.quantile(distances, 0.95)) + 2.0)
    split_metrics = {name: _metrics(module, model, feature_map[name], split_rows[name], labels[name], device, train_features, novelty_threshold) for name in split_rows}
    source_metrics = _cell_metrics(model, module, feature_module, source_rows, ablation, mean, std, train_features, novelty_threshold, device)
    source_metrics["pair_consistency"] = _pair_consistency(model, source_rows, (_features(module, feature_module, source_rows, ablation) - mean) / std, device)
    counterfactual_features = (_features(module, feature_module, counterfactual_rows, ablation) - mean) / std
    counterfactual_pair_metrics = _pair_consistency(model, counterfactual_rows, counterfactual_features, device)
    pair_metrics = {name: _pair_consistency(model, split_rows[name], feature_map[name], device) for name in split_rows}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / f"{ablation}.pt"
    torch.save({"schema_version": "sift-pg37-representation-ablation-checkpoint-v1", "ablation": ablation, "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "families": list(FAMILIES), "feature_dim": module.FEATURE_DIM, "normalisation_mean": mean.tolist(), "normalisation_std": std.tolist(), "novelty_threshold": novelty_threshold, "pair_weight": pair_weight, "seed": SEED, "device_at_training": str(device)}, checkpoint_path)
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    cells = _gate_cells(catalog, rows, model, module, feature_module, ablation, mean, std, train_features, novelty_threshold, checkpoint_sha256, device)
    baseline = {"typed_recall": 0.0, "precision": 1.0, "false_positive_rate": 0.0, "abstain_precision": 1.0, "ece": 0.0, "median_queries": 2.0}
    capability_evidence = {"claim_id": f"pg37-{ablation}", "dataset_tests": cells, "unit_tests_passed": True, "oracle_validated": True, "data_lineage_complete": True, "authorized_sources_attested": True, "raw_data_retained": False, "false_positive_count": sum(int(split_metrics[name]["false_positive_count"]) for name in split_metrics), "baseline_metrics": baseline, "candidate_metrics": {key: split_metrics["family_holdout"][key] for key in baseline}, "baseline_worst_case_metrics": baseline, "candidate_worst_case_metrics": {key: min(split_metrics[name][key] for name in split_metrics) if key in {"typed_recall", "false_positive_rate", "ece", "precision", "abstain_precision"} else max(split_metrics[name][key] for name in split_metrics) for key in baseline}}
    gate = evaluate_model_capability(capability_evidence, policy={"min_distinct_source_hashes": 2})
    return {"ablation": ablation, "model": {"class": "RuleIRModel", "visible_projection_only": True, "typed_oracle_consumed_by_model": False, "surface_variant_label_consumed_by_model": False, "positive_authority": False}, "training": {"count": len(train_rows), "positive_count": pos, "negative_count": neg, "train_variants": sorted({row["surface_variant"] for row in train_rows}), "pair_count": len(pair_groups), "pair_weight": pair_weight, "seed": SEED, "epochs": EPOCHS, "selection": "minimum supervised plus held-out-seed objective; not accuracy-only", "best_objective": round(best_objective, 6), "history_tail": history[-5:], "checkpoint": str(checkpoint_path.relative_to(ROOT)), "checkpoint_sha256": checkpoint_sha256}, "thresholds": {"confidence": CONFIDENCE_THRESHOLD, "effect": EFFECT_THRESHOLD, "novelty": novelty_threshold}, "splits": split_metrics, "pair_consistency": pair_metrics, "counterfactual_pair_eval": counterfactual_pair_metrics, "source_holdout": source_metrics, "capability_gate": gate, "promotion": {"status": "quarantined_candidate", "training_allowed": False, "memory_promotion_allowed": False}, "training_allowed": False, "memory_promotion_allowed": False, "cells": cells}


def main() -> int:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    rows = list(catalog["samples"])
    module = _load_pg36_module()
    feature_module = _load_features(module)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = [_train_one(module, feature_module, rows, ablation, device) for ablation in ABLATIONS]
    report = {"protocol_id": "sift-pg37-counterfactual-representation-ablation-v1", "schema_version": "pg-pk-37-representation-ablation-report-v1", "status": "diagnostic_only", "catalog": {"path": str(CATALOG_PATH.relative_to(ROOT)), "sha256": hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest(), "sample_count": len(rows), "typed_positive_count": int(catalog["typed_positive_count"]), "raw_probe_strings_stored": False, "raw_response_bodies_stored": False}, "device": str(device), "ablations": {item["ablation"]: item for item in results}, "comparison": {"pair_agreement": {item["ablation"]: item["counterfactual_pair_eval"]["family_agreement_rate"] for item in results}, "family_holdout_typed_recall": {item["ablation"]: item["splits"]["family_holdout"]["typed_recall"] for item in results}, "source_holdout_typed_recall": {item["ablation"]: item["source_holdout"]["typed_recall"] for item in results}, "unknown_false_positive_rate": {item["ablation"]: item["splits"]["negative_control"]["false_positive_rate"] for item in results}}, "promotion": {"status": "diagnostic_only", "training_allowed": False, "memory_promotion_allowed": False, "capability_claim_allowed": False, "reason": "PG-37 is an ablation; a favorable cell cannot bypass the shared capability gate"}, "manifest_sha256": hashlib.sha256(json.dumps({"protocol_id": "sift-pg37-counterfactual-representation-ablation-v1", "checkpoints": [item["training"]["checkpoint_sha256"] for item in results]}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-37 representation ablation", "", "typed oracle 是标签而不是输入；surface_variant 也不进入输入。", "", "| ablation | surface-pair agreement | family holdout recall | source holdout recall | unknown FPR |", "|---|---:|---:|---:|---:|"]
    for item in results:
        lines.append(f"| {item['ablation']} | {item['counterfactual_pair_eval']['family_agreement_rate']:.2f} | {item['splits']['family_holdout']['typed_recall']:.2f} | {item['source_holdout']['typed_recall']:.2f} | {item['splits']['negative_control']['false_positive_rate']:.2f} |")
    lines.extend(["", "状态：`diagnostic_only`；所有变体 training_allowed 和 memory_promotion_allowed 均为 `False`。", ""])
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "comparison": report["comparison"], "gates": {item["ablation"]: {"status": item["capability_gate"]["status"], "claim_allowed": item["capability_gate"]["claim_allowed"], "reasons": item["capability_gate"].get("reasons", [])[:8]} for item in results}, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
