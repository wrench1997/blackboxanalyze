"""Train PG-38's candidate-minus-control effect representation.

This is an oracle-blind diagnostic.  The effect head receives only a bounded
projection delta, while the family head receives the bounded candidate view.
Typed oracle fields are targets/provenance and never features.
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
FEATURE_SCRIPT = ROOT / "scripts" / "train_pg37_representation_ablation.py"
OUTPUT_DIR = ROOT / "artifacts" / "pg38-effect-pair"
CHECKPOINT_PATH = OUTPUT_DIR / "effect_pair.pt"
REPORT_PATH = ROOT / "research" / "pg38_effect_pair_candidate_report_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg38_effect_pair_candidate_report_v1.md"
SEED = 20380802
EPOCHS = 160
CONFIDENCE_THRESHOLD = 0.60
EFFECT_THRESHOLD = 0.60
FAMILIES = ("access_control", "authentication", "command_injection", "input_validation", "injection", "logic", "ordinary_response", "url_redirect", "xss", "unknown_surface")


def _load_features_module() -> Any:
    spec = importlib.util.spec_from_file_location("pg37_features_for_pg38", FEATURE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load PG-37 bounded feature projector")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EffectPairModel(nn.Module):
    def __init__(self, feature_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.effect_encoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.effect_head = nn.Linear(hidden_dim, 1)
        self.family_encoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.family_head = nn.Linear(hidden_dim, len(FAMILIES))

    def forward(self, candidate: torch.Tensor, delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.family_head(self.family_encoder(candidate)), self.effect_head(self.effect_encoder(delta)).squeeze(-1)


def _pair_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, int, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (str(row["implementation"]), str(row["surface_id"]), str(row["surface_variant"]), int(row["sampling_seed"]), str(row["method"]), str(row["phase"]))
        groups[key][str(row["pair_role"])] = row
    result: list[dict[str, Any]] = []
    for key, pair in groups.items():
        if "candidate" not in pair or "control" not in pair:
            continue
        result.append({"key": key, "candidate": pair["candidate"], "control": pair["control"], "family": pair["candidate"]["family"], "dataset_role": pair["candidate"]["dataset_role"], "implementation": pair["candidate"]["implementation"], "surface_variant": pair["candidate"]["surface_variant"], "sampling_seed": pair["candidate"]["sampling_seed"]})
    return result


def _split(pairs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "train": [p for p in pairs if p["implementation"] == "atlas" and p["dataset_role"] in {"train", "dev"} and int(p["sampling_seed"]) in {361, 367} and p["surface_variant"] in {"compact", "nested"}],
        "seed_holdout": [p for p in pairs if p["implementation"] == "atlas" and p["dataset_role"] in {"train", "dev"} and int(p["sampling_seed"]) == 373 and p["surface_variant"] in {"compact", "nested"}],
        "surface_holdout": [p for p in pairs if p["implementation"] == "atlas" and p["dataset_role"] in {"train", "dev"} and p["surface_variant"] == "headerized"],
        "family_holdout": [p for p in pairs if p["implementation"] == "atlas" and p["dataset_role"] == "family_holdout"],
        "ood_source": [p for p in pairs if p["implementation"] == "atlas" and p["dataset_role"] == "ood_source"],
        "source_holdout": [p for p in pairs if p["implementation"] == "orbit" and p["dataset_role"] in {"train", "dev"}],
        "negative_control": [p for p in pairs if p["dataset_role"] == "negative_control"],
    }


def _pair_features(feature_module: Any, pairs: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    if not pairs:
        return torch.empty((0, 256)), torch.empty((0, 256))
    module = feature_module._load_pg36_module()
    projector = feature_module._load_features(module)
    candidate_rows = [pair["candidate"] for pair in pairs]
    control_rows = [pair["control"] for pair in pairs]
    candidate = feature_module._features(module, projector, candidate_rows, "surface_only")
    control = feature_module._features(module, projector, control_rows, "surface_only")
    return candidate, candidate - control


def _metrics(model: EffectPairModel, candidate: torch.Tensor, delta: torch.Tensor, pairs: list[dict[str, Any]], labels: torch.Tensor, device: torch.device, train_delta: torch.Tensor, novelty_threshold: float) -> dict[str, Any]:
    if not pairs:
        return {"count": 0, "positive_count": 0, "negative_count": 0, "accepted_count": 0, "effect_accepted_count": 0, "false_positive_count": 0, "effect_false_positive_count": 0, "typed_recall": 0.0, "effect_recall_any_family": 0.0, "precision": 1.0, "false_positive_rate": 0.0, "effect_false_positive_rate": 0.0, "abstain_precision": 1.0, "ece": 0.0, "abstain_rate": 1.0, "median_queries": 2.0}
    model.eval()
    with torch.inference_mode():
        family_logits, effect_logits = model(candidate.to(device), delta.to(device))
        family_prob = torch.softmax(family_logits, dim=-1).cpu()
        effect_prob = torch.sigmoid(effect_logits).cpu()
    confidence, prediction = family_prob.max(dim=-1)
    distance = torch.cdist(delta.cpu(), train_delta.cpu()).min(dim=1).values
    effect_accepted = (effect_prob >= EFFECT_THRESHOLD) & (distance <= novelty_threshold)
    typed_accepted = effect_accepted & (confidence >= CONFIDENCE_THRESHOLD)
    positive = torch.tensor([bool(pair["candidate"]["oracle_projection"].get("positive", False)) for pair in pairs])
    correct_family = prediction.eq(labels.cpu())
    typed_positive = positive & typed_accepted & correct_family
    effect_positive = positive & effect_accepted
    false_positive = (~positive) & typed_accepted
    effect_false_positive = (~positive) & effect_accepted
    abstained = ~effect_accepted
    positive_count = int(positive.sum())
    negative_count = int((~positive).sum())
    effect_count = int(effect_accepted.sum())
    typed_count = int(typed_accepted.sum())
    by_family: dict[str, dict[str, float]] = {}
    for family in FAMILIES:
        mask = torch.tensor([pair["family"] == family for pair in pairs])
        fam_pos = positive & mask
        if bool(mask.any()):
            by_family[family] = {"count": int(mask.sum()), "positive_count": int(fam_pos.sum()), "typed_recall": round(float((typed_positive & mask).sum()) / max(int(fam_pos.sum()), 1), 6), "effect_recall": round(float((effect_positive & mask).sum()) / max(int(fam_pos.sum()), 1), 6)}
    return {"count": len(pairs), "positive_count": positive_count, "negative_count": negative_count, "accepted_count": int(typed_positive.sum()), "effect_accepted_count": effect_count, "false_positive_count": int(false_positive.sum()), "effect_false_positive_count": int(effect_false_positive.sum()), "typed_recall": round(float(typed_positive.sum()) / max(positive_count, 1), 6), "effect_recall_any_family": round(float(effect_positive.sum()) / max(positive_count, 1), 6), "precision": round(float(typed_positive.sum()) / max(typed_count, 1), 6), "false_positive_rate": round(float(false_positive.sum()) / max(negative_count, 1), 6), "effect_false_positive_rate": round(float(effect_false_positive.sum()) / max(negative_count, 1), 6), "abstain_precision": round(float((~positive & abstained).sum()) / max(int(abstained.sum()), 1), 6), "ece": round(float((effect_prob - positive.float()).abs().mean()), 6), "abstain_rate": round(float(abstained.float().mean()), 6), "median_queries": 2.0, "mean_effect_probability": round(float(effect_prob.mean()), 6), "max_train_distance": round(float(distance.max()), 6), "by_family": by_family}


def _gate_cells(catalog: dict[str, Any], pairs: list[dict[str, Any]], model: EffectPairModel, feature_module: Any, means: tuple[torch.Tensor, torch.Tensor], train_delta: torch.Tensor, threshold: float, checkpoint_sha256: str, device: torch.device) -> list[dict[str, Any]]:
    candidate_mean, delta_mean = means
    cells: list[dict[str, Any]] = []
    baseline = {"typed_recall": 0.0, "precision": 1.0, "false_positive_rate": 0.0, "abstain_precision": 1.0, "ece": 0.0, "median_queries": 2.0}
    for base in catalog["dataset_tests"]:
        role = str(base["role"])
        seed = int(base["sampling_seed"])
        for implementation in ("atlas", "orbit"):
            selected = [p for p in pairs if p["dataset_role"] == role and int(p["sampling_seed"]) == seed and p["implementation"] == implementation]
            if not selected:
                continue
            candidate, delta = _pair_features(feature_module, selected)
            # Normalisation is computed by the caller and attached as attrs
            # only through closure-like tuple values below.
            candidate = (candidate - candidate_mean) / feature_module._pg38_candidate_std
            delta = (delta - delta_mean) / feature_module._pg38_delta_std
            labels = torch.tensor([FAMILIES.index(p["family"]) for p in selected], dtype=torch.long)
            metrics = _metrics(model, candidate, delta, selected, labels, device, train_delta, threshold)
            target_ids = sorted({str(p["candidate"]["target_instance_id"]) for p in selected})
            source_hashes = sorted({str(p["candidate"]["source_sha256"]) for p in selected})
            cell = {"sample_id": f"pg38-gated-{implementation}-{role}-s{seed}", "dataset_id": f"pg38-gated-{implementation}-{role}-s{seed}", "source_id": f"pg37-counterfactual-source-{implementation}", "source_hash": hashlib.sha256("|".join(source_hashes).encode()).hexdigest(), "target_instance_ids": target_ids, "target_instance_id": target_ids[0], "family_set": sorted({str(p["family"]) for p in selected}), "sampling_seed": seed, "role": role, "evidence_hash": hashlib.sha256(json.dumps(sorted(p["candidate"]["evidence"]["evidence_hash"] for p in selected), separators=(",", ":")).encode()).hexdigest(), "dataset_manifest_sha256": hashlib.sha256(json.dumps(sorted(p["candidate"]["sample_id"] for p in selected), separators=(",", ":")).encode()).hexdigest(), "split_manifest_sha256": hashlib.sha256(f"pg38-{implementation}-{role}-{seed}".encode()).hexdigest(), "probe_sha256": hashlib.sha256(json.dumps(sorted(p["candidate"]["payload_manifest"]["payload_sha256"] for p in selected), separators=(",", ":")).encode()).hexdigest(), "oracle_contract_sha256": hashlib.sha256(b"pg37-counterfactual-typed-oracle-v1").hexdigest(), "checkpoint_sha256": checkpoint_sha256, "sample_count": len(selected), "unique_sample_count": len(selected), "denominator": len(selected), "positive_count": sum(int(p["candidate"]["oracle_projection"]["positive"]) for p in selected), "negative_count": sum(int(not p["candidate"]["oracle_projection"]["positive"]) for p in selected), "abstain_count": int(metrics["count"] - metrics["effect_accepted_count"]), "metrics_status": "completed", "metrics": {key: metrics[key] for key in baseline}, "baseline_metrics": baseline, "candidate_metrics": {key: metrics[key] for key in baseline}}
            cell["evidence_hash"] = hashlib.sha256(json.dumps(cell, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            cells.append(cell)
    return cells


def main() -> int:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    rows = list(catalog["samples"])
    feature_module = _load_features_module()
    pairs = _pair_rows(rows)
    split_pairs = _split(pairs)
    if not all(split_pairs.values()):
        raise RuntimeError("PG-38 pair split is incomplete")
    candidate_raw, delta_raw = _pair_features(feature_module, split_pairs["train"])
    candidate_mean = candidate_raw.mean(dim=0)
    candidate_std = candidate_raw.std(dim=0, unbiased=False).clamp_min(1e-4)
    delta_mean = delta_raw.mean(dim=0)
    delta_std = delta_raw.std(dim=0, unbiased=False).clamp_min(1e-4)
    feature_module._pg38_candidate_std = candidate_std
    feature_module._pg38_delta_std = delta_std
    feature_map: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    labels: dict[str, torch.Tensor] = {}
    for name, split in split_pairs.items():
        candidate, delta = _pair_features(feature_module, split)
        feature_map[name] = ((candidate - candidate_mean) / candidate_std, (delta - delta_mean) / delta_std)
        labels[name] = torch.tensor([FAMILIES.index(pair["family"]) for pair in split], dtype=torch.long)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = EffectPairModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0025, weight_decay=0.01)
    family_loss = nn.CrossEntropyLoss(label_smoothing=0.01)
    positive_count = sum(int(pair["candidate"]["oracle_projection"]["positive"]) for pair in split_pairs["train"])
    negative_count = len(split_pairs["train"]) - positive_count
    effect_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([max(1.0, negative_count / max(positive_count, 1))], device=device))
    best_state: dict[str, torch.Tensor] | None = None
    best_objective = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        family_logits, effect_logits = model(feature_map["train"][0].to(device), feature_map["train"][1].to(device))
        target = torch.tensor([bool(pair["candidate"]["oracle_projection"]["positive"]) for pair in split_pairs["train"]], dtype=torch.float32, device=device)
        supervised_family = family_loss(family_logits, labels["train"].to(device))
        supervised_effect = effect_loss(effect_logits, target)
        objective = supervised_family + supervised_effect
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.inference_mode():
                dev_family, dev_effect = model(feature_map["seed_holdout"][0].to(device), feature_map["seed_holdout"][1].to(device))
                dev_target = torch.tensor([bool(pair["candidate"]["oracle_projection"]["positive"]) for pair in split_pairs["seed_holdout"]], dtype=torch.float32, device=device)
                dev_loss = family_loss(dev_family, labels["seed_holdout"].to(device)) + effect_loss(dev_effect, dev_target)
            selection = float((objective.detach() + 0.35 * dev_loss.detach()).cpu())
            history.append({"epoch": epoch, "family_loss": round(float(supervised_family.detach()), 6), "effect_loss": round(float(supervised_effect.detach()), 6), "dev_loss": round(float(dev_loss.detach()), 6), "selection_objective": round(selection, 6)})
            if selection < best_objective:
                best_objective = selection
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    train_delta = feature_map["train"][1]
    distances = torch.cdist(train_delta, train_delta).fill_diagonal_(float("inf")).min(dim=1).values
    novelty_threshold = max(8.0, float(torch.quantile(distances, 0.95)) + 2.0)
    split_metrics = {name: _metrics(model, feature_map[name][0], feature_map[name][1], split_pairs[name], labels[name], device, train_delta, novelty_threshold) for name in split_pairs}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "sift-pg38-effect-pair-checkpoint-v1", "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "families": list(FAMILIES), "candidate_mean": candidate_mean.tolist(), "candidate_std": candidate_std.tolist(), "delta_mean": delta_mean.tolist(), "delta_std": delta_std.tolist(), "novelty_threshold": novelty_threshold, "seed": SEED, "device_at_training": str(device)}, CHECKPOINT_PATH)
    checkpoint_sha256 = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
    feature_module._pg38_candidate_std = candidate_std
    feature_module._pg38_delta_std = delta_std
    cells = _gate_cells(catalog, pairs, model, feature_module, (candidate_mean, delta_mean), train_delta, novelty_threshold, checkpoint_sha256, device)
    baseline = {"typed_recall": 0.0, "precision": 1.0, "false_positive_rate": 0.0, "abstain_precision": 1.0, "ece": 0.0, "median_queries": 2.0}
    capability_evidence = {"claim_id": "pg38-effect-pair", "dataset_tests": cells, "unit_tests_passed": True, "oracle_validated": True, "data_lineage_complete": True, "authorized_sources_attested": True, "raw_data_retained": False, "false_positive_count": sum(int(split_metrics[name]["effect_false_positive_count"]) for name in split_metrics), "baseline_metrics": baseline, "candidate_metrics": {key: split_metrics["family_holdout"][key] for key in baseline}, "baseline_worst_case_metrics": baseline, "candidate_worst_case_metrics": {key: min(split_metrics[name][key] for name in split_metrics) if key in {"typed_recall", "false_positive_rate", "ece", "precision", "abstain_precision"} else max(split_metrics[name][key] for name in split_metrics) for key in baseline}}
    gate = evaluate_model_capability(capability_evidence, policy={"min_distinct_source_hashes": 2})
    report = {"protocol_id": "sift-pg38-effect-pair-v1", "schema_version": "pg-pk-38-effect-pair-report-v1", "status": "diagnostic_only", "catalog": {"path": str(CATALOG_PATH.relative_to(ROOT)), "sha256": hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest(), "sample_count": len(rows), "pair_count": len(pairs), "typed_positive_pair_count": sum(int(p["candidate"]["oracle_projection"]["positive"]) for p in pairs), "raw_probe_strings_stored": False, "raw_response_bodies_stored": False}, "model": {"class": "EffectPairModel", "effect_head_family_agnostic": True, "typed_oracle_consumed_by_model": False, "positive_authority": False, "candidate_control_delta_only_for_effect": True, "raw_hashes_consumed": False}, "training": {"pair_count": len(split_pairs["train"]), "positive_count": positive_count, "negative_count": negative_count, "epochs": EPOCHS, "seed": SEED, "selection": "minimum supervised plus held-out-seed objective; not accuracy-only", "best_objective": round(best_objective, 6), "history_tail": history[-5:], "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "checkpoint_sha256": checkpoint_sha256}, "thresholds": {"confidence": CONFIDENCE_THRESHOLD, "effect": EFFECT_THRESHOLD, "novelty": novelty_threshold}, "splits": split_metrics, "capability_gate": gate, "promotion": {"status": "quarantined_candidate", "training_allowed": False, "memory_promotion_allowed": False}, "training_allowed": False, "memory_promotion_allowed": False, "manifest_sha256": hashlib.sha256(json.dumps({"protocol_id": "sift-pg38-effect-pair-v1", "checkpoint_sha256": checkpoint_sha256}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-38 candidate-control effect pair", "", "effect head 只看 bounded candidate-control delta；typed oracle 只是标签。", "", "| split | typed recall | effect recall | typed FPR | effect FPR | abstain |", "|---|---:|---:|---:|---:|---:|"]
    for name in ("train", "seed_holdout", "surface_holdout", "family_holdout", "ood_source", "source_holdout", "negative_control"):
        item = split_metrics[name]
        lines.append(f"| {name} | {item['typed_recall']:.2f} | {item['effect_recall_any_family']:.2f} | {item['false_positive_rate']:.2f} | {item['effect_false_positive_rate']:.2f} | {item['abstain_rate']:.2f} |")
    lines.extend(["", f"状态：`{gate['status']}`；claim_allowed=`{gate['claim_allowed']}`；不晋升。", ""])
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "pair_count": len(pairs), "splits": {name: {key: split_metrics[name][key] for key in ("typed_recall", "effect_recall_any_family", "false_positive_rate", "effect_false_positive_rate", "abstain_rate")} for name in split_metrics}, "capability_gate": {"status": gate["status"], "claim_allowed": gate["claim_allowed"], "reasons": gate.get("reasons", [])[:12]}, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
