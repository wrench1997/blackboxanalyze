"""Triage PG-42's held-out failure without retraining or using oracle labels as features."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CATALOG_PATH = ROOT / "research" / "pg42_independent_semantic_catalog_v1.json"
TRAIN_CATALOG_PATH = ROOT / "research" / "pg37_counterfactual_catalog_v1.json"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg39-coarse-delta" / "coarse_delta.pt"
PG38_SCRIPT = ROOT / "scripts" / "train_pg38_effect_pair_candidate.py"
PG39_SCRIPT = ROOT / "scripts" / "train_pg39_coarse_delta_candidate.py"
OUTPUT_PATH = ROOT / "research" / "pg42_independent_ood_triage_v1.json"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path.name}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def _normalised(pg39: Any, pairs: list[dict[str, Any]], checkpoint: dict[str, Any]) -> torch.Tensor:
    raw = torch.stack([pg39._coarse_pair(pair) for pair in pairs])
    mean = torch.tensor(checkpoint["delta_mean"], dtype=torch.float32)
    std = torch.tensor(checkpoint["delta_std"], dtype=torch.float32).clamp_min(1e-4)
    return (raw - mean) / std


def _summary(name: str, pairs: list[dict[str, Any]], delta: torch.Tensor, train_delta: torch.Tensor, model: Any, threshold: float) -> dict[str, Any]:
    positive = torch.tensor([bool(pair["candidate"]["oracle_projection"].get("positive", False)) for pair in pairs])
    with torch.inference_mode():
        _, logits = model(torch.zeros((len(pairs), 256)), delta)
        probability = torch.sigmoid(logits).cpu()
    distance = torch.cdist(delta.cpu(), train_delta.cpu()).min(dim=1).values
    accepted = (probability >= 0.60) & (distance <= threshold)
    raw_nonzero = (delta != 0).sum(dim=1)
    zero_std_shift = ((delta.abs() > 100.0).sum(dim=1) > 0)
    positive_delta = delta[positive]
    feature_abs = positive_delta.abs().mean(dim=0) if len(positive_delta) else torch.zeros(32)
    top = torch.argsort(feature_abs, descending=True)[:6].tolist()
    variants: dict[str, dict[str, Any]] = {}
    for variant in sorted({str(pair["candidate"].get("surface_variant")) for pair in pairs}):
        subset = [index for index, pair in enumerate(pairs) if str(pair["candidate"].get("surface_variant")) == variant]
        mask = torch.tensor(subset, dtype=torch.long)
        variants[variant] = {"pair_count": len(subset), "positive_count": int(positive[mask].sum()), "effect_recall": round(float((positive[mask] & accepted[mask]).sum()) / max(int(positive[mask].sum()), 1), 6), "novelty_reject_count": int((distance[mask] > threshold).sum()), "zero_variance_shift_count": int(zero_std_shift[mask].sum())}
    quantiles = torch.quantile(distance, torch.tensor([0.5, 0.9, 0.99])).tolist() if len(distance) else [0.0, 0.0, 0.0]
    return {"pair_count": len(pairs), "positive_count": int(positive.sum()), "effect_recall": round(float((positive & accepted).sum()) / max(int(positive.sum()), 1), 6), "effect_false_positive_rate": round(float((~positive & accepted).sum()) / max(int((~positive).sum()), 1), 6), "novelty_reject_count": int((distance > threshold).sum()), "zero_variance_shift_count": int(zero_std_shift.sum()), "raw_nonzero_feature_mean": round(float(raw_nonzero.float().mean()), 6), "positive_probability_mean": round(float(probability[positive].mean()), 6) if positive.any() else 0.0, "positive_probability_median": round(float(probability[positive].median()), 6) if positive.any() else 0.0, "distance_quantiles": [round(float(item), 6) for item in quantiles], "dominant_positive_normalized_feature_indices": top, "surface_variant_metrics": variants}


def main() -> int:
    pg38 = _load(PG38_SCRIPT, "pg38_for_pg42_triage")
    pg39 = _load(PG39_SCRIPT, "pg39_for_pg42_triage")
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    pairs = pg38._pair_rows(list(catalog["samples"]))
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = pg39.CoarsePairModel(); model.load_state_dict(checkpoint["model_state"]); model.eval()
    train_catalog = json.loads(TRAIN_CATALOG_PATH.read_text(encoding="utf-8"))
    train_pairs = pg38._split(pg38._pair_rows(list(train_catalog["samples"]))) ["train"]
    train_delta = _normalised(pg39, train_pairs, checkpoint)
    groups = {
        "pg39_train": train_pairs,
        "pg42_train": [pair for pair in pairs if pair["candidate"].get("dataset_role") == "train"],
        "pg42_dev": [pair for pair in pairs if pair["candidate"].get("dataset_role") == "dev"],
        "pg42_implementation_holdout": [pair for pair in pairs if pair["candidate"].get("dataset_role") == "ood_source"],
        "pg42_family_holdout": [pair for pair in pairs if pair["candidate"].get("dataset_role") == "family_holdout"],
    }
    report = {"schema_version": "pg-pk-42-independent-ood-triage-v1", "protocol_id": "PG-42", "status": "diagnostic_only", "checkpoint": {"sha256": hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest(), "zero_variance_feature_indices": [index for index, value in enumerate(checkpoint["delta_std"]) if float(value) <= 1e-4], "novelty_threshold": float(checkpoint["novelty_threshold"])}, "groups": {}, "root_cause_hypotheses": ["PG-39 normalization has zero-variance dimensions that become very large under PG-42 response envelope/layout shifts.", "The framed response variant is outside the PG-39 coarse-delta manifold and is rejected by the novelty gate even when effect logits are positive.", "The independent quartz layout changes status/body/header coarse features, reducing effect confidence without increasing negative false accepts."], "action": "Train a quarantined invariant-delta candidate on PG-37 only, with robust clipping/scale handling; rerun unchanged PG-42 holdout before any promotion.", "training_allowed": False, "memory_promotion_allowed": False}
    for name, group in groups.items():
        report["groups"][name] = _summary(name, group, _normalised(pg39, group, checkpoint), train_delta, model, float(checkpoint["novelty_threshold"]))
    report["manifest_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "zero_variance_feature_indices": report["checkpoint"]["zero_variance_feature_indices"], "groups": report["groups"], "action": report["action"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
