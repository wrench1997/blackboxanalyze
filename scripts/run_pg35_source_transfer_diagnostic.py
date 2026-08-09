"""PG-35 source-transfer ablation.

This deliberately differs from the strict family-holdout run: all nine safe
families from PG-35 alpha are used for training, while beta and gamma remain
implementation-blind.  It answers whether the remaining failure is family
coverage or route/response representation.  It is diagnostic only and cannot
promote a model or memory.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAINER_PATH = ROOT / "scripts" / "train_pg35_pair_rule_ir.py"
CATALOG_PATH = ROOT / "research" / "pg35_independent_fixture_catalog_v1.json"
OUTPUT_DIR = ROOT / "artifacts" / "pg35-source-transfer"
CHECKPOINT_PATH = OUTPUT_DIR / "pair_rule_ir.pt"
REPORT_PATH = ROOT / "research" / "pg35_source_transfer_diagnostic_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg35_source_transfer_diagnostic_v1.md"
SEED = 20350803
EPOCHS = 180


def _load_trainer() -> Any:
    spec = importlib.util.spec_from_file_location("pg35_pair_trainer_source_transfer", TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load PG-35 pair trainer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _train(module: Any, feature_module: Any, rows: list[dict[str, Any]], device: torch.device) -> tuple[Any, torch.Tensor, torch.Tensor, list[dict[str, float]], int, float]:
    raw = module._visible_features(feature_module, rows)
    mean = raw.mean(dim=0)
    std = raw.std(dim=0, unbiased=False).clamp_min(1e-4)
    features = (raw - mean) / std
    labels = torch.tensor([module.FAMILIES.index(row["family"]) for row in rows], dtype=torch.long)
    model = module.PairRuleIRModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0025, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.01)
    pair_groups = module._pair_groups(rows, features)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    best_state: dict[str, torch.Tensor] | None = None
    best_objective = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        family_logits, effect_logits = model(features.to(device))
        family_loss = criterion(family_logits, labels.to(device))
        effect_labels = torch.tensor([bool(row["oracle_projection"].get("positive", False)) for row in rows], dtype=torch.float32, device=device)
        effect_loss = nn.functional.binary_cross_entropy_with_logits(effect_logits, effect_labels)
        pair_loss = torch.tensor(0.0, device=device)
        if pair_groups:
            terms = []
            for left, right in pair_groups:
                pair_family_logits, pair_effect_logits = model(torch.stack((left, right)).to(device))
                terms.append(torch.mean((torch.softmax(pair_family_logits, dim=-1)[0] - torch.softmax(pair_family_logits, dim=-1)[1]) ** 2) + torch.mean((torch.sigmoid(pair_effect_logits)[0] - torch.sigmoid(pair_effect_logits)[1]) ** 2))
            pair_loss = torch.stack(terms).mean()
        objective = family_loss + effect_loss + module.PAIR_WEIGHT * pair_loss
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch % 20 == 0 or epoch == 1:
            value = float(objective.detach())
            history.append({"epoch": epoch, "family_loss": round(float(family_loss.detach()), 6), "effect_loss": round(float(effect_loss.detach()), 6), "pair_loss": round(float(pair_loss.detach()), 6), "objective": round(value, 6)})
            if value < best_objective:
                best_objective = value
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean, std, history, len(pair_groups), best_objective


def main() -> int:
    module = _load_trainer()
    feature_module = module._load_features()
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    rows = list(catalog["samples"])
    train_rows = [row for row in rows if row.get("variant") == "alpha"]
    beta_rows = [row for row in rows if row.get("variant") == "beta"]
    gamma_rows = [row for row in rows if row.get("variant") == "gamma"]
    negative_rows = [row for row in rows if row.get("variant") in {"beta", "gamma"} and row.get("family") == "ordinary_response"]
    split_rows = {"train_alpha": train_rows, "beta_source_holdout": beta_rows, "gamma_source_holdout": gamma_rows, "negative_control": negative_rows}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, history, pair_count, best_objective = _train(module, feature_module, train_rows, device)
    train_raw = module._visible_features(feature_module, train_rows)
    train_features = (train_raw - mean) / std
    distances = torch.cdist(train_features, train_features).fill_diagonal_(float("inf")).min(dim=1).values
    novelty_threshold = max(8.0, float(torch.quantile(distances, 0.95)) + 2.0)
    metrics: dict[str, dict[str, Any]] = {}
    pairs: dict[str, dict[str, Any]] = {}
    for name, split in split_rows.items():
        raw = module._visible_features(feature_module, split)
        features = (raw - mean) / std
        labels = torch.tensor([module.FAMILIES.index(row["family"]) for row in split], dtype=torch.long)
        metrics[name] = module._metrics(model, features, split, labels, device, novelty_threshold=novelty_threshold, train_features=train_features)
        pairs[name] = module._pair_consistency(model, split, features, device)
        metrics[name]["pair_consistency"] = pairs[name]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": "sift-pg35-source-transfer-checkpoint-v1",
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "feature_dim": module.FEATURE_DIM,
        "families": list(module.FAMILIES),
        "normalisation_mean": mean.tolist(),
        "normalisation_std": std.tolist(),
        "confidence_threshold": module.CONFIDENCE_THRESHOLD,
        "novelty_threshold": novelty_threshold,
        "pair_weight": module.PAIR_WEIGHT,
        "seed": SEED,
        "device_at_training": str(device),
    }, CHECKPOINT_PATH)
    report = {
        "protocol_id": "sift-pg35-source-transfer-diagnostic-v1",
        "schema_version": "pg-pk-35-source-transfer-diagnostic-v1",
        "status": "diagnostic_only",
        "training": {
            "source_variant": "alpha",
            "sample_count": len(train_rows),
            "family_set": sorted({row["family"] for row in train_rows}),
            "pair_count": pair_count,
            "seed": SEED,
            "epochs": EPOCHS,
            "best_objective": round(best_objective, 6),
            "history_tail": history[-5:],
        },
        "model": {
            "class": "PairRuleIRModel",
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "visible_projection_labels": False,
            "typed_oracle_consumed_by_model": False,
            "positive_authority": False,
        },
        "thresholds": {"confidence": module.CONFIDENCE_THRESHOLD, "novelty": novelty_threshold},
        "splits": metrics,
        "pair_consistency": pairs,
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest(),
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "capability_claim_allowed": False,
        "interpretation": "source_transfer_only; family coverage is not a family-holdout claim",
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-35 source-transfer diagnostic", "", "训练使用 alpha 的全部安全 family，beta/gamma 只作源外盲测。", "", "| split | recall | precision | FPR | abstain | pair agreement |", "|---|---:|---:|---:|---:|---:|"]
    for name, row in metrics.items():
        lines.append(f"| {name} | {row['typed_recall']:.2f} | {row['precision']:.2f} | {row['false_positive_rate']:.2f} | {row['abstain_rate']:.2f} | {row['pair_consistency']['agreement_rate']:.2f} |")
    lines.extend(["", "状态：`diagnostic_only`；不允许训练晋升或长期记忆。", ""])
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "train_count": len(train_rows), "splits": {name: {key: row[key] for key in ("typed_recall", "precision", "false_positive_rate", "abstain_rate")} for name, row in metrics.items()}, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
