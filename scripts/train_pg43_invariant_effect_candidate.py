"""Train a quarantined invariant effect candidate on PG-37 and test PG-42.

Only PG-37 is used for fitting.  The representation keeps bounded shape
delta/presence bits and discards response-envelope, header, body-length,
status-code, phase, and method nuisance fields.  PG-42 remains an untouched
independent implementation holdout.  This is a candidate experiment, not a
promotion path.
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

PG37_CATALOG_PATH = ROOT / "research" / "pg37_counterfactual_catalog_v1.json"
PG42_CATALOG_PATH = ROOT / "research" / "pg42_independent_semantic_catalog_v1.json"
PG38_SCRIPT = ROOT / "scripts" / "train_pg38_effect_pair_candidate.py"
PG39_SCRIPT = ROOT / "scripts" / "train_pg39_coarse_delta_candidate.py"
OUTPUT_DIR = ROOT / "artifacts" / "pg43-invariant-effect"
CHECKPOINT_PATH = OUTPUT_DIR / "invariant_effect.pt"
REPORT_PATH = ROOT / "research" / "pg43_invariant_effect_candidate_report_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg43_invariant_effect_candidate_report_v1.md"
SEED = 20430802
EPOCHS = 320
THRESHOLD = 0.60
INVARIANT_INDICES = tuple(range(12)) + (18, 19)


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {path.name}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


class InvariantEffectModel(nn.Module):
    def __init__(self, feature_dim: int = len(INVARIANT_INDICES), hidden_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(features)).squeeze(-1)


def _raw(pg39: Any, pairs: list[dict[str, Any]]) -> torch.Tensor:
    if not pairs:
        return torch.empty((0, len(INVARIANT_INDICES)), dtype=torch.float32)
    values = torch.stack([pg39._coarse_pair(pair) for pair in pairs])
    return values[:, INVARIANT_INDICES]


def _features(pg39: Any, pairs: list[dict[str, Any]]) -> torch.Tensor:
    # Sign-binning is deliberate: it preserves whether an observable shape
    # changed, without memorizing the target's envelope-specific magnitude.
    return torch.sign(_raw(pg39, pairs))


def _labels(pairs: list[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor([bool(pair["candidate"]["oracle_projection"].get("positive", False)) for pair in pairs], dtype=torch.float32)


def _distance_threshold(features: torch.Tensor) -> float:
    if len(features) <= 1:
        return 2.0
    distances = torch.cdist(features, features).fill_diagonal_(float("inf")).min(dim=1).values
    return max(2.0, float(torch.quantile(distances, 0.95)) + 0.5)


def _metrics(model: InvariantEffectModel, features: torch.Tensor, pairs: list[dict[str, Any]], train_features: torch.Tensor, novelty_threshold: float, device: torch.device) -> dict[str, Any]:
    if not pairs:
        return {"pair_count": 0, "positive_count": 0, "negative_count": 0, "effect_accepted_count": 0, "effect_recall_any_family": 0.0, "effect_false_positive_count": 0, "effect_false_positive_rate": 0.0, "abstain_rate": 1.0, "mean_effect_probability": 0.0, "mean_train_distance": 0.0}
    model.eval()
    with torch.inference_mode():
        probability = torch.sigmoid(model(features.to(device))).cpu()
    distance = torch.cdist(features.cpu(), train_features.cpu()).min(dim=1).values
    accepted = (probability >= THRESHOLD) & (distance <= novelty_threshold)
    positive = _labels(pairs).bool()
    negative = ~positive
    return {"pair_count": len(pairs), "positive_count": int(positive.sum()), "negative_count": int(negative.sum()), "effect_accepted_count": int(accepted.sum()), "effect_recall_any_family": round(float((positive & accepted).sum()) / max(int(positive.sum()), 1), 6), "effect_false_positive_count": int((negative & accepted).sum()), "effect_false_positive_rate": round(float((negative & accepted).sum()) / max(int(negative.sum()), 1), 6), "abstain_rate": round(float((~accepted).float().mean()), 6), "mean_effect_probability": round(float(probability.mean()), 6), "mean_train_distance": round(float(distance.mean()), 6), "max_train_distance": round(float(distance.max()), 6)}


def _split_pg42(pairs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "train_role_diagnostic": [pair for pair in pairs if pair["candidate"].get("dataset_role") == "train"],
        "dev_role": [pair for pair in pairs if pair["candidate"].get("dataset_role") == "dev"],
        "implementation_holdout": [pair for pair in pairs if pair["candidate"].get("dataset_role") == "ood_source"],
        "family_holdout": [pair for pair in pairs if pair["candidate"].get("dataset_role") == "family_holdout"],
        "negative_control": [pair for pair in pairs if pair["candidate"].get("dataset_role") == "negative_control"],
    }


def main() -> int:
    pg38 = _load(PG38_SCRIPT, "pg38_for_pg43")
    pg39 = _load(PG39_SCRIPT, "pg39_for_pg43")
    pg37 = json.loads(PG37_CATALOG_PATH.read_text(encoding="utf-8"))
    pg42 = json.loads(PG42_CATALOG_PATH.read_text(encoding="utf-8"))
    pg37_pairs = pg38._pair_rows(list(pg37["samples"]))
    pg42_pairs = pg38._pair_rows(list(pg42["samples"]))
    splits = pg38._split(pg37_pairs)
    train_pairs = splits["train"]
    seed_holdout_pairs = splits["seed_holdout"]
    train_features = _features(pg39, train_pairs)
    seed_features = _features(pg39, seed_holdout_pairs)
    train_labels = _labels(train_pairs)
    seed_labels = _labels(seed_holdout_pairs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    model = InvariantEffectModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    pos_weight = torch.tensor([max(1.0, float((len(train_labels) - train_labels.sum()) / max(train_labels.sum(), 1.0)))], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_state: dict[str, torch.Tensor] | None = None
    best_selection = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        train_logits = model(train_features.to(device)); train_loss = loss_fn(train_logits, train_labels.to(device))
        train_loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if epoch % 40 == 0 or epoch == 1:
            model.eval()
            with torch.inference_mode():
                seed_loss = loss_fn(model(seed_features.to(device)), seed_labels.to(device))
            selection = float((train_loss.detach() + 0.5 * seed_loss.detach()).cpu())
            history.append({"epoch": epoch, "train_loss": round(float(train_loss.detach()), 6), "seed_loss": round(float(seed_loss.detach()), 6), "selection": round(selection, 6)})
            if selection < best_selection:
                best_selection = selection; best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    novelty_threshold = _distance_threshold(train_features)
    pg37_metrics = {name: _metrics(model, _features(pg39, split), split, train_features, novelty_threshold, device) for name, split in {"train": train_pairs, "seed_holdout": seed_holdout_pairs, "negative_control": splits["negative_control"]}.items()}
    pg42_metrics = {name: _metrics(model, _features(pg39, split), split, train_features, novelty_threshold, device) for name, split in _split_pg42(pg42_pairs).items()}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "sift-pg43-invariant-effect-checkpoint-v1", "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "invariant_indices": list(INVARIANT_INDICES), "representation": "sign_binned_shape_delta_and_change_bits", "novelty_threshold": novelty_threshold, "seed": SEED, "device_at_training": str(device)}, CHECKPOINT_PATH)
    checkpoint_sha256 = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
    candidate_effect_gate = {"status": "passed" if pg42_metrics["implementation_holdout"]["effect_recall_any_family"] == 1.0 and pg42_metrics["family_holdout"]["effect_recall_any_family"] == 1.0 and pg42_metrics["negative_control"]["effect_false_positive_rate"] == 0.0 else "blocked", "claim_allowed": False, "reasons": [] if pg42_metrics["implementation_holdout"]["effect_recall_any_family"] == 1.0 and pg42_metrics["family_holdout"]["effect_recall_any_family"] == 1.0 and pg42_metrics["negative_control"]["effect_false_positive_rate"] == 0.0 else ["independent_effect_gate_not_met"], "training_allowed": False, "memory_promotion_allowed": False}
    report = {"protocol_id": "sift-pg43-invariant-effect-v1", "schema_version": "pg-pk-43-invariant-effect-report-v1", "status": "diagnostic_only", "training_source": {"catalog": str(PG37_CATALOG_PATH.relative_to(ROOT)), "pair_count": len(train_pairs), "typed_oracle_consumed_by_model": False, "pg42_used_for_training": False}, "model": {"class": "InvariantEffectModel", "representation": "sign_binned_shape_delta_and_change_bits", "invariant_indices": list(INVARIANT_INDICES), "nuisance_dimensions_excluded": [13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25], "typed_oracle_consumed_by_model": False, "family_agnostic": True, "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "checkpoint_sha256": checkpoint_sha256}, "training": {"epochs": EPOCHS, "seed": SEED, "device": str(device), "best_selection": round(best_selection, 6), "history_tail": history[-5:], "novelty_threshold": novelty_threshold}, "pg37_splits": pg37_metrics, "pg42_splits": pg42_metrics, "candidate_effect_gate": candidate_effect_gate, "promotion": {"status": "quarantined_candidate", "training_allowed": False, "memory_promotion_allowed": False}, "manifest_sha256": hashlib.sha256(json.dumps({"protocol_id": "sift-pg43-invariant-effect-v1", "checkpoint_sha256": checkpoint_sha256, "pg37_catalog_sha256": hashlib.sha256(PG37_CATALOG_PATH.read_bytes()).hexdigest(), "pg42_catalog_sha256": hashlib.sha256(PG42_CATALOG_PATH.read_bytes()).hexdigest()}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-43 invariant effect candidate", "", "只用 PG-37 训练；模型输入是 sign-binned shape delta 与 change bits，去除 envelope/body/header/status/method/phase 维度。", "", "| split | effect recall | effect FPR | abstain |", "|---|---:|---:|---:|"]
    for name, item in {**{f"pg37_{key}": value for key, value in pg37_metrics.items()}, **{f"pg42_{key}": value for key, value in pg42_metrics.items()}}.items():
        lines.append(f"| {name} | {item['effect_recall_any_family']:.2f} | {item['effect_false_positive_rate']:.2f} | {item['abstain_rate']:.2f} |")
    lines.extend(["", f"候选 effect 门禁：`{candidate_effect_gate['status']}`；完整能力 claim 仍关闭。", ""])
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "pg37_splits": pg37_metrics, "pg42_splits": pg42_metrics, "candidate_effect_gate": candidate_effect_gate, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
