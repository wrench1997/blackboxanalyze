"""Train a diagnostic family-specific surface discriminator on PG-34.

The head predicts which family-specific oracle should be scheduled.  Its input
is the same bounded visible projection used by the Rule IR candidate; family
labels are training targets only and never become positive vulnerability
authority.  Unseen families and the ordinary negative surface are evaluated
with a confidence/novelty abstain gate.
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

CATALOG_PATH = ROOT / "research" / "pg34_independent_fixture_catalog_v1.json"
TRAINER_PATH = ROOT / "scripts" / "train_pg33_formal_rule_ir_candidate.py"
ARTIFACT_DIR = ROOT / "artifacts" / "pg34-surface-discriminator"
CHECKPOINT_PATH = ARTIFACT_DIR / "surface_discriminator.pt"
REPORT_PATH = ROOT / "research" / "pg34_surface_discriminator_report_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg34_surface_discriminator_report_v1.md"
SEED = 20340802
EPOCHS = 220
FAMILIES = (
    "xss",
    "injection",
    "authentication",
    "access_control",
    "logic",
    "url_redirect",
    "input_validation",
    "command_injection",
    "ordinary_response",
)


class SurfaceDiscriminator(nn.Module):
    def __init__(self, feature_dim: int = 256, hidden_dim: int = 128, classes: int = len(FAMILIES)):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(hidden_dim, classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(features))


def _load_feature_module() -> Any:
    spec = importlib.util.spec_from_file_location("pg33_feature_projection_for_surface", TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load shared visible projection")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics(model: SurfaceDiscriminator, features: torch.Tensor, rows: list[dict[str, Any]], labels: torch.Tensor, device: torch.device, *, threshold: float, novelty_threshold: float, train_features: torch.Tensor) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(features.to(device)), dim=-1).cpu()
    confidence, indices = probabilities.max(dim=-1)
    distances = torch.cdist(features.cpu(), train_features.cpu()).min(dim=1).values if len(train_features) else torch.full((len(rows),), float("inf"))
    accepted = (confidence >= float(threshold)) & (distances <= float(novelty_threshold))
    correct = indices.eq(labels.cpu())
    route_correct = correct & accepted
    false_route = accepted & ~correct
    by_family: dict[str, dict[str, float]] = {}
    for index, family in enumerate(FAMILIES):
        mask = labels.cpu() == index
        by_family[family] = {
            "count": int(mask.sum()),
            "accepted": int((accepted & mask).sum()),
            "route_accuracy": round(float(route_correct[mask].float().mean()), 6) if bool(mask.any()) else 0.0,
            "abstain_rate": round(float((~accepted[mask]).float().mean()), 6) if bool(mask.any()) else 0.0,
        }
    return {
        "count": len(rows),
        "route_accuracy": round(float(route_correct.float().mean()), 6) if rows else 0.0,
        "accepted_accuracy": round(float(correct[accepted].float().mean()), 6) if bool(accepted.any()) else None,
        "false_route_rate": round(float(false_route.float().mean()), 6) if rows else 0.0,
        "abstain_rate": round(float((~accepted).float().mean()), 6) if rows else 0.0,
        "mean_confidence": round(float(confidence.mean()), 6) if rows else 0.0,
        "max_distance": round(float(distances.max()), 6) if rows else 0.0,
        "by_family": by_family,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PG-34 surface discriminator",
        "",
        "该 head 只做 family-specific oracle 路由，不拥有 positive authority。输入为脱敏可见投影；未见族和普通响应必须通过 confidence/novelty abstain。",
        "",
        "| split | route accuracy | accepted accuracy | false route | abstain |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("train", "dev", "family_holdout", "ood_source", "negative_control"):
        row = report["splits"][name]
        lines.append(f"| {name} | {row['route_accuracy']:.2f} | {row['accepted_accuracy'] if row['accepted_accuracy'] is not None else '—'} | {row['false_route_rate']:.2f} | {row['abstain_rate']:.2f} |")
    lines.extend([
        "",
        f"状态：`{report['status']}`；positive authority：`{report['positive_authority']}`；长期记忆：`{report['memory_promotion_allowed']}`。",
        "",
        "族外路由失败只会触发 abstain 和指定 oracle 探测，不会直接生成漏洞结论。",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    feature_module = _load_feature_module()
    rows = list(catalog["samples"])
    split_rows = {
        "train": [row for row in rows if row["dataset_role"] == "train"],
        "dev": [row for row in rows if row["dataset_role"] == "dev"],
        "family_holdout": [row for row in rows if row["dataset_role"] == "family_holdout"],
        "ood_source": [row for row in rows if row["dataset_role"] == "ood_source"],
        "negative_control": [row for row in rows if row["dataset_role"] == "negative_control"],
    }
    train_rows = split_rows["train"]
    raw_train = feature_module._features(train_rows)
    raw_all = {name: feature_module._features(value) for name, value in split_rows.items()}
    mean = raw_train.mean(dim=0)
    std = raw_train.std(dim=0, unbiased=False).clamp_min(1e-4)
    train_features = (raw_train - mean) / std
    features = {name: (value - mean) / std for name, value in raw_all.items()}
    label_index = {family: index for index, family in enumerate(FAMILIES)}
    labels = {name: torch.tensor([label_index[row["family"]] for row in value], dtype=torch.long) for name, value in split_rows.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = SurfaceDiscriminator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.01)
    # Route accuracy alone is not a useful checkpoint criterion here: with a
    # two-family training split, a nearly-uniform random head can already have
    # the right argmax on every row.  Select by the supervised loss (and keep
    # the route metric only as a diagnostic), otherwise we may persist the
    # initial random state while reporting a perfect fit accuracy.
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(train_features.to(device)), labels["train"].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch % 20 == 0 or epoch == 1:
            fit = _metrics(model, features["train"], split_rows["train"], labels["train"], device, threshold=0.0, novelty_threshold=float("inf"), train_features=train_features)
            loss_value = float(loss.detach())
            history.append({"epoch": epoch, "loss": round(loss_value, 6), "fit_route_accuracy": fit["route_accuracy"], "fit_mean_confidence": fit["mean_confidence"]})
            if loss_value < best_loss:
                best_loss = loss_value
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    # Strict confidence/novelty thresholds are diagnostic and intentionally do
    # not grant positive authority.  Novelty uses the train visible features.
    train_distances = torch.cdist(train_features, train_features).fill_diagonal_(float("inf")).min(dim=1).values if len(train_features) > 1 else torch.tensor([0.0])
    novelty_threshold = max(8.0, float(torch.quantile(train_distances, 0.95)) + 2.0)
    threshold = 0.60
    split_metrics = {name: _metrics(model, features[name], split_rows[name], labels[name], device, threshold=threshold, novelty_threshold=novelty_threshold, train_features=train_features) for name in split_rows}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_sha256: str
    torch.save({
        "schema_version": "sift-pg34-surface-discriminator-checkpoint-v1",
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "feature_dim": 256,
        "families": list(FAMILIES),
        "normalisation_mean": mean.tolist(),
        "normalisation_std": std.tolist(),
        "abstain_threshold": threshold,
        "novelty_threshold": novelty_threshold,
        "seed": SEED,
        "device_at_training": str(device),
    }, CHECKPOINT_PATH)
    checkpoint_sha256 = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
    report = {
        "protocol_id": "sift-pg34-surface-discriminator-v1",
        "schema_version": "pg-pk-34-surface-discriminator-report-v1",
        "status": "diagnostic_only",
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "training_sample_count": len(train_rows),
        "training_families": sorted({row["family"] for row in train_rows}),
        "heldout_families": sorted({row["family"] for row in split_rows["family_holdout"]}),
        "ood_families": sorted({row["family"] for row in split_rows["ood_source"]}),
        "visible_projection_labels": False,
        "typed_oracle_consumed_by_model": False,
        "thresholds": {"confidence": threshold, "novelty": novelty_threshold},
        "checkpoint_selection": "minimum_train_cross_entropy",
        "best_train_loss": round(best_loss, 6),
        "splits": split_metrics,
        "history_tail": history[-5:],
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_sha256,
        "positive_authority": False,
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "promotion_reason": "surface_route_head_requires_typed_oracle_and_capability_gate",
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "training_sample_count": len(train_rows), "splits": split_metrics, "checkpoint": report["checkpoint"], "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
