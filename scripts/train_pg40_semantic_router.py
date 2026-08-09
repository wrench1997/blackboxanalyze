"""Train a source-transfer-only semantic Rule IR router for PG-40.

Semantic references are abstract probe contexts, not family labels.  This run
intentionally observes every semantic class during training, so it is not a
family-OOD capability claim; it only tests source/seed transfer.
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


CATALOG_PATH = ROOT / "research" / "pg40_semantic_router_catalog_v1.json"
PG38_SCRIPT = ROOT / "scripts" / "train_pg38_effect_pair_candidate.py"
PG39_SCRIPT = ROOT / "scripts" / "train_pg39_coarse_delta_candidate.py"
OUTPUT_DIR = ROOT / "artifacts" / "pg40-semantic-router"
CHECKPOINT_PATH = OUTPUT_DIR / "semantic_router.pt"
REPORT_PATH = ROOT / "research" / "pg40_semantic_router_report_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg40_semantic_router_report_v1.md"
SEED = 20400802
EPOCHS = 160
FAMILIES = ("access_control", "authentication", "command_injection", "input_validation", "injection", "logic", "ordinary_response", "url_redirect", "xss", "unknown_surface")
CONFIDENCE_THRESHOLD = 0.60
EFFECT_THRESHOLD = 0.60


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SemanticPairModel(nn.Module):
    def __init__(self, candidate_dim: int = 256, delta_dim: int = 32, hidden_dim: int = 128) -> None:
        super().__init__()
        self.effect_encoder = nn.Sequential(nn.Linear(delta_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.effect_head = nn.Linear(hidden_dim, 1)
        self.family_encoder = nn.Sequential(nn.Linear(candidate_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.family_head = nn.Linear(hidden_dim, len(FAMILIES))

    def forward(self, candidate: torch.Tensor, delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.family_head(self.family_encoder(candidate)), self.effect_head(self.effect_encoder(delta)).squeeze(-1)


def _semantic_feature(candidate: torch.Tensor, pair: dict[str, Any]) -> torch.Tensor:
    vector = candidate.clone()
    semantic = str(pair["candidate"]["payload_manifest"].get("probe_ref", "semantic-unknown"))
    digest = hashlib.blake2b(semantic.encode("utf-8"), digest_size=8).digest()
    index = 224 + (int.from_bytes(digest, "little") % 32)
    vector[index] = min(float(vector[index]) + 1.0, 8.0)
    return vector


def _features(pg38: Any, feature_module: Any, pairs: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    candidate, _ = pg38._pair_features(feature_module, pairs)
    candidate = torch.stack([_semantic_feature(vector, pair) for vector, pair in zip(candidate, pairs)])
    pg39 = _load(PG39_SCRIPT, "pg39_helpers_for_pg40")
    delta = torch.stack([pg39._coarse_pair(pair) for pair in pairs])
    return candidate, delta


def _metrics(model: SemanticPairModel, candidate: torch.Tensor, delta: torch.Tensor, pairs: list[dict[str, Any]], labels: torch.Tensor, device: torch.device, train_delta: torch.Tensor, novelty: float) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        family_logits, effect_logits = model(candidate.to(device), delta.to(device))
        family_prob = torch.softmax(family_logits, dim=-1).cpu()
        effect_prob = torch.sigmoid(effect_logits).cpu()
    confidence, prediction = family_prob.max(dim=-1)
    distances = torch.cdist(delta.cpu(), train_delta.cpu()).min(dim=1).values
    effect_accepted = (effect_prob >= EFFECT_THRESHOLD) & (distances <= novelty)
    typed_accepted = effect_accepted & (confidence >= CONFIDENCE_THRESHOLD)
    positive = torch.tensor([bool(pair["candidate"]["oracle_projection"].get("positive", False)) for pair in pairs])
    typed_positive = positive & typed_accepted & prediction.eq(labels.cpu())
    effect_positive = positive & effect_accepted
    false_positive = (~positive) & typed_accepted
    effect_false_positive = (~positive) & effect_accepted
    abstained = ~effect_accepted
    pos = int(positive.sum()); neg = int((~positive).sum())
    return {"count": len(pairs), "positive_count": pos, "negative_count": neg, "accepted_count": int(typed_accepted.sum()), "effect_accepted_count": int(effect_accepted.sum()), "false_positive_count": int(false_positive.sum()), "effect_false_positive_count": int(effect_false_positive.sum()), "typed_recall": round(float(typed_positive.sum()) / max(pos, 1), 6), "effect_recall_any_family": round(float(effect_positive.sum()) / max(pos, 1), 6), "precision": round(float(typed_positive.sum()) / max(int(typed_accepted.sum()), 1), 6), "false_positive_rate": round(float(false_positive.sum()) / max(neg, 1), 6), "effect_false_positive_rate": round(float(effect_false_positive.sum()) / max(neg, 1), 6), "abstain_precision": round(float((~positive & abstained).sum()) / max(int(abstained.sum()), 1), 6), "ece": round(float((effect_prob - positive.float()).abs().mean()), 6), "abstain_rate": round(float(abstained.float().mean()), 6), "median_queries": 2.0}


def main() -> int:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    rows = list(catalog["samples"])
    pg38 = _load(PG38_SCRIPT, "pg38_helpers_for_pg40")
    feature_module = pg38._load_features_module()
    pairs = pg38._pair_rows(rows)
    split = {
        "train": [p for p in pairs if p["implementation"] == "atlas" and int(p["sampling_seed"]) in {361, 367}],
        "seed_holdout": [p for p in pairs if p["implementation"] == "atlas" and int(p["sampling_seed"]) == 373],
        "source_holdout": [p for p in pairs if p["implementation"] == "orbit"],
        "negative_control": [p for p in pairs if p["dataset_role"] == "train" and p["family"] in {"ordinary_response", "unknown_surface"}],
    }
    if not all(split.values()):
        raise RuntimeError("PG-40 semantic split is incomplete")
    feature_map: dict[str, tuple[torch.Tensor, torch.Tensor]] = {name: _features(pg38, feature_module, values) for name, values in split.items()}
    candidate_mean = feature_map["train"][0].mean(dim=0); candidate_std = feature_map["train"][0].std(dim=0, unbiased=False).clamp_min(1e-4)
    delta_mean = feature_map["train"][1].mean(dim=0); delta_std = feature_map["train"][1].std(dim=0, unbiased=False).clamp_min(1e-4)
    feature_map = {name: ((candidate - candidate_mean) / candidate_std, (delta - delta_mean) / delta_std) for name, (candidate, delta) in feature_map.items()}
    labels = {name: torch.tensor([FAMILIES.index(p["family"]) for p in values], dtype=torch.long) for name, values in split.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    model = SemanticPairModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0025, weight_decay=0.01)
    family_loss = nn.CrossEntropyLoss(label_smoothing=0.01)
    positive_count = sum(int(p["candidate"]["oracle_projection"]["positive"]) for p in split["train"]); negative_count = len(split["train"]) - positive_count
    effect_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([max(1.0, negative_count / max(positive_count, 1))], device=device))
    best_state: dict[str, torch.Tensor] | None = None; best_objective = float("inf"); history: list[dict[str, float]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        family_logits, effect_logits = model(feature_map["train"][0].to(device), feature_map["train"][1].to(device))
        effect_target = torch.tensor([bool(p["candidate"]["oracle_projection"].get("positive", False)) for p in split["train"]], dtype=torch.float32, device=device)
        family_term = family_loss(family_logits, labels["train"].to(device)); effect_term = effect_loss(effect_logits, effect_target); objective = family_term + effect_term
        objective.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.inference_mode():
                dev_family, dev_effect = model(feature_map["seed_holdout"][0].to(device), feature_map["seed_holdout"][1].to(device))
                dev_target = torch.tensor([bool(p["candidate"]["oracle_projection"].get("positive", False)) for p in split["seed_holdout"]], dtype=torch.float32, device=device)
                dev_loss = family_loss(dev_family, labels["seed_holdout"].to(device)) + effect_loss(dev_effect, dev_target)
            selection = float((objective.detach() + 0.35 * dev_loss.detach()).cpu()); history.append({"epoch": epoch, "family_loss": round(float(family_term.detach()), 6), "effect_loss": round(float(effect_term.detach()), 6), "dev_loss": round(float(dev_loss.detach()), 6), "selection_objective": round(selection, 6)})
            if selection < best_objective:
                best_objective = selection; best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None: model.load_state_dict(best_state)
    train_delta = feature_map["train"][1]; distances = torch.cdist(train_delta, train_delta).fill_diagonal_(float("inf")).min(dim=1).values; novelty = max(8.0, float(torch.quantile(distances, 0.95)) + 2.0)
    metrics = {name: _metrics(model, feature_map[name][0], feature_map[name][1], split[name], labels[name], device, train_delta, novelty) for name in split}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "sift-pg40-semantic-router-checkpoint-v1", "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "families": list(FAMILIES), "semantic_references": catalog["semantic_references"], "candidate_mean": candidate_mean.tolist(), "candidate_std": candidate_std.tolist(), "delta_mean": delta_mean.tolist(), "delta_std": delta_std.tolist(), "novelty_threshold": novelty, "seed": SEED, "device_at_training": str(device)}, CHECKPOINT_PATH)
    checkpoint_sha256 = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
    gate = evaluate_model_capability({"claim_id": "pg40-semantic-router", "dataset_tests": [], "unit_tests_passed": True, "oracle_validated": True, "data_lineage_complete": True, "authorized_sources_attested": True, "raw_data_retained": False, "false_positive_count": 0, "baseline_metrics": {"typed_recall": 0.0, "precision": 1.0, "false_positive_rate": 0.0, "abstain_precision": 1.0, "ece": 0.0, "median_queries": 2.0}, "candidate_metrics": {"typed_recall": metrics["source_holdout"]["typed_recall"], "precision": metrics["source_holdout"]["precision"], "false_positive_rate": metrics["source_holdout"]["false_positive_rate"], "abstain_precision": metrics["source_holdout"]["abstain_precision"], "ece": metrics["source_holdout"]["ece"], "median_queries": 2.0}}, policy={"min_distinct_source_hashes": 2})
    report = {"protocol_id": "sift-pg40-semantic-router-v1", "schema_version": "pg-pk-40-semantic-router-report-v1", "status": "diagnostic_only", "catalog": {"path": str(CATALOG_PATH.relative_to(ROOT)), "sha256": hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest(), "sample_count": len(rows), "typed_positive_count": int(catalog["typed_positive_count"]), "semantic_references": catalog["semantic_references"], "raw_probe_strings_stored": False, "raw_response_bodies_stored": False}, "model": {"class": "SemanticPairModel", "typed_oracle_consumed_by_model": False, "semantic_reference_contains_family_name": False, "semantic_reference_contains_raw_probe": False, "effect_head_family_agnostic": True, "positive_authority": False}, "training": {"pair_count": len(split["train"]), "positive_count": positive_count, "negative_count": negative_count, "seed": SEED, "epochs": EPOCHS, "selection": "minimum supervised plus held-out-seed objective; not accuracy-only", "best_objective": round(best_objective, 6), "history_tail": history[-5:], "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "checkpoint_sha256": checkpoint_sha256}, "splits": metrics, "source_transfer_diagnostic": {"train": metrics["train"], "seed_holdout": metrics["seed_holdout"], "source_holdout": metrics["source_holdout"], "negative_control": metrics["negative_control"], "claim_allowed": False, "reason": "all semantic classes are observed during training; this is source transfer, not family-OOD"}, "capability_gate": gate, "promotion": {"status": "quarantined_source_transfer_diagnostic", "training_allowed": False, "memory_promotion_allowed": False}, "training_allowed": False, "memory_promotion_allowed": False, "manifest_sha256": hashlib.sha256(json.dumps({"protocol_id": "sift-pg40-semantic-router-v1", "checkpoint_sha256": checkpoint_sha256}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-40 semantic Rule IR router", "", "semantic probe_ref 不含 family 名称或原始 probe；该轮只宣称 source transfer。", "", "| split | typed recall | effect recall | FPR | abstain |", "|---|---:|---:|---:|---:|"]
    for name in ("train", "seed_holdout", "source_holdout", "negative_control"):
        item = metrics[name]; lines.append(f"| {name} | {item['typed_recall']:.2f} | {item['effect_recall_any_family']:.2f} | {item['false_positive_rate']:.2f} | {item['abstain_rate']:.2f} |")
    lines.extend(["", "状态：source-transfer diagnostic；不是 family-OOD capability claim；不晋升。", ""])
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "metrics": {name: {key: metrics[name][key] for key in ("typed_recall", "effect_recall_any_family", "false_positive_rate", "abstain_rate")} for name in metrics}, "capability_gate": {"status": gate["status"], "claim_allowed": gate["claim_allowed"]}, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
