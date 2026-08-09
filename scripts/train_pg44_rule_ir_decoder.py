"""Train and evaluate PG-44's closed-world Rule IR family decoder.

The decoder is deliberately small and quarantined.  It learns bindings for
the nine semantic references observed in PG-40, while an explicit ontology
guard routes any unseen PG-42 reference to ``unknown_surface``.  Typed oracle
labels are targets only; the decoder never receives them as features.
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

PG40_CATALOG_PATH = ROOT / "research" / "pg40_semantic_router_catalog_v1.json"
PG42_CATALOG_PATH = ROOT / "research" / "pg42_independent_semantic_catalog_v1.json"
PG37_CATALOG_PATH = ROOT / "research" / "pg37_counterfactual_catalog_v1.json"
PG38_SCRIPT = ROOT / "scripts" / "train_pg38_effect_pair_candidate.py"
PG39_SCRIPT = ROOT / "scripts" / "train_pg39_coarse_delta_candidate.py"
PG43_SCRIPT = ROOT / "scripts" / "train_pg43_invariant_effect_candidate.py"
PG43_CHECKPOINT_PATH = ROOT / "artifacts" / "pg43-invariant-effect" / "invariant_effect.pt"
OUTPUT_DIR = ROOT / "artifacts" / "pg44-rule-ir-decoder"
CHECKPOINT_PATH = OUTPUT_DIR / "rule_ir_decoder.pt"
REPORT_PATH = ROOT / "research" / "pg44_rule_ir_decoder_report_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg44_rule_ir_decoder_report_v1.md"
SEED = 20440802
EPOCHS = 360
CONFIDENCE_THRESHOLD = 0.60
KNOWN_BINDINGS = {
    "markup-context": "xss",
    "operator-context": "injection",
    "auth-boundary": "authentication",
    "subject-boundary": "access_control",
    "state-invariant": "logic",
    "url-target": "url_redirect",
    "scalar-boundary": "input_validation",
    "local-canary": "command_injection",
    "ordinary-surface": "ordinary_response",
}
FAMILIES = tuple(sorted(set(KNOWN_BINDINGS.values())))


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {path.name}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


class RuleIRDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96, class_count: int = len(FAMILIES)) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.head = nn.Linear(hidden_dim, class_count)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(features))


def _semantic(pair: dict[str, Any], prefix: str) -> str:
    ref = str((pair["candidate"].get("payload_manifest") or {}).get("probe_ref", ""))
    return ref[len(prefix) :] if ref.startswith(prefix) else ref


def _invariant(pg39: Any, pg43: Any, pairs: list[dict[str, Any]], invariant_indices: tuple[int, ...]) -> torch.Tensor:
    if not pairs:
        return torch.empty((0, len(invariant_indices)), dtype=torch.float32)
    raw = torch.stack([pg39._coarse_pair(pair) for pair in pairs])[:, invariant_indices]
    return torch.sign(raw)


def _decoder_features(pg39: Any, pg43: Any, pairs: list[dict[str, Any]], prefix: str, semantic_index: dict[str, int], invariant_indices: tuple[int, ...]) -> torch.Tensor:
    invariant = _invariant(pg39, pg43, pairs, invariant_indices)
    one_hot = torch.zeros((len(pairs), len(semantic_index)), dtype=torch.float32)
    for index, pair in enumerate(pairs):
        semantic = _semantic(pair, prefix)
        if semantic in semantic_index:
            one_hot[index, semantic_index[semantic]] = 1.0
    return torch.cat([invariant, one_hot], dim=1)


def _labels(pairs: list[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor([FAMILIES.index(str(pair["family"])) for pair in pairs], dtype=torch.long)


def _effect_accepted(pg38: Any, pg39: Any, pg43: Any, pairs: list[dict[str, Any]], train_pairs: list[dict[str, Any]], checkpoint: dict[str, Any]) -> torch.Tensor:
    invariant_indices = tuple(int(index) for index in checkpoint["invariant_indices"])
    features = _invariant(pg39, pg43, pairs, invariant_indices)
    train_features = _invariant(pg39, pg43, train_pairs, invariant_indices)
    model = pg43.InvariantEffectModel(); model.load_state_dict(checkpoint["model_state"]); model.eval()
    with torch.inference_mode():
        probability = torch.sigmoid(model(features))
    distance = torch.cdist(features, train_features).min(dim=1).values
    return (probability >= 0.60) & (distance <= float(checkpoint["novelty_threshold"]))


def _decoder_metrics(model: RuleIRDecoder, pairs: list[dict[str, Any]], features: torch.Tensor, effect_accepted: torch.Tensor, prefix: str, semantic_index: dict[str, int], device: torch.device) -> tuple[dict[str, Any], dict[str, int]]:
    if not pairs:
        return {"pair_count": 0, "positive_count": 0, "negative_count": 0, "accepted_count": 0, "typed_recall": 0.0, "precision": 1.0, "false_positive_rate": 0.0, "unknown_misname_count": 0, "unknown_not_abstain_count": 0, "unknown_strict_abstain": True, "decoder_confidence_mean": 0.0}, {}
    with torch.inference_mode():
        logits = model(features.to(device)).cpu()
        probabilities = torch.softmax(logits, dim=-1)
    confidence, prediction = probabilities.max(dim=-1)
    positive = torch.tensor([bool(pair["candidate"]["oracle_projection"].get("positive", False)) for pair in pairs])
    routes: dict[str, int] = {}
    typed_correct = false_accept = unknown_misname = unknown_not_abstain = 0
    known_positive_count = known_correct = unknown_positive_count = unknown_effect_accepted = 0
    accepted_count = 0
    for index, pair in enumerate(pairs):
        semantic = _semantic(pair, prefix)
        known = semantic in semantic_index
        effect = bool(effect_accepted[index])
        confident = bool(confidence[index] >= CONFIDENCE_THRESHOLD)
        if known and effect and confident:
            route = FAMILIES[int(prediction[index])]
            abstain = False
        else:
            route = "unknown_surface"
            abstain = True
        routes[route] = routes.get(route, 0) + 1
        accepted_count += int(not abstain)
        if positive[index] and known:
            known_positive_count += 1
            known_correct += int(not abstain and route == KNOWN_BINDINGS[semantic])
        if positive[index] and not known:
            unknown_positive_count += 1
            unknown_effect_accepted += int(effect)
            unknown_misname += int(route != "unknown_surface")
            unknown_not_abstain += int(not abstain)
        if positive[index] and known:
            typed_correct += int(not abstain and route == KNOWN_BINDINGS[semantic])
        if (not positive[index]) and not abstain:
            false_accept += 1
    positive_count = int(positive.sum()); negative_count = int((~positive).sum())
    return {"pair_count": len(pairs), "positive_count": positive_count, "negative_count": negative_count, "accepted_count": accepted_count, "typed_recall": round(typed_correct / max(positive_count, 1), 6), "known_positive_count": known_positive_count, "known_family_recall": round(known_correct / max(known_positive_count, 1), 6), "unknown_positive_count": unknown_positive_count, "unknown_effect_recall": round(unknown_effect_accepted / max(unknown_positive_count, 1), 6), "precision": round(typed_correct / max(accepted_count, 1), 6), "false_positive_rate": round(false_accept / max(negative_count, 1), 6), "unknown_misname_count": unknown_misname, "unknown_not_abstain_count": unknown_not_abstain, "unknown_strict_abstain": unknown_misname == 0 and unknown_not_abstain == 0, "decoder_confidence_mean": round(float(confidence.mean()), 6)}, routes


def main() -> int:
    pg38 = _load(PG38_SCRIPT, "pg38_for_pg44")
    pg39 = _load(PG39_SCRIPT, "pg39_for_pg44")
    pg43 = _load(PG43_SCRIPT, "pg43_for_pg44")
    pg40 = json.loads(PG40_CATALOG_PATH.read_text(encoding="utf-8"))
    pg42 = json.loads(PG42_CATALOG_PATH.read_text(encoding="utf-8"))
    pg37 = json.loads(PG37_CATALOG_PATH.read_text(encoding="utf-8"))
    pg40_pairs = pg38._pair_rows(list(pg40["samples"]))
    pg42_pairs = pg38._pair_rows(list(pg42["samples"]))
    pg37_pairs = pg38._pair_rows(list(pg37["samples"]))
    # Only atlas seeds 361/367 and known ontology refs enter optimization.
    train_pairs = [pair for pair in pg40_pairs if pair["implementation"] == "atlas" and int(pair["sampling_seed"]) in {361, 367} and _semantic(pair, "pg40-semantic-") in KNOWN_BINDINGS and str(pair["family"]) in FAMILIES]
    seed_holdout_pairs = [pair for pair in pg40_pairs if pair["implementation"] == "atlas" and int(pair["sampling_seed"]) == 373 and _semantic(pair, "pg40-semantic-") in KNOWN_BINDINGS and str(pair["family"]) in FAMILIES]
    semantic_index = {semantic: index for index, semantic in enumerate(sorted(KNOWN_BINDINGS))}
    invariant_indices = tuple(int(index) for index in torch.load(PG43_CHECKPOINT_PATH, map_location="cpu", weights_only=False)["invariant_indices"])
    train_features = _decoder_features(pg39, pg43, train_pairs, "pg40-semantic-", semantic_index, invariant_indices)
    seed_features = _decoder_features(pg39, pg43, seed_holdout_pairs, "pg40-semantic-", semantic_index, invariant_indices)
    labels = _labels(train_pairs); seed_labels = _labels(seed_holdout_pairs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    model = RuleIRDecoder(train_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.01)
    best_state: dict[str, torch.Tensor] | None = None; best_selection = float("inf"); history: list[dict[str, float]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        train_loss = loss_fn(model(train_features.to(device)), labels.to(device)); train_loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if epoch % 60 == 0 or epoch == 1:
            model.eval()
            with torch.inference_mode(): seed_loss = loss_fn(model(seed_features.to(device)), seed_labels.to(device))
            selection = float((train_loss.detach() + 0.5 * seed_loss.detach()).cpu()); history.append({"epoch": epoch, "train_loss": round(float(train_loss.detach()), 6), "seed_loss": round(float(seed_loss.detach()), 6), "selection": round(selection, 6)})
            if selection < best_selection:
                best_selection = selection; best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None: model.load_state_dict(best_state)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "sift-pg44-rule-ir-decoder-checkpoint-v1", "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "semantic_index": semantic_index, "families": list(FAMILIES), "invariant_indices": list(invariant_indices), "seed": SEED, "device_at_training": str(device)}, CHECKPOINT_PATH)
    checkpoint_sha256 = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
    pg43_effect_checkpoint = torch.load(PG43_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    # PG-43 is evaluated independently; this model never sees PG-42 labels.
    pg42_train_pairs = [pair for pair in pg42_pairs if pair["candidate"].get("dataset_role") == "train"]
    pg42_dev_pairs = [pair for pair in pg42_pairs if pair["candidate"].get("dataset_role") == "dev"]
    pg42_impl_pairs = [pair for pair in pg42_pairs if pair["candidate"].get("dataset_role") == "ood_source"]
    pg42_family_pairs = [pair for pair in pg42_pairs if pair["candidate"].get("dataset_role") == "family_holdout"]
    pg42_negative_pairs = [pair for pair in pg42_pairs if pair["candidate"].get("dataset_role") == "negative_control"]
    # Decoder semantic prefix is different from training but the references are
    # still bounded identifiers; this is intentional source transfer.
    def evaluate_pg42(name: str, group: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, int]]:
        features = _decoder_features(pg39, pg43, group, "pg42-semantic-", semantic_index, invariant_indices)
        effect = _effect_accepted(pg38, pg39, pg43, group, pg37_pairs, pg43_effect_checkpoint)
        return _decoder_metrics(model, group, features, effect, "pg42-semantic-", semantic_index, device)
    pg42_splits: dict[str, dict[str, Any]] = {}; routes: dict[str, dict[str, int]] = {}
    for name, group in (("train_role_diagnostic", pg42_train_pairs), ("dev", pg42_dev_pairs), ("implementation_holdout", pg42_impl_pairs), ("family_holdout", pg42_family_pairs), ("negative_control", pg42_negative_pairs)):
        pg42_splits[name], routes[name] = evaluate_pg42(name, group)
    pg40_seed_metrics, pg40_seed_routes = _decoder_metrics(model, seed_holdout_pairs, seed_features, _effect_accepted(pg38, pg39, pg43, seed_holdout_pairs, train_pairs, pg43_effect_checkpoint), "pg40-semantic-", semantic_index, device)
    report = {"protocol_id": "sift-pg44-rule-ir-decoder-v1", "schema_version": "pg-pk-44-rule-ir-decoder-report-v1", "status": "diagnostic_only", "training": {"catalog": str(PG40_CATALOG_PATH.relative_to(ROOT)), "pair_count": len(train_pairs), "positive_count": sum(bool(pair["candidate"]["oracle_projection"].get("positive", False)) for pair in train_pairs), "implementation": "atlas", "seeds": [361, 367], "semantic_reference_contains_family_name": False, "semantic_reference_contains_raw_probe": False, "typed_oracle_consumed_by_model": False, "pg42_used_for_training": False, "epochs": EPOCHS, "seed": SEED, "device": str(device), "best_selection": round(best_selection, 6), "history_tail": history[-5:]}, "model": {"class": "RuleIRDecoder", "semantic_index": semantic_index, "families": list(FAMILIES), "input_contract": "invariant shape delta + bounded semantic one-hot", "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "checkpoint_sha256": checkpoint_sha256, "executable": False}, "pg40_seed_holdout": pg40_seed_metrics, "pg42_splits": pg42_splits, "routes": routes, "unknown_route": "unknown_surface", "unknown_requires_abstain": True, "formal_capability_claim_allowed": False, "formal_claim_blockers": ["closed_world_ontology_prior_is_not_unseen_family_learning", "Rule_IR_decoder_has_not_been_validated_on_a_new_family_name"], "promotion": {"status": "quarantined_candidate", "training_allowed": False, "memory_promotion_allowed": False}, "manifest_sha256": hashlib.sha256(json.dumps({"protocol_id": "sift-pg44-rule-ir-decoder-v1", "pg40_catalog_sha256": hashlib.sha256(PG40_CATALOG_PATH.read_bytes()).hexdigest(), "pg42_catalog_sha256": hashlib.sha256(PG42_CATALOG_PATH.read_bytes()).hexdigest(), "checkpoint_sha256": checkpoint_sha256}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-44 Rule IR decoder", "", "effect gate 与 family decoder 分离；未知 semantic ref 一律 `unknown_surface + abstain`。", "", "| split | typed recall | precision | FPR | unknown abstain |", "|---|---:|---:|---:|---:|"]
    for name, item in {"pg40_seed_holdout": pg40_seed_metrics, **{f"pg42_{key}": value for key, value in pg42_splits.items()}}.items():
        lines.append(f"| {name} | {item['typed_recall']:.2f} | {item['precision']:.2f} | {item['false_positive_rate']:.2f} | {item['unknown_strict_abstain']} |")
    lines.extend(["", "正式 capability claim=false；该候选只证明已知 ontology source transfer 与未知安全 abstain。", ""])
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "pg40_seed_holdout": pg40_seed_metrics, "pg42_splits": pg42_splits, "formal_capability_claim_allowed": False, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
