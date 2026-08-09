"""Run PG-45 leave-one-family-out Rule IR decoder experiment.

The injection/operator-context class is removed from optimization and from
the semantic ontology.  PG-42 is an untouched independent evaluation set.
The expected behavior for the removed family is safe abstention, not a forced
guess.
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
OUTPUT_DIR = ROOT / "artifacts" / "pg45-leave-one-family-out"
CHECKPOINT_PATH = OUTPUT_DIR / "leave_one_out.pt"
REPORT_PATH = ROOT / "research" / "pg45_leave_one_family_out_report_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg45_leave_one_family_out_report_v1.md"
SEED = 20450802
EPOCHS = 360
CONFIDENCE_THRESHOLD = 0.60
HELD_OUT_FAMILY = "injection"
HELD_OUT_SEMANTIC = "operator-context"
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


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {path.name}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


class LeaveOneOutDecoder(nn.Module):
    def __init__(self, input_dim: int, class_count: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.head = nn.Linear(hidden_dim, class_count)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(features))


def _semantic(pair: dict[str, Any], prefix: str) -> str:
    ref = str((pair["candidate"].get("payload_manifest") or {}).get("probe_ref", ""))
    return ref[len(prefix) :] if ref.startswith(prefix) else ref


def _invariant(pg39: Any, pairs: list[dict[str, Any]], invariant_indices: tuple[int, ...]) -> torch.Tensor:
    if not pairs:
        return torch.empty((0, len(invariant_indices)), dtype=torch.float32)
    return torch.sign(torch.stack([pg39._coarse_pair(pair) for pair in pairs])[:, invariant_indices])


def _features(pg39: Any, pairs: list[dict[str, Any]], prefix: str, semantic_index: dict[str, int], invariant_indices: tuple[int, ...]) -> torch.Tensor:
    invariant = _invariant(pg39, pairs, invariant_indices)
    one_hot = torch.zeros((len(pairs), len(semantic_index)), dtype=torch.float32)
    for index, pair in enumerate(pairs):
        semantic = _semantic(pair, prefix)
        if semantic in semantic_index:
            one_hot[index, semantic_index[semantic]] = 1.0
    return torch.cat([invariant, one_hot], dim=1)


def _effect_accepted(pg39: Any, pg43: Any, pairs: list[dict[str, Any]], train_pairs: list[dict[str, Any]], checkpoint: dict[str, Any]) -> torch.Tensor:
    indices = tuple(int(item) for item in checkpoint["invariant_indices"])
    features = _invariant(pg39, pairs, indices)
    train_features = _invariant(pg39, train_pairs, indices)
    effect_model = pg43.InvariantEffectModel(); effect_model.load_state_dict(checkpoint["model_state"]); effect_model.eval()
    with torch.inference_mode(): probability = torch.sigmoid(effect_model(features))
    distance = torch.cdist(features, train_features).min(dim=1).values
    return (probability >= 0.60) & (distance <= float(checkpoint["novelty_threshold"]))


def _metrics(model: LeaveOneOutDecoder, pairs: list[dict[str, Any]], features: torch.Tensor, effect: torch.Tensor, semantic_index: dict[str, int], output_families: tuple[str, ...], prefix: str, device: torch.device) -> tuple[dict[str, Any], dict[str, int]]:
    if not pairs:
        return {"pair_count": 0, "positive_count": 0, "negative_count": 0, "known_positive_count": 0, "known_family_recall": 0.0, "unknown_positive_count": 0, "unknown_effect_recall": 0.0, "accepted_count": 0, "precision": 1.0, "false_positive_rate": 0.0, "unknown_misname_count": 0, "unknown_not_abstain_count": 0, "unknown_strict_abstain": True}, {}
    with torch.inference_mode():
        logits = model(features.to(device)).cpu(); probs = torch.softmax(logits, dim=-1)
    confidence, prediction = probs.max(dim=-1)
    positive = torch.tensor([bool(pair["candidate"]["oracle_projection"].get("positive", False)) for pair in pairs])
    known_positive = known_correct = unknown_positive = unknown_effect = false_accept = misname = not_abstain = accepted_count = 0
    routes: dict[str, int] = {}
    for index, pair in enumerate(pairs):
        semantic = _semantic(pair, prefix); known = semantic in semantic_index; effect_accepted = bool(effect[index]); confident = bool(confidence[index] >= CONFIDENCE_THRESHOLD)
        if known and effect_accepted and confident:
            route = output_families[int(prediction[index])]; abstain = False; accepted_count += 1
        else:
            route = "unknown_surface"; abstain = True
        routes[route] = routes.get(route, 0) + 1
        if positive[index] and known:
            known_positive += 1; known_correct += int(not abstain and route == KNOWN_BINDINGS[semantic])
        if positive[index] and not known:
            unknown_positive += 1; unknown_effect += int(effect_accepted); misname += int(route != "unknown_surface"); not_abstain += int(not abstain)
        false_accept += int((not positive[index]) and not abstain)
    negative_count = int((~positive).sum())
    return {"pair_count": len(pairs), "positive_count": int(positive.sum()), "negative_count": negative_count, "known_positive_count": known_positive, "known_family_recall": round(known_correct / max(known_positive, 1), 6), "unknown_positive_count": unknown_positive, "unknown_effect_recall": round(unknown_effect / max(unknown_positive, 1), 6), "accepted_count": accepted_count, "precision": round(known_correct / max(accepted_count, 1), 6), "false_positive_rate": round(false_accept / max(negative_count, 1), 6), "unknown_misname_count": misname, "unknown_not_abstain_count": not_abstain, "unknown_strict_abstain": misname == 0 and not_abstain == 0, "mean_decoder_confidence": round(float(confidence.mean()), 6)}, routes


def main() -> int:
    pg38 = _load(PG38_SCRIPT, "pg38_for_pg45")
    pg39 = _load(PG39_SCRIPT, "pg39_for_pg45")
    pg43 = _load(PG43_SCRIPT, "pg43_for_pg45")
    pg40 = json.loads(PG40_CATALOG_PATH.read_text(encoding="utf-8")); pg42 = json.loads(PG42_CATALOG_PATH.read_text(encoding="utf-8")); pg37 = json.loads(PG37_CATALOG_PATH.read_text(encoding="utf-8"))
    pg40_pairs = pg38._pair_rows(list(pg40["samples"])); pg42_pairs = pg38._pair_rows(list(pg42["samples"])); pg37_pairs = pg38._pair_rows(list(pg37["samples"]))
    remaining_bindings = {semantic: family for semantic, family in KNOWN_BINDINGS.items() if semantic != HELD_OUT_SEMANTIC}
    output_families = tuple(sorted(set(remaining_bindings.values())))
    semantic_index = {semantic: index for index, semantic in enumerate(sorted(remaining_bindings))}
    train_pairs = [pair for pair in pg40_pairs if pair["implementation"] == "atlas" and int(pair["sampling_seed"]) in {361, 367} and _semantic(pair, "pg40-semantic-") in remaining_bindings and str(pair["family"]) in output_families]
    seed_holdout_pairs = [pair for pair in pg40_pairs if pair["implementation"] == "atlas" and int(pair["sampling_seed"]) == 373 and _semantic(pair, "pg40-semantic-") in remaining_bindings and str(pair["family"]) in output_families]
    invariant_indices = tuple(int(item) for item in torch.load(ROOT / "artifacts" / "pg43-invariant-effect" / "invariant_effect.pt", map_location="cpu", weights_only=False)["invariant_indices"])
    train_features = _features(pg39, train_pairs, "pg40-semantic-", semantic_index, invariant_indices); seed_features = _features(pg39, seed_holdout_pairs, "pg40-semantic-", semantic_index, invariant_indices)
    labels = torch.tensor([output_families.index(str(pair["family"])) for pair in train_pairs], dtype=torch.long); seed_labels = torch.tensor([output_families.index(str(pair["family"])) for pair in seed_holdout_pairs], dtype=torch.long)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    model = LeaveOneOutDecoder(train_features.shape[1], len(output_families)).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=0.01); loss_fn = nn.CrossEntropyLoss(label_smoothing=0.01)
    best_state: dict[str, torch.Tensor] | None = None; best_selection = float("inf"); history: list[dict[str, float]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); train_loss = loss_fn(model(train_features.to(device)), labels.to(device)); train_loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if epoch % 60 == 0 or epoch == 1:
            model.eval()
            with torch.inference_mode(): seed_loss = loss_fn(model(seed_features.to(device)), seed_labels.to(device))
            selection = float((train_loss.detach() + 0.5 * seed_loss.detach()).cpu()); history.append({"epoch": epoch, "train_loss": round(float(train_loss.detach()), 6), "seed_loss": round(float(seed_loss.detach()), 6), "selection": round(selection, 6)})
            if selection < best_selection: best_selection = selection; best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None: model.load_state_dict(best_state)
    effect_checkpoint = torch.load(ROOT / "artifacts" / "pg43-invariant-effect" / "invariant_effect.pt", map_location="cpu", weights_only=False)
    pg42_groups = {
        "retained_known": [pair for pair in pg42_pairs if _semantic(pair, "pg42-semantic-") in remaining_bindings],
        "held_out_family": [pair for pair in pg42_pairs if _semantic(pair, "pg42-semantic-") == HELD_OUT_SEMANTIC],
        "other_unknown": [pair for pair in pg42_pairs if _semantic(pair, "pg42-semantic-") not in remaining_bindings and _semantic(pair, "pg42-semantic-") != HELD_OUT_SEMANTIC],
        "negative_control": [pair for pair in pg42_pairs if pair["candidate"].get("dataset_role") == "negative_control"],
    }
    split_metrics: dict[str, dict[str, Any]] = {}; routes: dict[str, dict[str, int]] = {}
    for name, group in pg42_groups.items():
        features = _features(pg39, group, "pg42-semantic-", semantic_index, invariant_indices)
        effect = _effect_accepted(pg39, pg43, group, pg37_pairs, effect_checkpoint)
        split_metrics[name], routes[name] = _metrics(model, group, features, effect, semantic_index, output_families, "pg42-semantic-", device)
    seed_metrics, seed_routes = _metrics(model, seed_holdout_pairs, seed_features, _effect_accepted(pg39, pg43, seed_holdout_pairs, train_pairs, effect_checkpoint), semantic_index, output_families, "pg40-semantic-", device)
    heldout = split_metrics["held_out_family"]
    gate_reasons = []
    if split_metrics["retained_known"]["known_family_recall"] < 1.0: gate_reasons.append("retained_family_recall_below_1")
    if heldout["unknown_misname_count"] != 0 or heldout["unknown_not_abstain_count"] != 0: gate_reasons.append("held_out_family_not_strict_abstain")
    if split_metrics["negative_control"]["false_positive_rate"] != 0.0: gate_reasons.append("negative_false_accept")
    safe_gate = {"status": "passed" if not gate_reasons else "blocked", "claim_allowed": not gate_reasons, "reasons": gate_reasons, "training_allowed": False, "memory_promotion_allowed": False}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True); torch.save({"schema_version": "sift-pg45-leave-one-out-checkpoint-v1", "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "held_out_family": HELD_OUT_FAMILY, "held_out_semantic": HELD_OUT_SEMANTIC, "remaining_semantic_index": semantic_index, "output_families": list(output_families), "invariant_indices": list(invariant_indices), "seed": SEED}, CHECKPOINT_PATH)
    checkpoint_sha256 = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
    report = {"protocol_id": "sift-pg45-leave-one-family-out-v1", "schema_version": "pg-pk-45-leave-one-family-out-report-v1", "status": "diagnostic_only", "held_out": {"family": HELD_OUT_FAMILY, "semantic_reference": HELD_OUT_SEMANTIC, "removed_from_training": True}, "training": {"catalog": str(PG40_CATALOG_PATH.relative_to(ROOT)), "pair_count": len(train_pairs), "positive_count": sum(bool(pair["candidate"]["oracle_projection"].get("positive", False)) for pair in train_pairs), "remaining_semantic_index": semantic_index, "output_families": list(output_families), "pg42_used_for_training": False, "typed_oracle_consumed_by_model": False, "epochs": EPOCHS, "seed": SEED, "device": str(device), "best_selection": round(best_selection, 6), "history_tail": history[-5:]}, "seed_holdout": seed_metrics, "pg42_splits": split_metrics, "routes": routes, "safe_leave_one_out_gate": safe_gate, "formal_capability_claim_allowed": False, "formal_claim_blockers": ["removed_family_is_not_named", "positive_named_evidence_for_new_family_is_absent"], "promotion": {"status": "quarantined_candidate", "training_allowed": False, "memory_promotion_allowed": False}, "checkpoint": {"path": str(CHECKPOINT_PATH.relative_to(ROOT)), "sha256": checkpoint_sha256}, "manifest_sha256": hashlib.sha256(json.dumps({"protocol_id": "sift-pg45-leave-one-family-out-v1", "held_out_family": HELD_OUT_FAMILY, "pg40_sha256": hashlib.sha256(PG40_CATALOG_PATH.read_bytes()).hexdigest(), "pg42_sha256": hashlib.sha256(PG42_CATALOG_PATH.read_bytes()).hexdigest(), "checkpoint_sha256": checkpoint_sha256}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-45 leave-one-family-out", "", "训练中移除 injection/operator-context；被移除族必须 unknown + abstain。", "", "| split | known recall | unknown effect recall | FPR | strict abstain |", "|---|---:|---:|---:|---:|"]
    for name, item in {"seed_holdout": seed_metrics, **split_metrics}.items(): lines.append(f"| {name} | {item['known_family_recall']:.2f} | {item['unknown_effect_recall']:.2f} | {item['false_positive_rate']:.2f} | {item['unknown_strict_abstain']} |")
    lines.extend(["", f"安全 leave-one-out gate：`{safe_gate['status']}`；formal capability claim=false。", ""])
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "held_out": report["held_out"], "seed_holdout": seed_metrics, "pg42_splits": split_metrics, "safe_leave_one_out_gate": safe_gate, "formal_capability_claim_allowed": False, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
