"""Run PG-43's invariant effect head through the PG-41 safe router on PG-42."""

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
CHECKPOINT_PATH = ROOT / "artifacts" / "pg43-invariant-effect" / "invariant_effect.pt"
PG38_SCRIPT = ROOT / "scripts" / "train_pg38_effect_pair_candidate.py"
PG39_SCRIPT = ROOT / "scripts" / "train_pg39_coarse_delta_candidate.py"
PG43_SCRIPT = ROOT / "scripts" / "train_pg43_invariant_effect_candidate.py"
REPORT_PATH = ROOT / "research" / "pg43_pg42_safe_router_report_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg43_pg42_safe_router_report_v1.md"
KNOWN_BINDINGS = {"markup-context": "xss", "operator-context": "injection", "auth-boundary": "authentication", "subject-boundary": "access_control"}
UNKNOWN_ROUTE = "unknown_surface"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {path.name}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def main() -> int:
    pg38 = _load(PG38_SCRIPT, "pg38_for_pg43_router")
    pg39 = _load(PG39_SCRIPT, "pg39_for_pg43_router")
    pg43 = _load(PG43_SCRIPT, "pg43_for_pg43_router")
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    pairs = pg38._pair_rows(list(catalog["samples"]))
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = pg43.InvariantEffectModel(); model.load_state_dict(checkpoint["model_state"]); model.eval()
    raw = torch.stack([pg39._coarse_pair(pair) for pair in pairs])[:, tuple(checkpoint["invariant_indices"])]
    features = torch.sign(raw)
    train_catalog = json.loads((ROOT / "research" / "pg37_counterfactual_catalog_v1.json").read_text(encoding="utf-8"))
    train_pairs = pg38._split(pg38._pair_rows(list(train_catalog["samples"]))) ["train"]
    train_raw = torch.stack([pg39._coarse_pair(pair) for pair in train_pairs])[:, tuple(checkpoint["invariant_indices"])]
    train_features = torch.sign(train_raw)
    with torch.inference_mode():
        probability = torch.sigmoid(model(features)).cpu()
    distance = torch.cdist(features, train_features).min(dim=1).values
    accepted = (probability >= 0.60) & (distance <= float(checkpoint["novelty_threshold"]))
    known_positive = known_negative = unknown_positive = unknown_negative = 0
    known_named = unknown_effect = negative_false_accept = unknown_misname = unknown_not_abstain = 0
    routes: dict[str, int] = {}
    semantic_counts: dict[str, dict[str, int]] = {}
    for index, pair in enumerate(pairs):
        candidate = pair["candidate"]
        positive = bool(candidate["oracle_projection"].get("positive", False))
        probe_ref = str(candidate["payload_manifest"].get("probe_ref", ""))
        semantic = probe_ref[len("pg42-semantic-") :] if probe_ref.startswith("pg42-semantic-") else probe_ref
        known = semantic in KNOWN_BINDINGS
        effect = bool(accepted[index])
        route = KNOWN_BINDINGS[semantic] if known and effect else UNKNOWN_ROUTE
        abstain = not known or not effect
        routes[route] = routes.get(route, 0) + 1
        bucket = semantic_counts.setdefault(semantic, {"pairs": 0, "positive": 0, "effect_accepted": 0, "abstain": 0})
        bucket["pairs"] += 1; bucket["positive"] += int(positive); bucket["effect_accepted"] += int(effect); bucket["abstain"] += int(abstain)
        if known:
            known_positive += int(positive); known_negative += int(not positive); known_named += int(positive and effect)
        else:
            unknown_positive += int(positive); unknown_negative += int(not positive); unknown_effect += int(positive and effect); unknown_misname += int(route != UNKNOWN_ROUTE); unknown_not_abstain += int(not abstain)
        negative_false_accept += int((not positive) and effect)
    metrics = {"pair_count": len(pairs), "effect_accepted_count": int(accepted.sum()), "effect_recall_any_family": round(float((torch.tensor([bool(pair["candidate"]["oracle_projection"].get("positive", False)) for pair in pairs]) & accepted).sum()) / max(sum(bool(pair["candidate"]["oracle_projection"].get("positive", False)) for pair in pairs), 1), 6), "known_positive_count": known_positive, "known_negative_count": known_negative, "unknown_positive_count": unknown_positive, "unknown_negative_count": unknown_negative, "known_family_recall": round(known_named / max(known_positive, 1), 6), "unknown_effect_recall": round(unknown_effect / max(unknown_positive, 1), 6), "negative_effect_false_accept_count": negative_false_accept, "negative_effect_false_accept_rate": round(negative_false_accept / max(known_negative + unknown_negative, 1), 6), "unknown_misname_count": unknown_misname, "unknown_not_abstain_count": unknown_not_abstain, "unknown_strict_abstain": unknown_misname == 0 and unknown_not_abstain == 0, "abstain_rate": round(float((~accepted).float().mean()), 6)}
    effect_gate = {"status": "passed" if metrics["effect_recall_any_family"] == 1.0 and metrics["negative_effect_false_accept_count"] == 0 else "blocked", "claim_allowed": metrics["effect_recall_any_family"] == 1.0 and metrics["negative_effect_false_accept_count"] == 0, "reasons": [], "training_allowed": False, "memory_promotion_allowed": False}
    report = {"protocol_id": "sift-pg43-pg42-safe-router-v1", "schema_version": "pg-pk-43-pg42-safe-router-report-v1", "status": "diagnostic_only", "catalog": {"path": str(CATALOG_PATH.relative_to(ROOT)), "sha256": hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest(), "pair_count": len(pairs), "raw_probe_strings_stored": False, "raw_response_bodies_stored": False}, "model": {"checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "checkpoint_sha256": hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest(), "representation": "sign_binned_shape_delta_and_change_bits", "typed_oracle_consumed_by_model": False, "family_agnostic": True}, "routing": {"known_bindings": KNOWN_BINDINGS, "unknown_route": UNKNOWN_ROUTE, "unknown_requires_abstain": True, "metrics": metrics, "route_counts": routes, "semantic_counts": semantic_counts}, "effect_gate": effect_gate, "formal_capability_claim_allowed": False, "promotion": {"status": "quarantined_candidate", "training_allowed": False, "memory_promotion_allowed": False}, "manifest_sha256": hashlib.sha256(json.dumps({"protocol_id": "sift-pg43-pg42-safe-router-v1", "catalog_sha256": hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest(), "checkpoint_sha256": hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-43 on PG-42 safe router", "", "invariant effect head + known ontology route + unknown abstain。", "", "| 指标 | 值 |", "|---|---:|"]
    for key in ("pair_count", "effect_accepted_count", "effect_recall_any_family", "known_family_recall", "unknown_effect_recall", "negative_effect_false_accept_count", "unknown_misname_count", "unknown_not_abstain_count"):
        lines.append(f"| {key} | {metrics[key]} |")
    lines.extend(["", f"effect gate：`{effect_gate['status']}`；formal capability claim=false；训练/记忆不晋升。", ""])
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "metrics": metrics, "effect_gate": effect_gate, "formal_capability_claim_allowed": False, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
