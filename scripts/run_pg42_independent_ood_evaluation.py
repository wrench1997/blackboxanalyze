"""Evaluate the quarantined effect head on PG-42's independent target.

No training occurs here.  The PG-39 effect checkpoint consumes only a bounded
candidate/control coarse delta.  Semantic references are used after inference
by a fail-closed router: four pre-registered references may be named; every
other reference is ``unknown_surface`` and abstains.
"""

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
REPORT_PATH = ROOT / "research" / "pg42_independent_ood_evaluation_report_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg42_independent_ood_evaluation_report_v1.md"
EFFECT_THRESHOLD = 0.60
KNOWN_BINDINGS = {"markup-context": "xss", "operator-context": "injection", "auth-boundary": "authentication", "subject-boundary": "access_control"}
UNKNOWN_ROUTE = "unknown_surface"


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _semantic(pair: dict[str, Any]) -> str:
    ref = str((pair["candidate"].get("payload_manifest") or {}).get("probe_ref", ""))
    prefix = "pg42-semantic-"
    return ref[len(prefix) :] if ref.startswith(prefix) else ref


def _delta(pg39: Any, pairs: list[dict[str, Any]], checkpoint: dict[str, Any]) -> torch.Tensor:
    raw = torch.stack([pg39._coarse_pair(pair) for pair in pairs])
    mean = torch.tensor(checkpoint["delta_mean"], dtype=torch.float32)
    std = torch.tensor(checkpoint["delta_std"], dtype=torch.float32).clamp_min(1e-4)
    return (raw - mean) / std


def _train_delta(pg38: Any, pg39: Any, checkpoint: dict[str, Any]) -> torch.Tensor:
    train_catalog = json.loads(TRAIN_CATALOG_PATH.read_text(encoding="utf-8"))
    pairs = pg38._split(pg38._pair_rows(list(train_catalog["samples"]))) ["train"]
    return _delta(pg39, pairs, checkpoint)


def _metric_rows(model: Any, pairs: list[dict[str, Any]], delta: torch.Tensor, train_delta: torch.Tensor, novelty_threshold: float) -> dict[str, Any]:
    with torch.inference_mode():
        _, logits = model(torch.zeros((len(pairs), 256)), delta)
        probability = torch.sigmoid(logits).cpu()
    distance = torch.cdist(delta.cpu(), train_delta.cpu()).min(dim=1).values
    accepted = (probability >= EFFECT_THRESHOLD) & (distance <= novelty_threshold)
    positive = torch.tensor([bool(pair["candidate"]["oracle_projection"].get("positive", False)) for pair in pairs])
    return {"count": len(pairs), "positive_count": int(positive.sum()), "negative_count": int((~positive).sum()), "effect_accepted_count": int(accepted.sum()), "effect_recall_any_family": round(float((positive & accepted).sum()) / max(int(positive.sum()), 1), 6), "effect_false_positive_count": int((~positive & accepted).sum()), "effect_false_positive_rate": round(float((~positive & accepted).sum()) / max(int((~positive).sum()), 1), 6), "abstain_rate": round(float((~accepted).float().mean()), 6), "mean_effect_probability": round(float(probability.mean()), 6)}


def _routing_metrics(pairs: list[dict[str, Any]], accepted: torch.Tensor) -> tuple[dict[str, Any], dict[str, dict[str, int]], dict[str, int]]:
    known_pos = known_neg = unknown_pos = unknown_neg = 0
    known_named = unknown_effect = negative_false_accept = unknown_misname = unknown_not_abstain = 0
    semantic_counts: dict[str, dict[str, int]] = {}
    routes: dict[str, int] = {}
    for index, pair in enumerate(pairs):
        candidate = pair["candidate"]
        positive = bool(candidate["oracle_projection"].get("positive", False))
        semantic = _semantic(pair)
        known = semantic in KNOWN_BINDINGS
        effect = bool(accepted[index])
        route = KNOWN_BINDINGS[semantic] if known and effect else UNKNOWN_ROUTE
        abstain = not known or not effect
        routes[route] = routes.get(route, 0) + 1
        bucket = semantic_counts.setdefault(semantic, {"pairs": 0, "positive": 0, "effect_accepted": 0, "abstain": 0})
        bucket["pairs"] += 1; bucket["positive"] += int(positive); bucket["effect_accepted"] += int(effect); bucket["abstain"] += int(abstain)
        if known:
            known_pos += int(positive); known_neg += int(not positive); known_named += int(positive and effect)
        else:
            unknown_pos += int(positive); unknown_neg += int(not positive); unknown_effect += int(positive and effect); unknown_misname += int(route != UNKNOWN_ROUTE); unknown_not_abstain += int(not abstain)
        negative_false_accept += int((not positive) and effect)
    metrics = {"known_positive_count": known_pos, "known_negative_count": known_neg, "unknown_positive_count": unknown_pos, "unknown_negative_count": unknown_neg, "known_family_recall": round(known_named / max(known_pos, 1), 6), "unknown_effect_recall": round(unknown_effect / max(unknown_pos, 1), 6), "negative_effect_false_accept_count": negative_false_accept, "negative_effect_false_accept_rate": round(negative_false_accept / max(known_neg + unknown_neg, 1), 6), "unknown_misname_count": unknown_misname, "unknown_not_abstain_count": unknown_not_abstain, "unknown_strict_abstain": unknown_misname == 0 and unknown_not_abstain == 0}
    return metrics, semantic_counts, routes


def _safe_gate(metrics: dict[str, Any], split_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    if metrics["negative_effect_false_accept_count"] != 0:
        reasons.append("negative_effect_false_accept")
    if metrics["unknown_misname_count"] != 0 or metrics["unknown_not_abstain_count"] != 0:
        reasons.append("unknown_semantic_not_fail_closed")
    if split_metrics["implementation_holdout"]["effect_recall_any_family"] < 1.0:
        reasons.append("independent_implementation_effect_recall_below_1")
    if split_metrics["family_holdout"]["effect_recall_any_family"] < 1.0:
        reasons.append("new_family_effect_recall_below_1")
    return {"schema_version": "sift-pg42-independent-ood-gate-v1", "status": "passed" if not reasons else "blocked", "claim_allowed": not reasons, "reasons": reasons, "training_allowed": False, "memory_promotion_allowed": False}


def main() -> int:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    rows = list(catalog["samples"])
    pg38 = _load_module(PG38_SCRIPT, "pg38_for_pg42_eval")
    pg39 = _load_module(PG39_SCRIPT, "pg39_for_pg42_eval")
    pairs = pg38._pair_rows(rows)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = pg39.CoarsePairModel(); model.load_state_dict(checkpoint["model_state"]); model.eval()
    delta = _delta(pg39, pairs, checkpoint)
    train_delta = _train_delta(pg38, pg39, checkpoint)
    novelty_threshold = float(checkpoint["novelty_threshold"])
    with torch.inference_mode():
        _, logits = model(torch.zeros((len(pairs), 256)), delta)
        probability = torch.sigmoid(logits).cpu()
    distance = torch.cdist(delta.cpu(), train_delta.cpu()).min(dim=1).values
    accepted = (probability >= EFFECT_THRESHOLD) & (distance <= novelty_threshold)
    split_pairs = {
        "train": [pair for pair in pairs if pair["candidate"].get("dataset_role") == "train"],
        "dev": [pair for pair in pairs if pair["candidate"].get("dataset_role") == "dev"],
        "implementation_holdout": [pair for pair in pairs if pair["candidate"].get("dataset_role") == "ood_source"],
        "family_holdout": [pair for pair in pairs if pair["candidate"].get("dataset_role") == "family_holdout"],
        "negative_control": [pair for pair in pairs if pair["candidate"].get("dataset_role") == "negative_control"],
    }
    pair_index = {id(pair): index for index, pair in enumerate(pairs)}
    split_metrics: dict[str, dict[str, Any]] = {}
    for name, split in split_pairs.items():
        indices = [pair_index[id(pair)] for pair in split]
        split_metrics[name] = _metric_rows(model, split, delta[indices], train_delta, novelty_threshold)
    routing, semantic_counts, routes = _routing_metrics(pairs, accepted)
    gate = _safe_gate(routing, split_metrics)
    catalog_sha256 = hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest()
    checkpoint_sha256 = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
    report = {
        "protocol_id": "sift-pg42-independent-semantic-ood-v1",
        "schema_version": "pg-pk-42-independent-semantic-ood-report-v1",
        "status": "diagnostic_only",
        "catalog": {"path": str(CATALOG_PATH.relative_to(ROOT)), "sha256": catalog_sha256, "pair_count": len(pairs), "raw_probe_strings_stored": False, "raw_response_bodies_stored": False},
        "model": {"checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "checkpoint_sha256": checkpoint_sha256, "effect_head_source": "pg39-coarse-delta", "effect_head_input": "bounded candidate-control coarse delta", "typed_oracle_consumed_by_model": False, "family_consumed_by_model": False, "semantic_reference_consumed_by_effect_head": False},
        "splits": split_metrics,
        "routing": {"known_bindings": KNOWN_BINDINGS, "unknown_route": UNKNOWN_ROUTE, "unknown_requires_abstain": True, "route_counts": routes, "semantic_counts": semantic_counts, "metrics": routing},
        "safe_routing_gate": gate,
        "promotion": {"status": "quarantined_independent_ood", "training_allowed": False, "memory_promotion_allowed": False},
        "manifest_sha256": hashlib.sha256(json.dumps({"protocol_id": "sift-pg42-independent-semantic-ood-v1", "catalog_sha256": catalog_sha256, "checkpoint_sha256": checkpoint_sha256}, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-42 independent semantic OOD", "", "PG-39 effect head 只读取 bounded candidate/control coarse delta；语义未知时路由为 `unknown_surface + abstain`。", "", "| split | effect recall | effect FPR | positives |", "|---|---:|---:|---:|"]
    for name, item in split_metrics.items():
        lines.append(f"| {name} | {item['effect_recall_any_family']:.2f} | {item['effect_false_positive_rate']:.2f} | {item['positive_count']} |")
    lines.extend(["", f"安全门禁：`{gate['status']}`；claim_allowed=`{gate['claim_allowed']}`；训练/记忆不晋升。", ""])
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "splits": {name: {key: item[key] for key in ("effect_recall_any_family", "effect_false_positive_rate", "positive_count")} for name, item in split_metrics.items()}, "routing": routing, "safe_routing_gate": {"status": gate["status"], "claim_allowed": gate["claim_allowed"], "reasons": gate["reasons"]}, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
