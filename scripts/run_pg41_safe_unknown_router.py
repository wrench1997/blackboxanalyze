"""Run PG-41's fail-closed semantic routing diagnostic.

PG-39 supplies only the family-agnostic effect head.  PG-40 supplies a
language-neutral semantic reference.  This experiment deliberately keeps the
two responsibilities separate: an accepted effect may be named only when its
semantic reference is inside a pre-registered ontology; every unseen
reference is routed to ``unknown_surface`` and abstained.  The evaluator's
typed oracle is used only after inference for scoring, never as a feature.

The script is offline: it reads the already captured loopback catalog and the
quarantined checkpoint.  It emits bounded counts and hashes, not raw probes or
response bodies.
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

CATALOG_PATH = ROOT / "research" / "pg40_semantic_router_catalog_v1.json"
TRAIN_CATALOG_PATH = ROOT / "research" / "pg37_counterfactual_catalog_v1.json"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg39-coarse-delta" / "coarse_delta.pt"
PG38_SCRIPT = ROOT / "scripts" / "train_pg38_effect_pair_candidate.py"
PG39_SCRIPT = ROOT / "scripts" / "train_pg39_coarse_delta_candidate.py"
REPORT_PATH = ROOT / "research" / "pg41_safe_unknown_router_report_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg41_safe_unknown_router_report_v1.md"

EFFECT_THRESHOLD = 0.60
KNOWN_BINDINGS = {
    "markup-context": "xss",
    "operator-context": "injection",
    "auth-boundary": "authentication",
    "subject-boundary": "access_control",
}
UNKNOWN_ROUTE = "unknown_surface"


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper module: {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _semantic_reference(row: dict[str, Any]) -> str:
    manifest = row.get("payload_manifest") or {}
    ref = str(manifest.get("probe_ref") or row.get("semantic_reference") or "")
    prefix = "pg40-semantic-"
    return ref[len(prefix) :] if ref.startswith(prefix) else ref


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _prepare_delta(pg38: Any, pg39: Any, pairs: list[dict[str, Any]], checkpoint: dict[str, Any]) -> torch.Tensor:
    raw = torch.stack([pg39._coarse_pair(pair) for pair in pairs])
    mean = torch.tensor(checkpoint["delta_mean"], dtype=torch.float32)
    std = torch.tensor(checkpoint["delta_std"], dtype=torch.float32).clamp_min(1e-4)
    return (raw - mean) / std


def _training_delta(pg38: Any, pg39: Any, checkpoint: dict[str, Any]) -> torch.Tensor:
    train_catalog = json.loads(TRAIN_CATALOG_PATH.read_text(encoding="utf-8"))
    train_pairs = pg38._split(pg38._pair_rows(list(train_catalog["samples"]))) ["train"]
    return _prepare_delta(pg38, pg39, train_pairs, checkpoint)


def _safe_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if metrics["known_family_recall"] < 1.0:
        reasons.append("known_family_recall_below_1")
    if metrics["negative_effect_false_accept_count"] != 0:
        reasons.append("negative_effect_false_accept")
    if metrics["unknown_misname_count"] != 0:
        reasons.append("unknown_semantic_misnamed")
    if not metrics["unknown_strict_abstain"]:
        reasons.append("unknown_semantic_not_strict_abstain")
    claim_allowed = not reasons
    return {
        "schema_version": "sift-pg41-safe-routing-gate-v1",
        "status": "passed" if claim_allowed else "blocked",
        "claim_allowed": claim_allowed,
        "reasons": reasons,
        "training_allowed": False,
        "memory_promotion_allowed": False,
    }


def main() -> int:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    rows = list(catalog["samples"])
    pg38 = _load_module(PG38_SCRIPT, "pg38_for_pg41")
    pg39 = _load_module(PG39_SCRIPT, "pg39_for_pg41")
    pairs = pg38._pair_rows(rows)
    if not pairs:
        raise RuntimeError("PG-41 catalog has no candidate-control pairs")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = pg39.CoarsePairModel()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    delta = _prepare_delta(pg38, pg39, pairs, checkpoint)
    train_delta = _training_delta(pg38, pg39, checkpoint)
    novelty_threshold = float(checkpoint["novelty_threshold"])
    distance = torch.cdist(delta, train_delta).min(dim=1).values
    with torch.inference_mode():
        # The effect head is independent of the family branch.  The zero
        # candidate is intentional and makes that separation auditable.
        _, effect_logits = model(torch.zeros((len(pairs), 256)), delta)
        effect_probability = torch.sigmoid(effect_logits)
    effect_accepted = (effect_probability >= EFFECT_THRESHOLD) & (distance <= novelty_threshold)

    known_positive = known_negative = unknown_positive = unknown_negative = 0
    known_named_positive = unknown_effect_positive = 0
    negative_effect_false_accept = 0
    unknown_misname = 0
    unknown_not_abstain = 0
    route_counts: dict[str, int] = {}
    semantic_counts: dict[str, dict[str, int]] = {}
    per_method: dict[str, dict[str, int]] = {"GET": {"pairs": 0, "positive": 0, "effect_accepted": 0}, "POST": {"pairs": 0, "positive": 0, "effect_accepted": 0}}
    for index, pair in enumerate(pairs):
        candidate = pair["candidate"]
        positive = bool((candidate.get("oracle_projection") or {}).get("positive", False))
        semantic = _semantic_reference(candidate)
        known = semantic in KNOWN_BINDINGS
        accepted = bool(effect_accepted[index])
        route = KNOWN_BINDINGS[semantic] if known and accepted else UNKNOWN_ROUTE
        abstain = not known or not accepted
        route_counts[route] = route_counts.get(route, 0) + 1
        bucket = semantic_counts.setdefault(semantic, {"pairs": 0, "positive": 0, "effect_accepted": 0, "abstain": 0})
        bucket["pairs"] += 1
        bucket["positive"] += int(positive)
        bucket["effect_accepted"] += int(accepted)
        bucket["abstain"] += int(abstain)
        method = str(candidate.get("method"))
        if method in per_method:
            per_method[method]["pairs"] += 1
            per_method[method]["positive"] += int(positive)
            per_method[method]["effect_accepted"] += int(accepted)
        if known:
            if positive:
                known_positive += 1
                known_named_positive += int(accepted)
            else:
                known_negative += 1
        else:
            if positive:
                unknown_positive += 1
                unknown_effect_positive += int(accepted)
            else:
                unknown_negative += 1
            unknown_misname += int(route != UNKNOWN_ROUTE)
            unknown_not_abstain += int(not abstain)
        negative_effect_false_accept += int((not positive) and accepted)

    metrics = {
        "pair_count": len(pairs),
        "known_pair_count": known_positive + known_negative,
        "known_positive_count": known_positive,
        "known_negative_count": known_negative,
        "unknown_pair_count": unknown_positive + unknown_negative,
        "unknown_positive_count": unknown_positive,
        "unknown_negative_count": unknown_negative,
        "known_family_recall": round(known_named_positive / max(known_positive, 1), 6),
        "unknown_effect_recall": round(unknown_effect_positive / max(unknown_positive, 1), 6),
        "negative_effect_false_accept_count": negative_effect_false_accept,
        "negative_effect_false_accept_rate": round(negative_effect_false_accept / max(known_negative + unknown_negative, 1), 6),
        "unknown_misname_count": unknown_misname,
        "unknown_not_abstain_count": unknown_not_abstain,
        "unknown_strict_abstain": unknown_misname == 0 and unknown_not_abstain == 0,
        "effect_accepted_count": int(effect_accepted.sum()),
        "effect_abstain_count": int((~effect_accepted).sum()),
        "effect_abstain_rate": round(float((~effect_accepted).float().mean()), 6),
        "max_train_distance": round(float(distance.max()), 6),
        "novelty_threshold": novelty_threshold,
    }
    gate = _safe_gate(metrics)
    catalog_sha256 = hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest()
    checkpoint_sha256 = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
    report = {
        "protocol_id": "sift-pg41-safe-unknown-router-v1",
        "schema_version": "pg-pk-41-safe-unknown-router-report-v1",
        "status": "diagnostic_only",
        "catalog": {"path": str(CATALOG_PATH.relative_to(ROOT)), "sha256": catalog_sha256, "pair_count": len(pairs), "raw_probe_strings_stored": False, "raw_response_bodies_stored": False},
        "checkpoint": {"path": str(CHECKPOINT_PATH.relative_to(ROOT)), "sha256": checkpoint_sha256, "effect_head_source": "pg39-coarse-delta", "typed_oracle_consumed_by_model": False},
        "ontology": {"known_bindings": KNOWN_BINDINGS, "unknown_route": UNKNOWN_ROUTE, "unknown_requires_abstain": True, "semantic_reference_contains_family_name": False},
        "routing": {"route_counts": route_counts, "semantic_counts": semantic_counts, "per_method": per_method},
        "metrics": metrics,
        "safe_routing_gate": gate,
        "promotion": {"status": "quarantined_diagnostic", "training_allowed": False, "memory_promotion_allowed": False},
        "manifest_sha256": _canonical_hash({"protocol_id": "sift-pg41-safe-unknown-router-v1", "catalog_sha256": catalog_sha256, "checkpoint_sha256": checkpoint_sha256}),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-41 safe unknown router", "", "效果确认与族命名分离；未见语义统一 `unknown_surface + abstain`。", "", "| 指标 | 值 |", "|---|---:|"]
    for key in ("pair_count", "known_positive_count", "unknown_positive_count", "effect_accepted_count", "known_family_recall", "unknown_effect_recall", "negative_effect_false_accept_count", "unknown_misname_count"):
        lines.append(f"| {key} | {metrics[key]} |")
    lines.extend(["", f"门禁：`{gate['status']}`；claim_allowed=`{gate['claim_allowed']}`；训练/长期记忆均不晋升。", ""])
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "metrics": metrics, "safe_routing_gate": {"status": gate["status"], "claim_allowed": gate["claim_allowed"], "reasons": gate["reasons"]}, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
