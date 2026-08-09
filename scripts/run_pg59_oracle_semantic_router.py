"""PG-59 typed-oracle semantic routing ablation.

PG-54 demonstrated that a generic effect contract cannot identify a family.
This experiment adds only bounded, post-probe oracle semantics (DOM/AST/
boundary/redirect/canary/negative), never raw probes, response bodies, family
names, source identifiers, or evaluator-only field names.  The router is
trained on PG-53 and evaluated on the independent PG-42 catalog; an unknown
template oracle modality is fail-closed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
PG53_PATH = ROOT / "research" / "pg53_cross_source_typed_replay_report_v1.json"
PG42_PATH = ROOT / "research" / "pg42_independent_semantic_catalog_v1.json"
REPORT_PATH = ROOT / "research" / "pg59_oracle_semantic_router_report_v1.json"
TRACE_PATH = ROOT / "research" / "pg59_oracle_semantic_router_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg59_oracle_semantic_router_report_v1.md"
OUTPUT_DIR = ROOT / "artifacts" / "pg59-oracle-semantic-router"
CHECKPOINT_PATH = OUTPUT_DIR / "router.pt"
SEED = 20260803
EPOCHS = 120

FAMILIES = [
    "xss",
    "injection",
    "authentication",
    "access_control",
    "logic",
    "url_redirect",
    "input_validation",
    "command_injection",
]
UNKNOWN_FAMILY = "template_injection"
MODALITIES = ["DOM", "AST", "AUTH_BOUNDARY", "ACCESS_BOUNDARY", "LOGIC", "REDIRECT", "VALIDATION", "CANARY", "OTHER", "NEGATIVE"]
KNOWN_MODALITIES = set(MODALITIES[:8])


class OracleSemanticRouter(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(96, 64),
            nn.GELU(),
        )
        self.family_head = nn.Linear(64, len(FAMILIES))
        self.emit_head = nn.Linear(64, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(features)
        return self.family_head(hidden), self.emit_head(hidden).squeeze(-1)


def _bucket(value: Any, scale: float = 8.0) -> float:
    try:
        return min(max(float(value) / scale, 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.0


def _modality(value: Any) -> str:
    text = str(value or "").casefold()
    if "negative" in text:
        return "NEGATIVE"
    if "dom" in text or "markup" in text:
        return "DOM"
    if "ast" in text or "sql" in text or "interpreter" in text:
        return "AST"
    if "authentication" in text or "auth_" in text or "auth-boundary" in text:
        return "AUTH_BOUNDARY"
    if "authoriz" in text or "subject" in text:
        return "ACCESS_BOUNDARY"
    if "logic" in text or "invariant" in text or "state-boundary" in text:
        return "LOGIC"
    if "redirect" in text or "destination" in text or "origin" in text:
        return "REDIRECT"
    if "validation" in text or "scalar-boundary" in text:
        return "VALIDATION"
    if "canary" in text or "local-callback" in text or "command" in text:
        return "CANARY"
    return "OTHER"


def _one_hot(value: str, values: list[str]) -> list[float]:
    return [float(value == candidate) for candidate in values]


def _extract(row: dict[str, Any], source: str) -> tuple[list[float], dict[str, Any]]:
    if source == "pg53":
        candidate = row.get("candidate") or {}
        oracle = candidate.get("oracle") or {}
        response = candidate.get("response") or {}
        shape = response.get("shape") or {}
        surface = candidate.get("surface_observation") or {}
        geometry = candidate.get("generic_effect_geometry") or {}
        descriptor = row.get("payload_manifest") or {}
        phase = str(oracle.get("stage") or "confirm")
        method = str(row.get("method") or "GET").upper()
        placement = str(descriptor.get("placement") or "query").lower()
        status = str(response.get("status_class") or "unknown").upper()
        shape_kind = str(shape.get("kind") or "unknown").lower()
        modality = _modality(oracle.get("modality"))
        positive = bool(oracle.get("positive"))
        authority = bool(oracle.get("positive_authority"))
        bounded = bool((oracle.get("safety") or {}).get("state_mutated") is False)
        readonly = bool((oracle.get("safety") or {}).get("external_network") is False)
        effect = str(oracle.get("confirmed_effect") or "none")
        family = str(row.get("family") or "unknown")
        evidence_hash_present = bool(row.get("evidence_sha256"))
    else:
        oracle = row.get("oracle_projection") or {}
        response = row.get("response_projection") or {}
        shape = response.get("shape") or {}
        surface = {}
        geometry = {}
        descriptor = row.get("payload_manifest") or {}
        phase = str(row.get("phase") or "confirm")
        method = str(row.get("method") or "GET").upper()
        placement = str(descriptor.get("placement") or "query").lower()
        status = str(response.get("status_class") or "unknown").upper()
        shape_kind = str(shape.get("kind") or "unknown").lower()
        modality = _modality(oracle.get("modality"))
        positive = bool(oracle.get("positive"))
        authority = bool(oracle.get("positive_authority"))
        proof = (oracle.get("signals") or {}).get("proof") or {}
        bounded = bool(proof.get("bounded", True))
        readonly = bool(proof.get("read_only", True))
        effect = str(oracle.get("confirmed_effect") or "none")
        family = str(row.get("family") or "unknown")
        evidence_hash_present = bool((row.get("evidence") or {}).get("evidence_hash"))
    vector = (
        _one_hot(modality, MODALITIES)
        + _one_hot(method, ["GET", "POST"])
        + _one_hot(placement, ["query", "body", "path"])
        + _one_hot(phase, ["screen", "confirm", "error", "timeout"])
        + _one_hot(status, ["2XX", "3XX", "4XX", "5XX", "UNKNOWN"])
        + _one_hot(shape_kind, ["object", "array", "unknown"])
        + [_bucket(shape.get("key_count")), _bucket(shape.get("scalar_count")), _bucket(shape.get("array_count")), _bucket(surface.get("true_boolean_count")), _bucket(surface.get("nonzero_numeric_count")), _bucket(geometry.get("leaf_count"))]
        + [float(positive), float(authority), float(bounded), float(readonly), float(evidence_hash_present), _bucket(len(effect), 32.0)]
    )
    metadata = {
        "sample_id": str(row.get("sample_id") or ""),
        "family": family,
        "positive": positive,
        "authority": authority,
        "modality": modality,
        "known_modality": modality in KNOWN_MODALITIES,
        "method": method,
        "source": source,
        "dataset_role": str(row.get("dataset_role") or "train"),
        "raw_probe_stored": False,
        "raw_response_stored": False,
    }
    return vector, metadata


def _accepted(scores: dict[str, Any], *, emit_threshold: float, family_threshold: float, margin_threshold: float, require_known_modality: bool) -> bool:
    return bool(
        scores["emit_probability"] >= emit_threshold
        and scores["family_confidence"] >= family_threshold
        and scores["family_margin"] >= margin_threshold
        and scores["predicted_family"] != "ordinary_response"
        and (not require_known_modality or scores["known_modality"])
    )


def _metrics(rows: list[dict[str, Any]], scores: list[dict[str, Any]], *, thresholds: dict[str, float], require_known_modality: bool) -> dict[str, Any]:
    accepted = [_accepted(score, **thresholds, require_known_modality=require_known_modality) for score in scores]
    known = [i for i, meta in enumerate(rows) if meta["positive"] and meta["family"] in FAMILIES]
    unknown = [i for i, meta in enumerate(rows) if meta["positive"] and meta["family"] == UNKNOWN_FAMILY]
    negatives = [i for i, meta in enumerate(rows) if not meta["positive"] or meta["family"] == "ordinary_response"]
    hits = [i for i in known if accepted[i] and scores[i]["predicted_family"] == rows[i]["family"]]
    return {
        "count": len(rows),
        "emitted_count": sum(accepted),
        "known_positive_count": len(known),
        "known_family_hits": len(hits),
        "known_family_recall": round(len(hits) / max(len(known), 1), 6),
        "known_wrong_family_count": sum(int(accepted[i] and scores[i]["predicted_family"] != rows[i]["family"]) for i in known),
        "unknown_positive_count": len(unknown),
        "unknown_misname_count": sum(int(accepted[i]) for i in unknown),
        "unknown_strict_abstain": all(not accepted[i] for i in unknown),
        "negative_false_accept_count": sum(int(accepted[i]) for i in negatives),
        "abstain_rate": round(1.0 - sum(accepted) / max(len(rows), 1), 6),
    }


def _score(model: OracleSemanticRouter, features: torch.Tensor, metadata: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    model.eval()
    with torch.inference_mode():
        family_logits, emit_logits = model(features.to(device))
        family_probabilities = family_logits.softmax(-1).cpu()
        emit_probabilities = emit_logits.sigmoid().cpu()
    results = []
    for index, meta in enumerate(metadata):
        row = family_probabilities[index]
        values, indices = row.sort(descending=True)
        predicted = FAMILIES[int(indices[0])]
        results.append({
            "predicted_family": predicted,
            "family_confidence": float(values[0]),
            "family_margin": float(values[0] - values[1]),
            "emit_probability": float(emit_probabilities[index]),
            "known_modality": bool(meta["known_modality"]),
            "modality": meta["modality"],
        })
    return results


def _calibrate(dev_meta: list[dict[str, Any]], dev_scores: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, Any]]:
    candidates = {
        "emit_threshold": sorted({score["emit_probability"] for score in dev_scores} | {1.0}, reverse=True),
        "family_threshold": sorted({score["family_confidence"] for score in dev_scores} | {1.0}, reverse=True),
        "margin_threshold": sorted({score["family_margin"] for score in dev_scores} | {1.0}, reverse=True),
    }
    # Bound a potentially large Cartesian grid by quantiles while retaining
    # all values when the dev set is small.
    for key, values in candidates.items():
        if len(values) > 24:
            positions = [round(i * (len(values) - 1) / 23) for i in range(24)]
            candidates[key] = sorted({values[position] for position in positions} | {1.0}, reverse=True)
    best = None
    for emit_threshold in candidates["emit_threshold"]:
        for family_threshold in candidates["family_threshold"]:
            for margin_threshold in candidates["margin_threshold"]:
                thresholds = {"emit_threshold": emit_threshold, "family_threshold": family_threshold, "margin_threshold": margin_threshold}
                metrics = _metrics(dev_meta, dev_scores, thresholds=thresholds, require_known_modality=True)
                if metrics["negative_false_accept_count"] or metrics["known_wrong_family_count"] or metrics["unknown_misname_count"]:
                    continue
                if best is None or metrics["known_family_hits"] > best[1]["known_family_hits"]:
                    best = (thresholds, metrics)
    if best is None:
        return {"emit_threshold": 1.0, "family_threshold": 1.0, "margin_threshold": 1.0}, {"known_family_hits": 0, "known_wrong_family_count": 0, "unknown_misname_count": 0, "negative_false_accept_count": 0}
    return best


def main() -> int:
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    pg53_rows = json.loads(PG53_PATH.read_text(encoding="utf-8"))["rows"]
    pg42_rows = json.loads(PG42_PATH.read_text(encoding="utf-8"))["samples"]
    train_raw = [(_extract(row, "pg53")) for row in pg53_rows]
    dev_raw = [(_extract(row, "pg42")) for row in pg42_rows if row.get("dataset_role") == "dev"]
    holdout_raw = [(_extract(row, "pg42")) for row in pg42_rows if row.get("dataset_role") in {"ood_source", "family_holdout", "negative_control"}]
    train_features = torch.tensor([item[0] for item in train_raw], dtype=torch.float32)
    dev_features = torch.tensor([item[0] for item in dev_raw], dtype=torch.float32)
    holdout_features = torch.tensor([item[0] for item in holdout_raw], dtype=torch.float32)
    train_meta = [item[1] for item in train_raw]
    dev_meta = [item[1] for item in dev_raw]
    holdout_meta = [item[1] for item in holdout_raw]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OracleSemanticRouter(train_features.shape[1]).to(device)
    positive_indices = [i for i, meta in enumerate(train_meta) if meta["positive"] and meta["family"] in FAMILIES]
    positive_features = train_features[positive_indices].to(device)
    positive_labels = torch.tensor([FAMILIES.index(train_meta[i]["family"]) for i in positive_indices], dtype=torch.long, device=device)
    emit_labels = torch.tensor([float(meta["positive"] and meta["authority"] and meta["known_modality"]) for meta in train_meta], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.015)
    family_loss = nn.CrossEntropyLoss(label_smoothing=0.02)
    emit_loss = nn.BCEWithLogitsLoss()
    best_state = None
    best_dev_loss = float("inf")
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        # Family head trains only on confirmed known positives; emit head sees
        # matched negative controls as well.
        optimizer.zero_grad(set_to_none=True)
        family_logits, emit_logits = model(train_features.to(device))
        loss = family_loss(family_logits[positive_indices], positive_labels) + emit_loss(emit_logits, emit_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            dev_logits, dev_emit = model(dev_features.to(device))
            known_dev = [i for i, meta in enumerate(dev_meta) if meta["positive"] and meta["family"] in FAMILIES]
            dev_loss = float(family_loss(dev_logits[known_dev], torch.tensor([FAMILIES.index(dev_meta[i]["family"]) for i in known_dev], device=device)) + emit_loss(dev_emit, torch.tensor([float(meta["positive"] and meta["authority"] and meta["known_modality"]) for meta in dev_meta], device=device)))
        history.append({"epoch": epoch, "train_loss": round(float(loss.detach().cpu()), 6), "dev_loss": round(dev_loss, 6)})
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    dev_scores = _score(model, dev_features, dev_meta, device)
    holdout_scores = _score(model, holdout_features, holdout_meta, device)
    raw_thresholds = {"emit_threshold": 0.5, "family_threshold": 0.5, "margin_threshold": 0.05}
    calibrated_thresholds, calibration = _calibrate(dev_meta, dev_scores)
    dev_raw = _metrics(dev_meta, dev_scores, thresholds=raw_thresholds, require_known_modality=False)
    holdout_raw = _metrics(holdout_meta, holdout_scores, thresholds=raw_thresholds, require_known_modality=False)
    dev_semantic = _metrics(dev_meta, dev_scores, thresholds=raw_thresholds, require_known_modality=True)
    holdout_semantic = _metrics(holdout_meta, holdout_scores, thresholds=raw_thresholds, require_known_modality=True)
    dev_calibrated = _metrics(dev_meta, dev_scores, thresholds=calibrated_thresholds, require_known_modality=True)
    holdout_calibrated = _metrics(holdout_meta, holdout_scores, thresholds=calibrated_thresholds, require_known_modality=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": "pg59-oracle-semantic-router-checkpoint-v1",
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "input_dim": int(train_features.shape[1]),
        "families": FAMILIES,
        "modalities": MODALITIES,
        "family_name_in_features": False,
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "long_term_memory_write": False,
        "device_at_training": str(device),
    }, CHECKPOINT_PATH)
    trace = {
        "schema_version": "pg59-oracle-semantic-router-trace-v1",
        "evaluation_only": True,
        "rows": [
            {
                "sample_id": meta["sample_id"],
                "dataset_role": meta["dataset_role"],
                "modality": meta["modality"],
                "known_modality": meta["known_modality"],
                "predicted_family": score["predicted_family"],
                "emit_probability_bucket": "high" if score["emit_probability"] >= 0.8 else "low",
                "raw_probe_stored": False,
                "raw_response_stored": False,
            }
            for meta, score in zip(holdout_meta, holdout_scores)
        ],
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "long_term_memory_write": False,
    }
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "protocol_id": "pg-pk-59-oracle-semantic-router-v1",
        "schema_version": "pg59-oracle-semantic-router-report-v1",
        "device": str(device),
        "input_dim": int(train_features.shape[1]),
        "modalities": MODALITIES,
        "training_contract": {
            "training_source": "PG-53 typed oracle",
            "evaluation_source": "PG-42 independent semantic catalog",
            "family_name_in_features": False,
            "raw_probe_in_features": False,
            "raw_response_body_in_features": False,
            "oracle_semantics_are_post_probe_evidence": True,
            "unknown_modality_fail_closed": True,
        },
        "split_counts": {"train": len(train_meta), "dev": len(dev_meta), "holdout": len(holdout_meta)},
        "metrics": {
            "dev_raw": dev_raw,
            "holdout_raw": holdout_raw,
            "dev_semantic_gate": dev_semantic,
            "holdout_semantic_gate": holdout_semantic,
            "dev_calibrated": dev_calibrated,
            "holdout_calibrated": holdout_calibrated,
        },
        "thresholds": {"raw": raw_thresholds, "calibrated": calibrated_thresholds, "calibration_source": "PG-59 PG-42 dev only", "calibration_metrics": calibration},
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest(),
        "trace": str(TRACE_PATH.relative_to(ROOT)),
        "history_tail": history[-5:],
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "formal_capability_claim_allowed": False,
            "status": "quarantined_typed_oracle_router_ablation",
            "reason": "PG-59 tests post-probe typed semantic routing; it does not prove pre-oracle discovery or authorize memory promotion.",
        },
    }
    report["report_sha256"] = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(
        "\n".join([
            "# PG-59 typed-oracle semantic router",
            "",
            f"设备：`{device}`；train/dev/holdout：`{len(train_meta)}/{len(dev_meta)}/{len(holdout_meta)}`；输入维度：`{train_features.shape[1]}`。",
            f"独立 PG-42 盲测（semantic gate）known recall：`{holdout_semantic['known_family_recall']:.3f}`；wrong family：`{holdout_semantic['known_wrong_family_count']}`；unknown misname：`{holdout_semantic['unknown_misname_count']}`；negative false accept：`{holdout_semantic['negative_false_accept_count']}`。",
            f"未知/负例门控后的弃权率：`{holdout_semantic['abstain_rate']:.3f}`。",
            "该实验只验证“typed oracle 已提供语义后能否路由 Rule IR”，不等同于黑盒探测发现能力；仍不进入长期记忆。",
            "",
        ]) + "",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
