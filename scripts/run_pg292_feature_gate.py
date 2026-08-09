"""Remote A800 PG-292 feature-compositional send gate."""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESEARCH = ROOT / "research"
BASE = RESEARCH / "pg285_payload_grounding_dataset_v1.json"
BASE_AUDIT = RESEARCH / "pg285_payload_grounding_dataset_audit_v1.json"
AUGMENTED = RESEARCH / "pg289_safe_abstain_dataset_v1.json"
AUGMENTED_AUDIT = RESEARCH / "pg289_safe_abstain_dataset_audit_v1.json"
HARD = RESEARCH / "pg285_payload_grounding_hard_negative_v1.json"
REMOTE_PROBE = RESEARCH / "pg280_remote_docker_probe_v2.json"
OUT_DIR = ROOT / "artifacts" / "pg292-feature-gate"
CHECKPOINT = OUT_DIR / "pg292_feature_gate.pt"
REPORT = RESEARCH / "pg292_feature_gate_report_v1.json"
TRACE = RESEARCH / "pg292_feature_gate_trace_v1.json"
PROTOCOL = RESEARCH / "pg292_feature_gate_protocol_v1.json"
MARKDOWN = RESEARCH / "pg292_feature_gate_report_v1.md"
SEEDS = (29201, 29202, 29203)
WEIGHTS = {"plain": 1.0, "guarded": 2.0, "strict": 4.0}
THRESHOLDS = (0.5, 0.8, 0.95, 0.99, 0.999, 0.9999)


def _sha(value: Any) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate(values: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({key for value in values for key, item in value.items() if isinstance(item, (int, float)) and not isinstance(item, bool) and item is not None})
    return {key: {"mean": round(sum(float(value[key]) for value in values) / len(values), 6), "min": round(min(float(value[key]) for value in values), 6), "max": round(max(float(value[key]) for value in values), 6)} for key in keys}


def main() -> None:
    if os.environ.get("PG292_REMOTE_RUN") != "1":
        raise RuntimeError("PG-292 is remote-only; set PG292_REMOTE_RUN=1 on authorized executor")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    from app.pg292_feature_gate import build_feature_vocab, predict, train_gate

    base = _load(BASE)
    base_audit = _load(BASE_AUDIT)
    augmented = _load(AUGMENTED)
    augmented_audit = _load(AUGMENTED_AUDIT)
    hard_data = _load(HARD)
    remote_probe = _load(REMOTE_PROBE)
    if base_audit.get("status") != "passed" or augmented_audit.get("status") != "passed":
        raise RuntimeError("PG-292 source audits must pass")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assignment = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"), "visible_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0, "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None, "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    if device.type != "cuda" or assignment["cuda_visible_devices"] != "0" or assignment["visible_device_count"] != 1 or "A800" not in str(assignment["device_name"]):
        raise RuntimeError(f"PG-292 requires remote A800 GPU0, got {assignment}")
    base_rows = list(base.get("records") or [])
    base_train = [row for row in base_rows if row.get("split") == "train"]
    route_rows = [row for row in base_rows if row.get("split") == "route_dev"]
    family_rows = [row for row in base_rows if row.get("split") == "family_holdout"]
    counterfactual_rows = list(augmented.get("records") or [])
    train_rows = base_train + counterfactual_rows
    hard_rows = list(hard_data.get("records") or [])
    vocab = build_feature_vocab(train_rows)
    started = time.perf_counter()
    per_seed: dict[str, list[dict[str, Any]]] = {name: [] for name in WEIGHTS}
    checkpoints: dict[str, Any] = {}
    for seed in SEEDS:
        random.seed(seed)
        for variant, weight in WEIGHTS.items():
            model = train_gate(train_rows, vocab, device, seed, negative_weight=weight, epochs=240, hidden_dim=96)
            sections = {name: {str(threshold): predict(model, rows, vocab, device, threshold=threshold) for threshold in THRESHOLDS} for name, rows in (("train", train_rows), ("route_dev", route_rows), ("family_holdout", family_rows), ("hard_negative", hard_rows))}
            per_seed[variant].append({"seed": seed, **sections})
            if seed == SEEDS[-1]:
                checkpoints[variant] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    aggregated = {variant: {section: {threshold: _aggregate([value[section][threshold] for value in values]) for threshold in [str(item) for item in THRESHOLDS]} for section in ("train", "route_dev", "family_holdout", "hard_negative")} for variant, values in per_seed.items()}
    candidates = []
    for variant in WEIGHTS:
        for threshold in [str(item) for item in THRESHOLDS]:
            route = aggregated[variant]["route_dev"][threshold]
            family = aggregated[variant]["family_holdout"][threshold]
            hard = aggregated[variant]["hard_negative"][threshold]
            candidates.append({"variant": variant, "threshold": float(threshold), "route_positive_recall_min": float(route.get("positive_recall", {}).get("min", 0.0)), "family_positive_recall_min": float(family.get("positive_recall", {}).get("min", 0.0)), "hard_negative_false_allow_max": int(hard.get("false_allow_count", {}).get("max", 0)), "hard_negative_safe_reject_min": float(hard.get("safe_reject_rate", {}).get("min", 0.0))})
    selected = max(candidates, key=lambda item: (item["hard_negative_false_allow_max"] == 0, item["route_positive_recall_min"], item["family_positive_recall_min"], item["hard_negative_safe_reject_min"], -item["threshold"]))
    checks = {"base_audit_pass": base_audit.get("status") == "passed", "augmentation_audit_pass": augmented_audit.get("status") == "passed", "mixed_train_present": len(base_train) > 0 and len(counterfactual_rows) > 0, "a800_gpu0_only": assignment["cuda_visible_devices"] == "0" and assignment["visible_device_count"] == 1 and assignment["current_device"] == 0 and "A800" in assignment["device_name"], "hard_negative_evaluated": len(hard_rows) > 0, "remote_docker_honest": remote_probe.get("status") != "available" or remote_probe.get("real_application_gold_rows", 0) == 0, "promotion_blocked": True}
    report = {"protocol_id": "pg292-feature-gate-v1", "schema_version": "pg292-feature-gate-report-v1", "status": "completed_remote_pg292_feature_gate", "source": {"base_dataset": str(BASE.relative_to(ROOT).as_posix()), "base_dataset_sha256": base["dataset_sha256"], "augmentation_dataset": str(AUGMENTED.relative_to(ROOT).as_posix()), "augmentation_dataset_sha256": augmented["dataset_sha256"], "hard_negative_dataset": str(HARD.relative_to(ROOT).as_posix()), "hard_negative_sha256": hard_data["dataset_sha256"], "remote_probe": str(REMOTE_PROBE.relative_to(ROOT).as_posix()), "remote_docker_status": remote_probe.get("status", "unknown"), "real_application_gold_rows": int(remote_probe.get("real_application_gold_rows", 0) or 0), "remote_host": "112.111.7.91:60228", "literal_payload_in_context": False, "live_send": False}, "device": assignment, "split": {"base_train": len(base_train), "counterfactual_train": len(counterfactual_rows), "mixed_train": len(train_rows), "route_dev": len(route_rows), "family_holdout": len(family_rows), "hard_negative": len(hard_rows), "seeds": list(SEEDS)}, "vocabulary": {"feature_count": len(vocab)}, "aggregated": aggregated, "per_seed": per_seed, "selection": {"selected_variant": selected["variant"], "selected_threshold": selected["threshold"], "candidates": candidates, "rule": "zero hard-negative false-allow first; then route/family positive recall; then hard-negative rejection"}, "engineering_gate": {"status": "passed" if all(checks.values()) else "blocked", "checks": checks, "claim_allowed": False}, "scientific_gate": {"status": "blocked", "checks": {"live_target_evaluator": False, "real_application_gold": False, "fresh_target_replay": False, "literal_payload_success": False, "feature_gate_evaluated": True}, "reasons": ["feature gate trained on counterfactual/template rows", "remote Docker/evaluator unavailable", "no fresh target replay"], "claim_allowed": False}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "reason": "PG-292 feature gate is a safety component, not a vulnerability detector"}, "formal_conclusion": "PG-292 tests key/value compositional context features for unknown evaluator states; it must be combined with a typed oracle and fresh replay before any claim.", "elapsed_seconds": round(time.perf_counter() - started, 3)}
    report["report_sha256"] = _sha(report)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg292-feature-gate-checkpoint-v1", "assignment": assignment, "vocab": vocab, "selected_variant": selected["variant"], "selected_threshold": selected["threshold"], "state": checkpoints[selected["variant"]]}, CHECKPOINT)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg292-feature-gate-trace-v1", "report_sha256": report["report_sha256"], "selected_variant": selected["variant"], "selected_threshold": selected["threshold"], "training_eligible": False, "memory_write": False, "literal_payload_in_context": False, "live_send": False}
    trace["trace_sha256"] = _sha(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg292-feature-gate-protocol-v1", "remote_a800_gpu0_only": True, "key_value_features": True, "threshold_sweep": list(THRESHOLDS), "hard_negative_evaluation_only": True, "literal_payload_generation": False, "live_send": False, "report_sha256": report["report_sha256"], "next_experiment": "PG-292-live: attach feature gate to constrained Rule-IR decoder with fresh typed oracle."}
    protocol["protocol_sha256"] = _sha(protocol)
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN.write_text("\n".join(["# PG-292 key/value feature gate", "", f"engineering_gate={report['engineering_gate']['status']}", "scientific_gate=blocked", f"selected={selected['variant']}@{selected['threshold']}", f"hard false allow={selected['hard_negative_false_allow_max']}", f"route recall={selected['route_positive_recall_min']}", "context feature gate; literal payload/live send=false", ""]), encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": assignment, "selection": report["selection"], "engineering_gate": report["engineering_gate"], "scientific_gate": report["scientific_gate"], "report": str(REPORT.relative_to(ROOT).as_posix())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
