"""PG-75: candidate triplet-delta head with source/seed isolation.

PG-74 supplies neutral, typed-negative and typed-positive observations.  The
candidate learns only projection deltas; family labels, oracle fields, route
words and raw probe/response values are excluded.  PG-69/PG-72 and the
unknown workflow family are evaluation-only.  This is not a promotion run.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

PG74_TRACE_PATH = ROOT / "research" / "pg74_causal_triplet_collector_trace_v1.json"
PG74_REPORT_PATH = ROOT / "research" / "pg74_causal_triplet_collector_report_v1.json"
PG69_TRACE_PATH = ROOT / "research" / "pg69_per_action_reset_unseen_family_trace_v1.json"
PG72_TRACE_PATH = ROOT / "research" / "pg72_independent_seed_fresh_docker_matrix_trace_v1.json"
PG76_TRACE_PATH = ROOT / "research" / "pg76_independent_unknown_triplet_trace_v1.json"
PG76_REPORT_PATH = ROOT / "research" / "pg76_independent_unknown_triplet_report_v1.json"
PG71_PATH = ROOT / "scripts" / "train_pg71_trace_abstention_head_v2.py"
REPORT_PATH = ROOT / "research" / "pg75_triplet_context_delta_ablation_report_v4.json"
PROTOCOL_PATH = ROOT / "research" / "pg75_triplet_context_delta_ablation_protocol_v4.json"
TRACE_PATH = ROOT / "research" / "pg75_triplet_context_delta_ablation_trace_v4.json"
MARKDOWN_PATH = ROOT / "research" / "pg75_triplet_context_delta_ablation_report_v4.md"
OUTPUT_DIR = ROOT / "artifacts" / "pg75-triplet-delta"
CHECKPOINT_PATH = OUTPUT_DIR / "trace_triplet_context_head_v4.pt"
SEED = 20750403
TRAIN_SEEDS = (74101, 74102)
DEV_SEEDS = (74103,)
STD_FLOOR = 0.25
CLIP = 6.0
OOD_DISTANCE_THRESHOLD = 18.0
OOD_CALIBRATION_MARGIN = 0.25
CONFIDENCE_THRESHOLD = 0.70
CLASSES = ("confirm", "reject")
CONTEXT_FEATURE_DIM = 256
MODEL_FEATURE_DIM = CONTEXT_FEATURE_DIM * 2


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _delta_features(v2: Any, step: dict[str, Any], projection: dict[str, Any], neutral: dict[str, Any]) -> list[float]:
    current = torch.tensor(v2._features(step, projection), dtype=torch.float32)
    base = torch.tensor(v2._features(step, neutral), dtype=torch.float32)
    return (current - base).tolist()


def _context_delta_features(v2: Any, step: dict[str, Any], projection: dict[str, Any], neutral: dict[str, Any]) -> list[float]:
    """Expose neutral surface context plus causal probe delta, never labels."""

    return list(v2._features(step, neutral)) + _delta_features(v2, step, projection, neutral)


def _build_rows(v2: Any, steps: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    for step in steps:
        seed = int(step.get("sampling_seed", -1))
        target = train if seed in TRAIN_SEEDS else dev if seed in DEV_SEEDS else None
        if target is None:
            continue
        neutral = dict(step.get("neutral_projection") or {})
        if not neutral or "negative_probe_projection" not in step:
            raise RuntimeError("PG-75 requires the PG-74 causal triplet fields")
        target.extend([
            {"step_id": step["step_id"], "seed": seed, "role": "positive_probe", "label": "confirm", "features": _context_delta_features(v2, step, dict(step["response_projection"]), neutral)},
            {"step_id": step["step_id"], "seed": seed, "role": "negative_probe", "label": "reject", "features": _context_delta_features(v2, step, dict(step["negative_probe_projection"]), neutral)},
        ])
    return train, dev


def _normalise(train_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train = torch.tensor([row["features"] for row in train_rows], dtype=torch.float32)
    mean = train.mean(dim=0)
    std = train.std(dim=0, unbiased=False).clamp_min(STD_FLOOR)
    values = ((torch.tensor([row["features"] for row in rows], dtype=torch.float32) - mean) / std).clamp(-CLIP, CLIP)
    return values, mean, std


def _evaluate(model: nn.Module, values: torch.Tensor, rows: list[dict[str, Any]], reference: torch.Tensor, device: torch.device, *, unknown: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not rows:
        return {"count": 0}, []
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(values.to(device)).detach().cpu(), dim=-1)
    details: list[dict[str, Any]] = []
    for index, (row, probability) in enumerate(zip(rows, probabilities)):
        confidence, predicted = torch.max(probability, dim=0)
        distance = float(torch.cdist(values[index:index + 1], reference).min().item()) if len(reference) else float("inf")
        raw = CLASSES[int(predicted)]
        decision = "abstain" if distance >= OOD_DISTANCE_THRESHOLD or float(confidence) < CONFIDENCE_THRESHOLD else raw
        details.append({"step_id": row["step_id"], "seed": row.get("seed"), "role": row["role"], "expected": row["label"], "raw_prediction": raw, "decision": decision, "confidence": round(float(confidence), 6), "ood_distance": round(distance, 6), "unknown": unknown})
    if unknown:
        return {"count": len(details), "misname_count": sum(int(item["decision"] != "abstain") for item in details), "strict_abstain": all(item["decision"] == "abstain" for item in details), "min_ood_distance": round(min(item["ood_distance"] for item in details), 6), "max_confidence": round(max(item["confidence"] for item in details), 6)}, details
    return {"count": len(details), "accuracy": round(sum(int(item["decision"] == item["expected"]) for item in details) / max(len(details), 1), 6), "false_accept_count": sum(int(item["expected"] == "reject" and item["decision"] == "confirm") for item in details), "confirm_recall": round(sum(int(item["expected"] == "confirm" and item["decision"] == "confirm") for item in details) / max(sum(item["expected"] == "confirm" for item in details), 1), 6), "abstain_count": sum(int(item["decision"] == "abstain") for item in details)}, details


def _calibrate_ood_threshold(train_values: torch.Tensor, dev_values: torch.Tensor) -> float:
    """Pre-registered bounded margin above known train/dev nearest distances."""

    if len(train_values) > 1:
        loo = torch.cdist(train_values, train_values)
        loo.fill_diagonal_(float("inf"))
        train_max = float(loo.min(dim=1).values.max().item())
    else:
        train_max = 0.0
    dev_max = float(torch.cdist(dev_values, train_values).min(dim=1).values.max().item()) if len(dev_values) else train_max
    return round(max(train_max, dev_max) + OOD_CALIBRATION_MARGIN, 6)


def _unknown_rows(v2: Any, trace: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in trace.get("steps", []):
        if "workflow" not in str(step.get("episode_id", "")):
            continue
        # PG-69 has no triplet; use its family-free candidate-minus-control
        # delta solely as an unknown-family abstention probe.
        neutral = dict(step.get("baseline_projection") or {})
        rows.append({"step_id": step["step_id"], "seed": step.get("sampling_seed"), "role": "unknown_family", "label": "abstain", "features": _context_delta_features(v2, step, dict(step.get("response_projection") or {}), neutral)})
    return rows


def _independent_unknown_triplet_rows(v2: Any, trace: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in trace.get("steps", []):
        neutral = dict(step.get("neutral_projection") or {})
        if not neutral or "response_projection" not in step:
            continue
        rows.append({"step_id": step["step_id"], "seed": step.get("sampling_seed"), "role": "independent_unknown_triplet", "label": "abstain", "features": _context_delta_features(v2, step, dict(step["response_projection"]), neutral)})
    return rows


def run() -> dict[str, Any]:
    v2 = _load(PG71_PATH, "pg75_pg71_v2")
    pg74_report = _read(PG74_REPORT_PATH)
    pg74_trace = _read(PG74_TRACE_PATH)
    if pg74_report["hard_gate"]["status"] != "passed" or pg74_report["metrics"]["triplet_case_count"] != 21:
        raise RuntimeError("PG-75 requires the complete PG-74 triplet collection gate")
    if pg74_trace.get("validation_failures") or not pg74_trace.get("evaluation_only") or pg74_trace.get("training_eligible"):
        raise RuntimeError("PG-75 requires a valid evaluation-only PG-74 trace")
    train_rows, dev_rows = _build_rows(v2, [dict(step) for step in pg74_trace["steps"]])
    if not train_rows or not dev_rows:
        raise RuntimeError("PG-75 seed split is incomplete")
    train_values, mean, std = _normalise(train_rows, train_rows)
    dev_values, _, _ = _normalise(train_rows, dev_rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    base = _load(PG71_PATH, "pg75_head_base")
    model = base._load_pg70().TraceDecisionHead(feature_dim=MODEL_FEATURE_DIM).to(device)
    labels = torch.tensor([CLASSES.index(row["label"]) for row in train_rows], dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=0.02)
    best_state: dict[str, torch.Tensor] | None = None
    best_dev = float("inf")
    history: list[dict[str, Any]] = []
    dev_labels = torch.tensor([CLASSES.index(row["label"]) for row in dev_rows], dtype=torch.long, device=device)
    for epoch in range(1, 501):
        model.train()
        loss = nn.functional.cross_entropy(model(train_values.to(device)), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch == 1 or epoch % 50 == 0:
            with torch.inference_mode():
                dev_loss = float(nn.functional.cross_entropy(model(dev_values.to(device)), dev_labels).detach().cpu())
            history.append({"epoch": epoch, "train_loss": round(float(loss.detach().cpu()), 6), "dev_loss": round(dev_loss, 6)})
            if dev_loss < best_dev:
                best_dev = dev_loss
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    reference = train_values
    global OOD_DISTANCE_THRESHOLD
    OOD_DISTANCE_THRESHOLD = _calibrate_ood_threshold(train_values, dev_values)
    train_metrics, train_details = _evaluate(model, train_values, train_rows, reference, device)
    dev_metrics, dev_details = _evaluate(model, dev_values, dev_rows, reference, device)
    unknown_rows = _unknown_rows(v2, _read(PG69_TRACE_PATH))
    unknown_values, _, _ = _normalise(train_rows, unknown_rows)
    unknown_metrics, unknown_details = _evaluate(model, unknown_values, unknown_rows, reference, device, unknown=True)
    pg76_report = _read(PG76_REPORT_PATH)
    pg76_trace = _read(PG76_TRACE_PATH)
    if pg76_report["hard_gate"]["status"] != "passed" or not bool(pg76_report["source"].get("family_outside_training_registry")):
        raise RuntimeError("PG-75 requires the PG-76 independent unknown triplet evaluation gate")
    independent_unknown_rows = _independent_unknown_triplet_rows(v2, pg76_trace)
    independent_unknown_values, _, _ = _normalise(train_rows, independent_unknown_rows)
    independent_unknown_metrics, independent_unknown_details = _evaluate(model, independent_unknown_values, independent_unknown_rows, reference, device, unknown=True)
    external_steps = [dict(step) for step in _read(PG72_TRACE_PATH).get("steps", [])]
    external_rows = [{"step_id": step["step_id"], "seed": step.get("sampling_seed"), "role": "external_known", "label": "confirm", "features": _context_delta_features(v2, step, dict(step.get("response_projection") or {}), dict(step.get("baseline_projection") or {}))} for step in external_steps]
    external_values, _, _ = _normalise(train_rows, external_rows)
    external_metrics, external_details = _evaluate(model, external_values, external_rows, reference, device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "sift-pg75-triplet-context-delta-head-v4", "feature_dim": MODEL_FEATURE_DIM, "classes": CLASSES, "seed": SEED, "train_seeds": TRAIN_SEEDS, "dev_seeds": DEV_SEEDS, "std_floor": STD_FLOOR, "clip": CLIP, "ood_distance_threshold": OOD_DISTANCE_THRESHOLD, "confidence_threshold": CONFIDENCE_THRESHOLD, "normalisation_mean": mean, "normalisation_std": std, "model_state": model.state_dict()}, CHECKPOINT_PATH)
    checks = {"dev_confirm_recall": float(dev_metrics.get("confirm_recall", 0.0)) >= 0.80, "dev_false_accept_zero": int(dev_metrics.get("false_accept_count", 1)) == 0, "legacy_unknown_misname_zero": int(unknown_metrics.get("misname_count", 1)) == 0, "legacy_unknown_strict_abstain": bool(unknown_metrics.get("strict_abstain", False)), "independent_triplet_unknown_misname_zero": int(independent_unknown_metrics.get("misname_count", 1)) == 0, "independent_triplet_unknown_strict_abstain": bool(independent_unknown_metrics.get("strict_abstain", False)), "independent_source_attested": bool(pg76_report["source"].get("fixture_source_sha256"))}
    report = {"protocol_id": "pg-pk-75-triplet-context-delta-ablation-v4", "schema_version": "sift-pg75-triplet-context-delta-ablation-report-v4", "status": "candidate_training_completed", "source": {"pg74_trace": str(PG74_TRACE_PATH.relative_to(ROOT)), "pg69_unknown_trace": str(PG69_TRACE_PATH.relative_to(ROOT)), "pg72_external_trace": str(PG72_TRACE_PATH.relative_to(ROOT)), "pg76_independent_unknown_trace": str(PG76_TRACE_PATH.relative_to(ROOT)), "model_retrained_on_unknown_family": False, "family_in_features": False, "oracle_in_features": False, "negative_oracle_in_features": False, "device": str(device)}, "dataset": {"pg74_triplet_count": 21, "train_example_count": len(train_rows), "dev_example_count": len(dev_rows), "legacy_unknown_family_holdout_count": len(unknown_rows), "independent_unknown_triplet_holdout_count": len(independent_unknown_rows), "external_known_replay_count": len(external_rows), "train_seeds": list(TRAIN_SEEDS), "dev_seeds": list(DEV_SEEDS), "triplet_delta_features": True, "neutral_surface_context_features": True, "feature_dim": MODEL_FEATURE_DIM, "external_pg72_schema_mismatch_diagnostic_only": True}, "metrics": {"train": train_metrics, "dev_holdout": dev_metrics, "legacy_unknown_family_holdout": unknown_metrics, "independent_unknown_triplet_holdout": independent_unknown_metrics, "external_pg72_known_schema_mismatch_diagnostic": external_metrics}, "details": {"train": train_details, "dev_holdout": dev_details, "legacy_unknown_family_holdout": unknown_details, "independent_unknown_triplet_holdout": independent_unknown_details, "external_pg72_known_schema_mismatch_diagnostic": external_details}, "training": {"checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "epochs": 500, "seed": SEED, "ood_distance_threshold": OOD_DISTANCE_THRESHOLD, "ood_calibration_margin": OOD_CALIBRATION_MARGIN, "online_weight_update": False, "long_term_memory_write": False}, "capability_gate": {"status": "passed" if all(checks.values()) else "blocked", "checks": checks, "claim_allowed": False}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "status": "candidate_checkpoint_evaluation_only", "reason": "capability gates pass only for this candidate; require more family/source replay and human/Codex audit before promotion"}, "formal_claim": {"allowed": False, "reason": "PG-75 is a triplet-context-delta candidate, not a broad web vulnerability detector"}}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps({"schema_version": "sift-pg75-triplet-delta-training-trace-v1", "evaluation_only": True, "training_eligible": False, "model_retrained_on_unknown_family": False, "family_in_features": False, "oracle_in_features": False, "negative_oracle_in_features": False, "train_step_ids": [row["step_id"] for row in train_rows], "dev_step_ids": [row["step_id"] for row in dev_rows], "details": report["details"], "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "online_weight_update": False, "long_term_memory_write": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps({"protocol_id": "pg-pk-75-triplet-context-delta-ablation-v4", "schema_version": "sift-pg75-triplet-context-delta-ablation-protocol-v4", "input_contract": {"accepted_pg74_triplet_trace_only": True, "family_and_oracle_features_forbidden": True, "negative_oracle_features_forbidden": True, "unknown_family_training_forbidden": True, "raw_persistence_forbidden": True}, "split_contract": {"train_seeds": list(TRAIN_SEEDS), "dev_seeds": list(DEV_SEEDS), "legacy_pg69_unknown_diagnostic_only": True, "pg72_external_known_schema_mismatch_diagnostic_only": True, "pg76_independent_unknown_triplet_required": True, "triplet_delta_features": True, "neutral_surface_context_features": True, "ood_threshold_calibrated_from_known_train_dev": True, "all_abstain_not_capability_success": True}, "required_gates": {"dev_confirm_recall_min": 0.80, "dev_false_accept_zero": True, "legacy_unknown_misname_zero": True, "legacy_unknown_strict_abstain": True, "independent_triplet_unknown_holdout": True, "independent_source_attested": True, "independent_implementation_review": True, "family_holdout_review": True}, "run_result": {"capability_gate": report["capability_gate"], "training_allowed": False, "memory_promotion_allowed": False}, "next_experiment": "PG77 independent implementation and multi-family triplet replay before any promotion"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("# PG-75 Triplet context + delta ablation\n\n" + f"train={len(train_rows)}；dev={len(dev_rows)}；legacy unknown={len(unknown_rows)}；independent unknown triplet={len(independent_unknown_rows)}；external known diagnostic={len(external_rows)}；device={device}；OOD threshold={OOD_DISTANCE_THRESHOLD}。\n\ndev recall={dev_metrics.get('confirm_recall', 0.0)}；legacy unknown misname={unknown_metrics.get('misname_count', 0)}；independent unknown misname={independent_unknown_metrics.get('misname_count', 0)}。\n\n能力门：`{report['capability_gate']['status']}`；training/memory promotion=`false`。\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps({"protocol_id": report["protocol_id"], "capability_gate": report["capability_gate"]["status"], "dev_confirm_recall": report["metrics"]["dev_holdout"].get("confirm_recall", 0.0), "legacy_unknown_misname_count": report["metrics"]["legacy_unknown_family_holdout"].get("misname_count", 0), "ood_distance_threshold": report["training"]["ood_distance_threshold"], "device": report["source"]["device"], "training_allowed": False}, ensure_ascii=False, indent=2))
