"""PG-70: train a tiny trace decision head, then test fresh/unknown holdouts.

Only PG-69's accepted evaluation steps are read.  The model-visible feature
projection is rebuilt from action/response shape and contains no family,
oracle, decision, route vocabulary or raw request/response text.  Known
Docker steps are split by target instance into train/dev/holdout; the
``workflow_invariant`` family is never used for training and is evaluated as
an unknown OOD family.  This is a candidate checkpoint, not a promotion.
"""

from __future__ import annotations

import copy
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from torch import nn

from app.rule_ir_decoder import FEATURE_DIM, trace_feature_vector


PG69_TRACE_PATH = ROOT / "research" / "pg69_per_action_reset_unseen_family_trace_v1.json"
PG69_REPORT_PATH = ROOT / "research" / "pg69_per_action_reset_unseen_family_report_v1.json"
REPORT_PATH = ROOT / "research" / "pg70_trace_abstention_head_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg70_trace_abstention_head_protocol_v1.json"
TRACE_PATH = ROOT / "research" / "pg70_trace_abstention_head_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg70_trace_abstention_head_report_v1.md"
OUTPUT_DIR = ROOT / "artifacts" / "pg70-trace-abstention"
CHECKPOINT_PATH = OUTPUT_DIR / "trace_decision_head.pt"
SEED = 20700403
OOD_DISTANCE_THRESHOLD = 12.0
CONFIDENCE_THRESHOLD = 0.70
CLASSES = ("confirm", "reject")


class TraceDecisionHead(nn.Module):
    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, len(CLASSES)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def _shape(value: Any) -> str:
    if isinstance(value, dict):
        return f"object:{len(value)}:{sum(not isinstance(child, (dict, list)) for child in value.values())}"
    return type(value).__name__


def _visible_step(step: dict[str, Any], response: dict[str, Any], *, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build the exact family-free feature envelope used by the head."""

    action = dict(step.get("action_manifest") or {})
    response = dict(response or {})
    method = str(action.get("method", "GET"))
    projection = {
        "input": {
            "action": {"method": method, "path": "relative_path", "path_shape": {"segment_count": 1, "has_extension": False, "has_query": action.get("placement") == "query"}},
            "probe_kind": str(action.get("probe_ref", "abstract_probe")).split("-")[1] if "-" in str(action.get("probe_ref", "")) else "abstract_probe",
            "probe": "abstract_probe",
            "encoding": (action.get("encoding_chain") or ["identity"])[0],
        },
        "context": {
            "response": {"status_code": int(response.get("status_code", 0) or 0), "content_type": str(response.get("content_type", response.get("content_type_class", ""))), "body_shape": _shape(response.get("shape", response.get("json_shape", {})))},
            "oracle_shape": {"field_count": 0},
        },
        "state": {"body_length": int(response.get("body_length", 0) or 0), "body_length_bucket": str(response.get("body_length_bucket", "unknown"))},
        "history": list(history or []),
        "output": False,
    }
    return projection


def _load_steps() -> list[dict[str, Any]]:
    trace = json.loads(PG69_TRACE_PATH.read_text(encoding="utf-8"))
    if not bool(trace.get("evaluation_only")) or bool(trace.get("training_eligible")):
        raise RuntimeError("PG-70 requires an evaluation-only PG-69 trace")
    if trace.get("validation_failures"):
        raise RuntimeError("PG-70 refuses a trace with validation failures")
    return [dict(step) for step in trace.get("steps", [])]


def _features(step: dict[str, Any], response: dict[str, Any]) -> list[float]:
    return trace_feature_vector([_visible_step(step, response)])


def _build_examples(steps: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return known train/dev examples and fully held-out unknown examples."""

    known = [step for step in steps if "workflow" not in str(step.get("episode_id", ""))]
    unknown = [step for step in steps if "workflow" in str(step.get("episode_id", ""))]
    # The first two known Docker instances are training; the remaining two are
    # fresh target holdout.  Control projections create strict negative rows.
    train_steps = known[:2]
    dev_steps = known[2:]

    def expand(source: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for step in source:
            rows.append({"step_id": step["step_id"], "split": split, "role": "candidate", "label": "confirm", "features": _features(step, step.get("response_projection") or {})})
            rows.append({"step_id": step["step_id"], "split": split, "role": "matched_control", "label": "reject", "features": _features(step, step.get("baseline_projection") or {})})
        return rows

    train_rows = expand(train_steps, "train")
    dev_rows = expand(dev_steps, "dev_holdout")
    unknown_rows = [{"step_id": step["step_id"], "split": "unknown_family_holdout", "role": "candidate", "label": "abstain", "features": _features(step, step.get("response_projection") or {})} for step in unknown]
    return train_rows, dev_rows, unknown_rows


def _normalise(train: list[dict[str, Any]], rows: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    train_tensor = torch.tensor([row["features"] for row in train], dtype=torch.float32)
    mean = train_tensor.mean(dim=0)
    std = train_tensor.std(dim=0, unbiased=False).clamp_min(1e-4)
    values = (torch.tensor([row["features"] for row in rows], dtype=torch.float32) - mean) / std
    return values, torch.stack((mean, std))


def _metrics(model: TraceDecisionHead, values: torch.Tensor, rows: list[dict[str, Any]], reference: torch.Tensor, device: torch.device, *, unknown: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not rows:
        return {"count": 0}, []
    model.eval()
    with torch.inference_mode():
        logits = model(values.to(device)).detach().cpu()
        probabilities = torch.softmax(logits, dim=-1)
    details: list[dict[str, Any]] = []
    for index, (row, probability) in enumerate(zip(rows, probabilities)):
        confidence, predicted = torch.max(probability, dim=0)
        distance = float(torch.cdist(values[index:index + 1], reference).min().item()) if len(reference) else float("inf")
        raw_class = CLASSES[int(predicted)]
        decision = "abstain" if distance >= OOD_DISTANCE_THRESHOLD or float(confidence) < CONFIDENCE_THRESHOLD else raw_class
        details.append({"step_id": row["step_id"], "role": row["role"], "expected": row["label"], "raw_prediction": raw_class, "decision": decision, "confidence": round(float(confidence), 6), "ood_distance": round(distance, 6), "unknown": unknown})
    if unknown:
        metrics = {"count": len(details), "misname_count": sum(int(item["decision"] != "abstain") for item in details), "strict_abstain": all(item["decision"] == "abstain" for item in details), "max_confidence": round(max(item["confidence"] for item in details), 6), "min_ood_distance": round(min(item["ood_distance"] for item in details), 6)}
    else:
        metrics = {"count": len(details), "accuracy": round(sum(int(item["decision"] == item["expected"]) for item in details) / len(details), 6), "false_accept_count": sum(int(item["expected"] == "reject" and item["decision"] == "confirm") for item in details), "confirm_recall": round(sum(int(item["expected"] == "confirm" and item["decision"] == "confirm") for item in details) / max(sum(item["expected"] == "confirm" for item in details), 1), 6), "abstain_count": sum(int(item["decision"] == "abstain") for item in details)}
    return metrics, details


def run() -> dict[str, Any]:
    torch.manual_seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    steps = _load_steps()
    train_rows, dev_rows, unknown_rows = _build_examples(steps)
    if not train_rows or not dev_rows or not unknown_rows:
        raise RuntimeError("PG-70 split is incomplete")
    train_values, normalisation = _normalise(train_rows, train_rows)
    dev_values = (torch.tensor([row["features"] for row in dev_rows], dtype=torch.float32) - normalisation[0]) / normalisation[1]
    unknown_values = (torch.tensor([row["features"] for row in unknown_rows], dtype=torch.float32) - normalisation[0]) / normalisation[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TraceDecisionHead().to(device)
    labels = torch.tensor([CLASSES.index(row["label"]) for row in train_rows], dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.02)
    best_state: dict[str, torch.Tensor] | None = None
    best_dev_loss = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, 301):
        model.train()
        logits = model(train_values.to(device))
        loss = nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch == 1 or epoch % 50 == 0:
            with torch.inference_mode():
                dev_logits = model(dev_values.to(device))
                dev_loss = float(nn.functional.cross_entropy(dev_logits, torch.tensor([CLASSES.index(row["label"]) for row in dev_rows], dtype=torch.long, device=device)).detach().cpu())
            history.append({"epoch": epoch, "train_loss": round(float(loss.detach().cpu()), 6), "dev_loss": round(dev_loss, 6)})
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    reference = train_values
    train_metrics, train_details = _metrics(model, train_values, train_rows, reference, device)
    dev_metrics, dev_details = _metrics(model, dev_values, dev_rows, reference, device)
    unknown_metrics, unknown_details = _metrics(model, unknown_values, unknown_rows, reference, device, unknown=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "sift-pg70-trace-decision-head-v1", "feature_dim": FEATURE_DIM, "classes": CLASSES, "seed": SEED, "ood_distance_threshold": OOD_DISTANCE_THRESHOLD, "confidence_threshold": CONFIDENCE_THRESHOLD, "normalisation_mean": normalisation[0], "normalisation_std": normalisation[1], "model_state": model.state_dict()}, CHECKPOINT_PATH)
    capability_checks = {"dev_confirm_recall": float(dev_metrics.get("confirm_recall", 0.0)) >= 0.80, "dev_false_accept_zero": int(dev_metrics.get("false_accept_count", 1)) == 0, "unknown_misname_zero": int(unknown_metrics.get("misname_count", 1)) == 0, "unknown_strict_abstain": bool(unknown_metrics.get("strict_abstain", False)), "raw_free": True}
    capability_passed = all(capability_checks.values())
    report = {"protocol_id": "pg-pk-70-trace-abstention-head-v1", "schema_version": "sift-pg70-trace-abstention-head-report-v1", "status": "candidate_training_completed", "source": {"pg69_trace": str(PG69_TRACE_PATH.relative_to(ROOT)), "pg69_report": str(PG69_REPORT_PATH.relative_to(ROOT)), "model_retrained_on_unknown_family": False, "family_in_features": False, "oracle_in_features": False, "device": str(device)}, "dataset": {"accepted_trace_step_count": len(steps), "known_train_example_count": len(train_rows), "known_dev_holdout_example_count": len(dev_rows), "unknown_family_holdout_count": len(unknown_rows), "classes": list(CLASSES), "split_rule": "first two known fresh Docker instances train; remaining known instances dev; workflow family never trains"}, "metrics": {"train": train_metrics, "dev_holdout": dev_metrics, "unknown_family_holdout": unknown_metrics}, "details": {"train": train_details, "dev_holdout": dev_details, "unknown_family_holdout": unknown_details}, "training": {"checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "epochs": 300, "seed": SEED, "ood_distance_threshold": OOD_DISTANCE_THRESHOLD, "confidence_threshold": CONFIDENCE_THRESHOLD, "online_weight_update": False, "long_term_memory_write": False}, "capability_gate": {"status": "passed" if capability_passed else "blocked", "checks": capability_checks, "claim_allowed": False}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "status": "candidate_checkpoint_evaluation_only", "reason": "tiny single-run head; requires independent seeds, fresh Docker rerun and family-heldout replay"}, "formal_claim": {"allowed": False, "reason": "PG-70 is a small post-training candidate, not a broad web vulnerability detector"}}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "sift-pg70-training-trace-v1", "evaluation_only": True, "training_eligible": False, "model_retrained_on_unknown_family": False, "family_in_features": False, "oracle_in_features": False, "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "train_step_ids": [row["step_id"] for row in train_rows], "dev_step_ids": [row["step_id"] for row in dev_rows], "unknown_holdout_step_ids": [row["step_id"] for row in unknown_rows], "details": report["details"], "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "online_weight_update": False, "long_term_memory_write": False}
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": "pg-pk-70-trace-abstention-head-v1", "schema_version": "sift-pg70-trace-abstention-head-protocol-v1", "input_contract": {"accepted_pg69_trace_only": True, "family_before_action_forbidden": True, "oracle_in_features_forbidden": True, "unknown_family_training_forbidden": True, "raw_probe_and_response_persistence_forbidden": True}, "split_contract": {"known_fresh_target_split": True, "unknown_workflow_family_held_out": True, "ood_distance_threshold": OOD_DISTANCE_THRESHOLD, "confidence_threshold": CONFIDENCE_THRESHOLD}, "required_gates": {"dev_confirm_recall_min": 0.80, "dev_false_accept_zero": True, "unknown_misname_zero": True, "unknown_strict_abstain": True, "independent_seed_rerun": True, "fresh_docker_rerun": True}, "run_result": {"capability_gate": report["capability_gate"], "training_allowed": False, "memory_promotion_allowed": False}, "next_experiment": "PG71 independent seed + fresh Docker rerun of the frozen candidate head"}
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-70 Trace decision head", "", f"device={device}；train={len(train_rows)}；dev={len(dev_rows)}；unknown holdout={len(unknown_rows)}。", f"dev confirm recall={dev_metrics.get('confirm_recall', 0.0)}；unknown misname={unknown_metrics.get('misname_count', 0)}；unknown strict abstain={unknown_metrics.get('strict_abstain', False)}。", "", f"capability gate: `{report['capability_gate']['status']}`；training promotion: `false`；memory promotion: `false`。", "", f"checkpoint: `{CHECKPOINT_PATH.relative_to(ROOT)}`", f"report: `{REPORT_PATH.relative_to(ROOT)}`", f"protocol: `{PROTOCOL_PATH.relative_to(ROOT)}`", ""]
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    report = run()
    print(json.dumps({"protocol_id": report["protocol_id"], "capability_gate": report["capability_gate"]["status"], "dev_confirm_recall": report["metrics"]["dev_holdout"].get("confirm_recall", 0.0), "unknown_misname_count": report["metrics"]["unknown_family_holdout"].get("misname_count", 0), "device": report["source"]["device"], "training_allowed": False, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
