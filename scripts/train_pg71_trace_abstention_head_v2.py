"""PG-71 v2: bounded-shape projection repair for the PG-70 head.

The split and labels are frozen from PG-69.  Only the model-visible feature
projection and normalization are changed in response to the PG-71 audit;
family/oracle values remain evaluator-only and the unknown family stays out of
training.  This is still a candidate checkpoint, never a promotion.
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

from app.rule_ir_decoder import FEATURE_DIM, trace_feature_vector  # noqa: E402


PG69_TRACE_PATH = ROOT / "research" / "pg69_per_action_reset_unseen_family_trace_v1.json"
AUDIT_PATH = ROOT / "research" / "pg71_trace_feature_drift_audit_report_v1.json"
REPORT_PATH = ROOT / "research" / "pg71_trace_abstention_head_v2_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg71_trace_abstention_head_v2_protocol_v1.json"
TRACE_PATH = ROOT / "research" / "pg71_trace_abstention_head_v2_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg71_trace_abstention_head_v2_report_v1.md"
OUTPUT_DIR = ROOT / "artifacts" / "pg71-trace-abstention-v2"
CHECKPOINT_PATH = OUTPUT_DIR / "trace_decision_head_v2.pt"
SEED = 20710403
STD_FLOOR = 0.25
CLIP = 6.0
OOD_DISTANCE_THRESHOLD = 18.0
CONFIDENCE_THRESHOLD = 0.70
CLASSES = ("confirm", "reject")


def _load_pg70() -> Any:
    spec = importlib.util.spec_from_file_location("pg70_v2_base", ROOT / "scripts" / "train_pg70_trace_abstention_head.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-70 base head")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _shape(value: Any) -> str:
    if isinstance(value, dict):
        return f"object:{len(value)}:{sum(not isinstance(child, (dict, list)) for child in value.values())}"
    return type(value).__name__


def _visible_v2(step: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    action = dict(step.get("action_manifest") or {})
    response = dict(response or {})
    # All values below are bounded shape classes or clipped scalars.  Hashes,
    # evaluator labels, route words and response bodies are deliberately absent.
    shape_class = ":".join(str(response.get(key, "unknown")) for key in ("status_class", "content_type", "content_type_class", "body_length_bucket", "html_tag_count", "form_count", "input_count", "script_count", "result_row_count", "marker_reflected", "marker_count", "has_location", "location_origin"))
    tag_count = int(response.get("html_tag_count", 0) or 0)
    form_count = int(response.get("form_count", 0) or 0)
    row_count = int(response.get("result_row_count", 0) or 0)
    dom_shape = dict(response.get("dom_shape") or {})
    dom_node_count = int(dom_shape.get("node_count", 0) or 0)
    dom_svg_count = int(dom_shape.get("svg_count", 0) or 0)
    dom_event_count = int(dom_shape.get("event_handler_attribute_count", 0) or 0)
    dom_form_count = int(dom_shape.get("form_count", 0) or 0)
    dom_input_count = int(dom_shape.get("input_count", 0) or 0)
    dom_script_count = int(dom_shape.get("script_count", 0) or 0)
    json_shape = dict(response.get("shape") or response.get("json_shape") or {})
    json_kind = str(json_shape.get("kind", json_shape.get("type", "none")))
    json_key_count = int(json_shape.get("key_count", 0) or 0)
    json_scalar_count = int(json_shape.get("scalar_count", 0) or 0)
    json_array_count = int(json_shape.get("array_count", 0) or 0)
    header_names = response.get("header_names") if isinstance(response.get("header_names"), list) else []
    header_count = min(len(header_names), 32)
    shape_bins = {
        f"tag_mod2_{tag_count % 2}": True,
        f"tag_mod4_{tag_count % 4}": True,
        f"form_mod2_{form_count % 2}": True,
        f"rows_mod4_{row_count % 4}": True,
        f"marker_{int(bool(response.get('marker_reflected', False)))}": True,
        f"location_{str(response.get('location_origin', 'none'))}": True,
        f"dom_svg_mod2_{dom_svg_count % 2}": True,
        f"dom_event_mod2_{dom_event_count % 2}": True,
        f"dom_node_mod4_{dom_node_count % 4}": True,
        f"json_keys_mod4_{json_key_count % 4}": True,
        f"json_scalars_mod4_{json_scalar_count % 4}": True,
        f"headers_mod4_{header_count % 4}": True,
    }
    return {
        "input": {"action": {"method": str(action.get("method", "GET")), "path": "relative_path", "path_shape": {"segment_count": 1, "has_extension": False, "has_query": action.get("placement") == "query"}}, "probe_kind": str(action.get("probe_ref", "abstract_probe")).split("-")[1] if "-" in str(action.get("probe_ref", "")) else "abstract_probe", "probe": "abstract_probe", "encoding": (action.get("encoding_chain") or ["identity"])[0]},
        "context": {"response": {"status_code": int(response.get("status_code", 0) or 0), "content_type": str(response.get("content_type", response.get("content_type_class", ""))), "body_shape": shape_class, "observable_shape": {"tag_count": tag_count, "form_count": form_count, "script_count": int(response.get("script_count", 0) or 0), "result_row_count": row_count, "marker_reflected": bool(response.get("marker_reflected", False)), "has_location": bool(response.get("has_location", False)), "location_origin": str(response.get("location_origin", "none")), "dom_node_count": dom_node_count, "dom_svg_count": dom_svg_count, "dom_event_handler_count": dom_event_count, "dom_form_count": dom_form_count, "dom_input_count": dom_input_count, "dom_script_count": dom_script_count, "json_kind": json_kind, "json_key_count": json_key_count, "json_scalar_count": json_scalar_count, "json_array_count": json_array_count, "header_count": header_count}, "shape_bins": shape_bins}, "oracle_shape": {"field_count": 0}},
        "state": {"body_length_bucket": str(response.get("body_length_bucket", "unknown")), "bounded_observation": True},
        "history": [],
        "output": False,
    }


def _features(step: dict[str, Any], response: dict[str, Any]) -> list[float]:
    return trace_feature_vector([_visible_v2(step, response)])


def _pair_features(step: dict[str, Any]) -> list[float]:
    """Encode the causal candidate-minus-baseline observation."""

    candidate = torch.tensor(_features(step, step.get("response_projection") or {}), dtype=torch.float32)
    baseline = torch.tensor(_features(step, step.get("baseline_projection") or {}), dtype=torch.float32)
    return (candidate - baseline).tolist()


def _build_examples(steps: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    known = [step for step in steps if "workflow" not in str(step.get("episode_id", ""))]
    unknown = [step for step in steps if "workflow" in str(step.get("episode_id", ""))]
    # Keep one markup and one SQL case in training; hold out the POST markup
    # and redirect case.  This is a pre-registered surface-diverse split.
    train_steps, dev_steps = [known[0], known[2]], [known[1], known[3]]

    def expand(source: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for step in source:
            rows.append({"step_id": step["step_id"], "split": split, "role": "candidate", "label": "confirm", "features": _pair_features(step)})
            rows.append({"step_id": step["step_id"], "split": split, "role": "matched_control", "label": "reject", "features": [0.0] * FEATURE_DIM})
        return rows

    return expand(train_steps, "train"), expand(dev_steps, "dev_holdout"), [{"step_id": step["step_id"], "split": "unknown_family_holdout", "role": "candidate", "label": "abstain", "features": _pair_features(step)} for step in unknown]


def _normalise(train_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train = torch.tensor([row["features"] for row in train_rows], dtype=torch.float32)
    mean = train.mean(dim=0)
    std = train.std(dim=0, unbiased=False).clamp_min(STD_FLOOR)
    values = ((torch.tensor([row["features"] for row in rows], dtype=torch.float32) - mean) / std).clamp(-CLIP, CLIP)
    return values, mean, std


def _evaluate(model: nn.Module, values: torch.Tensor, rows: list[dict[str, Any]], reference: torch.Tensor, device: torch.device, *, unknown: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(values.to(device)).detach().cpu(), dim=-1)
    details: list[dict[str, Any]] = []
    for index, (row, probability) in enumerate(zip(rows, probabilities)):
        confidence, predicted = torch.max(probability, dim=0)
        distance = float(torch.cdist(values[index:index + 1], reference).min().item()) if len(reference) else float("inf")
        raw = CLASSES[int(predicted)]
        decision = "abstain" if distance >= OOD_DISTANCE_THRESHOLD or float(confidence) < CONFIDENCE_THRESHOLD else raw
        details.append({"step_id": row["step_id"], "role": row["role"], "expected": row["label"], "raw_prediction": raw, "decision": decision, "confidence": round(float(confidence), 6), "ood_distance": round(distance, 6), "unknown": unknown})
    if unknown:
        return {"count": len(details), "misname_count": sum(int(item["decision"] != "abstain") for item in details), "strict_abstain": all(item["decision"] == "abstain" for item in details), "min_ood_distance": round(min((item["ood_distance"] for item in details), default=0.0), 6), "max_confidence": round(max((item["confidence"] for item in details), default=0.0), 6)}, details
    return {"count": len(details), "accuracy": round(sum(int(item["decision"] == item["expected"]) for item in details) / max(len(details), 1), 6), "false_accept_count": sum(int(item["expected"] == "reject" and item["decision"] == "confirm") for item in details), "confirm_recall": round(sum(int(item["expected"] == "confirm" and item["decision"] == "confirm") for item in details) / max(sum(item["expected"] == "confirm" for item in details), 1), 6), "abstain_count": sum(int(item["decision"] == "abstain") for item in details)}, details


def run() -> dict[str, Any]:
    base = _load_pg70()
    steps = [dict(step) for step in _read(PG69_TRACE_PATH).get("steps", [])]
    audit = _read(AUDIT_PATH)
    if int(audit["metrics"]["legacy_candidate_control_duplicate_label_conflict_count"]) <= 0:
        raise RuntimeError("PG-71 v2 requires an observed legacy feature collision")
    train_rows, dev_rows, unknown_rows = _build_examples(steps)
    train_values, mean, std = _normalise(train_rows, train_rows)
    dev_values, _, _ = _normalise(train_rows, dev_rows)
    unknown_values, _, _ = _normalise(train_rows, unknown_rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = base.TraceDecisionHead().to(device)
    labels = torch.tensor([("confirm", "reject").index(row["label"]) for row in train_rows], dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.02)
    best_state: dict[str, torch.Tensor] | None = None
    best_dev = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, 301):
        model.train()
        loss = nn.functional.cross_entropy(model(train_values.to(device)), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch == 1 or epoch % 50 == 0:
            with torch.inference_mode():
                dev_loss = float(nn.functional.cross_entropy(model(dev_values.to(device)), torch.tensor([("confirm", "reject").index(row["label"]) for row in dev_rows], dtype=torch.long, device=device)).detach().cpu())
            history.append({"epoch": epoch, "train_loss": round(float(loss.detach().cpu()), 6), "dev_loss": round(dev_loss, 6)})
            if dev_loss < best_dev:
                best_dev = dev_loss
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    train_metrics, train_details = _evaluate(model, train_values, train_rows, train_values, device)
    dev_metrics, dev_details = _evaluate(model, dev_values, dev_rows, train_values, device)
    unknown_metrics, unknown_details = _evaluate(model, unknown_values, unknown_rows, train_values, device, unknown=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "sift-pg71-trace-decision-head-v2", "feature_dim": FEATURE_DIM, "classes": ("confirm", "reject"), "seed": SEED, "std_floor": STD_FLOOR, "clip": CLIP, "ood_distance_threshold": OOD_DISTANCE_THRESHOLD, "confidence_threshold": CONFIDENCE_THRESHOLD, "normalisation_mean": mean, "normalisation_std": std, "model_state": model.state_dict()}, CHECKPOINT_PATH)
    checks = {"dev_confirm_recall": float(dev_metrics.get("confirm_recall", 0.0)) >= 0.80, "dev_false_accept_zero": int(dev_metrics.get("false_accept_count", 1)) == 0, "unknown_misname_zero": int(unknown_metrics.get("misname_count", 1)) == 0, "unknown_strict_abstain": bool(unknown_metrics.get("strict_abstain", False))}
    report = {"protocol_id": "pg-pk-71-trace-abstention-head-v2", "schema_version": "sift-pg71-trace-abstention-head-v2-report-v1", "status": "candidate_training_completed", "source": {"pg69_trace": str(PG69_TRACE_PATH.relative_to(ROOT)), "feature_audit": str(AUDIT_PATH.relative_to(ROOT)), "model_retrained_on_unknown_family": False, "family_in_features": False, "oracle_in_features": False, "device": str(device)}, "feature_repair": {"bounded_shape_scalars": ["html_tag_count", "form_count", "script_count", "result_row_count", "marker_reflected", "has_location", "location_origin"], "std_floor": STD_FLOOR, "clip": CLIP}, "dataset": {"known_train_example_count": len(train_rows), "known_dev_holdout_example_count": len(dev_rows), "unknown_family_holdout_count": len(unknown_rows), "split_frozen_from_pg69": True}, "metrics": {"train": train_metrics, "dev_holdout": dev_metrics, "unknown_family_holdout": unknown_metrics}, "details": {"train": train_details, "dev_holdout": dev_details, "unknown_family_holdout": unknown_details}, "training": {"checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "epochs": 300, "seed": SEED, "online_weight_update": False, "long_term_memory_write": False}, "capability_gate": {"status": "passed" if all(checks.values()) else "blocked", "checks": checks, "claim_allowed": False}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "status": "candidate_checkpoint_evaluation_only", "reason": "requires independent seed and fresh Docker rerun"}, "formal_claim": {"allowed": False, "reason": "feature repair candidate only"}}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps({"schema_version": "sift-pg71-trace-abstention-head-v2-trace-v1", "evaluation_only": True, "training_eligible": False, "model_retrained_on_unknown_family": False, "family_in_features": False, "oracle_in_features": False, "details": report["details"], "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "online_weight_update": False, "long_term_memory_write": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps({"protocol_id": "pg-pk-71-trace-abstention-head-v2", "schema_version": "sift-pg71-trace-abstention-head-v2-protocol-v1", "input_contract": {"accepted_pg69_trace_only": True, "unknown_family_training_forbidden": True, "family_and_oracle_features_forbidden": True, "raw_persistence_forbidden": True}, "feature_repair": {"bounded_shape_scalars_only": True, "std_floor": STD_FLOOR, "clip": CLIP, "same_split_as_pg70": True}, "required_gates": {"dev_confirm_recall_min": 0.80, "dev_false_accept_zero": True, "unknown_misname_zero": True, "unknown_strict_abstain": True, "independent_seed_rerun": True, "fresh_docker_rerun": True}, "run_result": {"capability_gate": report["capability_gate"], "training_allowed": False, "memory_promotion_allowed": False}, "next_experiment": "PG72 independent seed and fresh Docker replay of the frozen v2 head"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("# PG-71 Trace decision head v2\n\n" + f"bounded-shape repair；device={device}；dev confirm recall={dev_metrics.get('confirm_recall', 0.0)}；unknown misname={unknown_metrics.get('misname_count', 0)}；unknown abstain={unknown_metrics.get('strict_abstain', False)}。\n\ncapability gate=`{report['capability_gate']['status']}`；training/memory promotion=false。\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps({"protocol_id": report["protocol_id"], "capability_gate": report["capability_gate"]["status"], "dev_confirm_recall": report["metrics"]["dev_holdout"].get("confirm_recall", 0.0), "unknown_misname_count": report["metrics"]["unknown_family_holdout"].get("misname_count", 0), "device": report["source"]["device"], "training_allowed": False}, ensure_ascii=False, indent=2))
