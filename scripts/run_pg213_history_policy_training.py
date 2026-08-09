"""PG-213: train/evaluate a history-aware process policy from local traces."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg213_history_policy import HistoryProcessPolicy, train_history_policy


RESEARCH = ROOT / "research"
PG210 = RESEARCH / "pg210_ai_reference_payload_validation_report_v1.json"
PG212 = RESEARCH / "pg212_pikachu_sql_response_shape_loop_report_v1.json"
REPORT = RESEARCH / "pg213_history_policy_training_report_v1.json"
PROTOCOL = RESEARCH / "pg213_history_policy_training_protocol_v1.json"
TRACE = RESEARCH / "pg213_history_policy_training_trace_v1.json"
MARKDOWN = RESEARCH / "pg213_history_policy_training_report_v1.md"
ARTIFACT = ROOT / "artifacts" / "pg213-history-policy-v1" / "history_process_policy.pt"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _base_row(*, method: str, family: str, typed_available: bool, backend_state: str, status_class: str, history_len: int, previous_feedback: str, label: str, seed: int, field_count: int, binding_valid: bool = True, counterfactual: bool = False) -> dict[str, Any]:
    return {"method": method, "surface_family": family, "typed_available": bool(typed_available), "backend_state": backend_state, "status_class": status_class, "history_len": int(history_len), "previous_feedback": previous_feedback, "label": label, "seed": int(seed), "field_count": int(field_count), "binding_valid": bool(binding_valid), "negative_control": True, "counterfactual": bool(counterfactual)}


def _rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    pg210 = _load(PG210)
    pg212 = _load(PG212)
    all_rows: list[dict[str, Any]] = []
    for episode in pg210.get("episodes", []):
        ai = dict(episode.get("ai") or {})
        response = dict(ai.get("response") or {})
        oracle = dict(response.get("oracle") or {})
        decision = dict(ai.get("model_decision") or {})
        features = dict(decision.get("features") or {})
        typed = bool(oracle.get("typed_available", True))
        status_class = str(response.get("status_class", features.get("status_class", "2xx")))
        all_rows.append(_base_row(method=str(episode.get("method", "GET")), family="xss", typed_available=typed, backend_state="backend_response_observed", status_class=status_class, history_len=0, previous_feedback="none", label="safe_candidate" if ai.get("sent") else "abstain", seed=int(episode.get("seed", 0)), field_count=len(episode.get("fields") or [])))
        feedback = str((ai.get("ai_feedback") or {}).get("status", "dead_end"))
        post_label = "abstain" if bool(episode.get("ai_surface_effect")) else "retry_alternate"
        all_rows.append(_base_row(method=str(episode.get("method", "GET")), family="xss", typed_available=typed, backend_state="backend_response_observed", status_class=status_class, history_len=1, previous_feedback=feedback, label=post_label, seed=int(episode.get("seed", 0)), field_count=len(episode.get("fields") or [])))
    for episode in pg212.get("episodes", []):
        control = dict(episode.get("control") or {})
        oracle = dict(control.get("oracle") or {})
        projection = dict(control.get("response_projection") or {})
        state = str(oracle.get("backend_state", "database_unavailable"))
        status_class = str(projection.get("status_class", "2xx"))
        all_rows.append(_base_row(method=str(episode.get("method", "GET")), family="injection", typed_available=bool(oracle.get("typed_available")), backend_state=state, status_class=status_class, history_len=0, previous_feedback="none", label="abstain", seed=int(episode.get("seed", 0)), field_count=len(episode.get("fields") or [])))
        all_rows.append(_base_row(method=str(episode.get("method", "GET")), family="injection", typed_available=False, backend_state=state, status_class=status_class, history_len=1, previous_feedback="environment_failure", label="abstain", seed=int(episode.get("seed", 0)), field_count=len(episode.get("fields") or [])))
    train = [row for row in all_rows if int(row["seed"]) in {21001, 21201}]
    holdout = [row for row in all_rows if int(row["seed"]) in {21002, 21202}]
    # Counterfactuals are never sent.  They verify that a method/field binding
    # failure forces abstain, rather than letting a route-looking template win.
    # One seed's counterfactuals are seen during training; the other seed's are
    # held out, so this is a real binding-OOD check rather than a post-hoc veto.
    for rows in (train, holdout):
        rows.extend({**row, "method": "POST" if row["method"] == "GET" else "GET", "field_count": 0, "binding_valid": False, "counterfactual": True, "label": "abstain"} for row in list(rows))
    return train, holdout, {"all_rows": len(all_rows), "train_rows": len(train), "holdout_rows_with_counterfactuals": len(holdout), "counterfactual_rows": sum(int(row.get("counterfactual")) for row in holdout)}


def main() -> int:
    train_rows, holdout_rows, data = _rows()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(213)
    model = HistoryProcessPolicy().to(device)
    training = train_history_policy(model, train_rows, holdout_rows, epochs=120, seed=213)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg213-history-process-policy-v1", "model_state": model.state_dict(), "raw_inputs_retained": False, "promotion_allowed": False}, ARTIFACT)
    report = {"protocol_id": "pg-pk-213-history-policy-training-v1", "schema_version": "pg213-history-policy-training-report-v1", "status": "completed_history_policy_counterfactual_holdout", "device": str(device), "data": data, "training": training, "model": {"variant": "history_process_policy", "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())), "raw_inputs_retained": False, "online_weight_update": False}, "counterfactual_abstain_count": sum(int(row.get("counterfactual") and row.get("label") == "abstain") for row in holdout_rows), "promotion": {"training_eligible": False, "artifact_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}, "safety": {"local_trace_sources_only": True, "external_network_targets": False, "script_execution": False, "database_write": False, "oracle_labels_as_features": False, "raw_payloads_in_model": False, "raw_responses_in_model": False}}
    report["report_sha256"] = _digest(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg213-history-policy-training-protocol-v1", "train_sources": [str(PG210.relative_to(ROOT)), str(PG212.relative_to(ROOT))], "split": "seed holdout 21001/21201 train vs 21002/21202 holdout", "feedback_features": ["none", "dead_end", "candidate", "environment_failure", "rejected"], "actions": ["abstain", "safe_candidate", "retry_alternate"], "counterfactual_method_field_binding_required": True, "oracle_labels_as_features": False, "raw_payload_and_response_excluded": True, "promotion_allowed": False}
    protocol["protocol_sha256"] = _digest(protocol)
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE.write_text(json.dumps({"schema_version": "pg213-history-policy-training-trace-v1", "data": data, "training": training, "training_eligible": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN.write_text("\n".join(["# PG-213 history policy training", "", f"device={device}; train={data['train_rows']}; holdout={data['holdout_rows_with_counterfactuals']}; counterfactual={data['counterfactual_rows']}", f"train={training['train']}; holdout={training['holdout']}", "", "该 head 只学习失败反馈与绑定失败后的动作选择；artifact 仍是诊断用，未接管真实发包，也未提升长期记忆。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "data": data, "train": training["train"], "holdout": training["holdout"], "artifact": str(ARTIFACT.relative_to(ROOT)), "training_eligible": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
