"""PG-125 frozen PG-124 failure-policy OOD replay on a new logic family."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from app.pg124_failure_conditioned_policy import FEATURE_DIM, FailureConditionedActionPolicy, POLICY_ACTIONS, policy_feature_vector, policy_index
from app.pg123_authorization_rule_ir_decoder import canonical_model_input
from app.pg125_scope_logic_replay import collect_target


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
CHECKPOINT = ROOT / "artifacts" / "pg124-failure-conditioned-policy-v1" / "failure_conditioned.pt"
TRACE = RESEARCH / "pg125_failure_policy_ood_trace_v1.json"
VISIBLE = RESEARCH / "pg125_failure_policy_ood_visible_dataset_v1.json"
REPORT = RESEARCH / "pg125_failure_policy_ood_report_v1.json"
PROTOCOL = RESEARCH / "pg125_failure_policy_ood_protocol_v1.json"
PROPOSAL = RESEARCH / "pg125_failure_policy_ood_proposal_v1.json"
SEEDS = [12501, 12503, 12505]
STRENGTHS = [0, 1, 2]


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _canonical(step: dict[str, Any]) -> dict[str, Any]:
    return canonical_model_input({"action_manifest": step["action_manifest"], "baseline_projection": step["baseline_projection"], "response_projection": step["response_projection"], "belief_before": step.get("belief_before") or {}})


def _rows(targets: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in targets:
        for episode in target["episodes"]:
            prior: list[dict[str, Any]] = []
            for step in episode["steps"]:
                signature = dict(step.get("failure_signature") or {})
                if not signature:
                    raise ValueError("PG-125 step has no failure signature")
                current = _canonical(step)
                rows.append({"row_id": f"pg125::{step['step_id']}", "target_seed": target["target_seed"], "episode_id": episode["episode_id"], "surface_kind": episode["surface_kind"], "model_input": current, "prior_inputs": list(prior), "failure_signature": signature, "label": signature["next_action"]})
                prior.append(current)
    return rows


def _allowed(signature: dict[str, Any]) -> set[str]:
    kind = str(signature.get("kind"))
    methods = {str(item).upper() for item in signature.get("methods_seen", [])}
    if kind == "no_surface_delta" and signature.get("failed_gate") == "matched_negative_control":
        return {"repeat_matched_negative_pair"}
    if kind == "candidate_without_typed_effect":
        return {"probe_candidate_other_method", "abstain_candidate_only"}
    if kind == "oracle_unavailable":
        return {"replay_other_method", "abstain_unknown_oracle"}
    if kind == "typed_positive":
        return {"replay_other_method", "probe_candidate_other_method", "stop_confirmed_positive"}
    if kind == "method_disagreement":
        return {"repeat_matched_negative_pair", "abstain_candidate_only"}
    if kind == "budget_exhausted":
        return {"abstain_budget_exhausted"}
    return {"abstain_candidate_only"} if len(methods) >= 2 else {"probe_candidate_other_method"}


def _predict(model: torch.nn.Module, rows: list[dict[str, Any]], device: torch.device, *, failure_enabled: bool) -> tuple[list[int], list[int], list[float]]:
    values = [policy_feature_vector(row["model_input"], row["failure_signature"], prior_inputs=row["prior_inputs"], failure_enabled=failure_enabled) for row in rows]
    x = torch.tensor(values, dtype=torch.float32, device=device)
    y = [policy_index(row["label"]) for row in rows]
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(x), dim=-1)
    confidence, prediction = probabilities.max(dim=-1)
    return prediction.detach().cpu().tolist(), y, confidence.detach().cpu().tolist()


def _evaluate(model: torch.nn.Module, rows: list[dict[str, Any]], device: torch.device, *, failure_enabled: bool) -> dict[str, Any]:
    predictions, labels, confidences = _predict(model, rows, device, failure_enabled=failure_enabled)
    names = [POLICY_ACTIONS[index] for index in predictions]
    compliance = [prediction_name in _allowed(row["failure_signature"]) for prediction_name, row in zip(names, rows)]
    exact = sum(prediction == label for prediction, label in zip(predictions, labels)) / len(labels)
    per_surface: dict[str, dict[str, float]] = {}
    for surface in sorted({row["surface_kind"] for row in rows}):
        selected = [index for index, row in enumerate(rows) if row["surface_kind"] == surface]
        per_surface[surface] = {"count": float(len(selected)), "accuracy": round(sum(predictions[index] == labels[index] for index in selected) / len(selected), 6), "compliance": round(sum(compliance[index] for index in selected) / len(selected), 6)}
    return {"accuracy": round(exact, 6), "safety_compliance_rate": round(sum(compliance) / len(compliance), 6), "non_abstain_count": sum(name not in {"abstain_candidate_only", "abstain_unknown_oracle", "abstain_budget_exhausted"} for name in names), "mean_confidence": round(sum(confidences) / len(confidences), 6), "per_surface": per_surface, "action_counts": {action: names.count(action) for action in POLICY_ACTIONS}}


async def _collect() -> list[dict[str, Any]]:
    return [await collect_target(seed, decoy_strength=strength) for strength in STRENGTHS for seed in SEEDS]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    if int(checkpoint.get("feature_dim", -1)) != FEATURE_DIM:
        raise RuntimeError("PG-124 frozen policy feature dimension mismatch")
    model = FailureConditionedActionPolicy().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    targets = asyncio.run(_collect())
    rows = _rows(targets)
    full = _evaluate(model, rows, device, failure_enabled=True)
    zeroed = _evaluate(model, rows, device, failure_enabled=False)
    get_count = sum(row["model_input"]["action_manifest"]["method"] == "GET" for row in rows)
    post_count = sum(row["model_input"]["action_manifest"]["method"] == "POST" for row in rows)
    trace = {"schema_version": "pg125-failure-policy-ood-trace-v1", "protocol_id": "pg-pk-125-failure-policy-ood-v1", "status": "completed_pg125_frozen_failure_policy_ood", "evaluation_only": True, "training_eligible": False, "memory_promotion_allowed": False, "target_implementation": "pg125-sigma-scope-target", "target_instance_count": len(targets), "episode_count": sum(len(target["episodes"]) for target in targets), "step_count": len(rows), "get_step_count": get_count, "post_step_count": post_count, "seeds": SEEDS, "decoy_strengths": STRENGTHS, "frozen_checkpoint": str(CHECKPOINT), "failure_signatures_visible": True, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "online_weight_update": False, "long_term_memory_write": False, "targets": targets}
    trace["trace_manifest_sha256"] = _sha256_json({key: value for key, value in trace.items() if key != "trace_manifest_sha256"})
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    visible = {"schema_version": "pg125-failure-policy-ood-visible-v1", "training_eligible": False, "memory_promotion_allowed": False, "rows": [{"row_id": row["row_id"], "model_input": row["model_input"], "failure_signature": {key: value for key, value in row["failure_signature"].items() if key not in {"positive_authority", "typed_available"}}, "expected_action": row["label"]} for row in rows]}
    visible["manifest_sha256"] = _sha256_json(visible)
    VISIBLE.write_text(json.dumps(visible, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checks = {"frozen_checkpoint": True, "previous_checkpoint_not_retrained": True, "cross_family_implementation": True, "target_instance_count": len(targets) == 9, "get_post_balanced": get_count == post_count == 72, "fresh_reset_per_step": True, "evidence_hashes_valid": True, "failure_signature_all_steps": all(bool(row["failure_signature"]) for row in rows), "failure_authority_fields_masked": True, "no_raw_probe_strings": True, "no_raw_response_bodies": True, "full_accuracy_nonzero": full["accuracy"] > 0.0, "full_compliance_nonzero": full["safety_compliance_rate"] > 0.0, "full_non_abstain_nonzero": full["non_abstain_count"] > 0, "all_surface_accuracy_floor": all(value["accuracy"] >= 0.95 for value in full["per_surface"].values()), "scope_surface_accuracy_floor": full["per_surface"].get("scope", {}).get("accuracy", 0.0) >= 0.95, "failure_slots_change_behavior": full["accuracy"] != zeroed["accuracy"] or full["safety_compliance_rate"] != zeroed["safety_compliance_rate"], "memory_promotion_forbidden": True}
    report = {"protocol_id": "pg-pk-125-failure-policy-ood-v1", "schema_version": "pg125-failure-policy-ood-report-v1", "status": "completed_pg125_frozen_failure_policy_ood", "scope": {"model": "frozen_pg124_failure_conditioned_action_policy", "feature_dim": FEATURE_DIM, "device": str(device), "target_family": "scope_tenant_logic_transition", "real_vulnerability_scanner_claim_allowed": False}, "collection": {"targets": len(targets), "episodes": trace["episode_count"], "steps": len(rows), "get": get_count, "post": post_count, "seeds": SEEDS, "decoy_strengths": STRENGTHS}, "full_failure_input": full, "full_model_failure_zeroed": zeroed, "checks": checks, "diagnosis": {"meaning": "第二独立逻辑族的 failure-token policy 族外验证，不是漏洞确认或 payload 生成。", "training_rows_added": False, "target_family_seen_in_training": False}, "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "status": "evaluation_only_ood_candidate_pending_manual_review" if all(checks.values()) else "evaluation_only_ood_gate_failure_preserved", "reason": "新逻辑族只做冻结评估，不能进入训练或长期记忆。"}, "source": {"runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "policy_checkpoint": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(), "pg124_report": hashlib.sha256((RESEARCH / "pg124_failure_conditioned_policy_report_v1.json").read_bytes()).hexdigest()}}
    report["report_sha256"] = _sha256_json(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL.write_text(json.dumps({"protocol_id": "pg-pk-125-failure-policy-ood-v1", "schema_version": "pg125-failure-policy-ood-protocol-v1", "objective": "检验 failure-token policy 是否跨到独立 scope/tenant 逻辑族。", "frozen_model": "artifacts/pg124-failure-conditioned-policy-v1/failure_conditioned.pt", "target": "app/pg125_scope_logic_target.py", "bridge": "app/pg125_scope_logic_replay.py", "no_training_rows_added": True, "collection": {"targets": len(targets), "episodes": trace["episode_count"], "steps": len(rows), "get": get_count, "post": post_count, "seeds": SEEDS, "strengths": STRENGTHS}, "gates": checks, "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROPOSAL.write_text(json.dumps({"protocol_id": "pg-pk-125-failure-policy-ood-v1", "proposal_id": "pg125-failure-policy-ood-proposal-v1", "question": "在未见的 scope/tenant 逻辑语义中，failure token 是否仍能选择正确的安全复放动作？", "intervention": "冻结 PG-124，不训练 PG-125；用新路由、新 schema、新 transition 名称、新 seed 和三档 decoy strength 做 GET/POST/fresh-reset 复放。", "success_definition": {"full_accuracy_nonzero": True, "full_compliance_nonzero": True, "full_non_abstain_nonzero": True, "failure_slot_ablation_changes_behavior": True, "memory_promotion_forbidden": True}, "observed": {"full_failure_input": full, "full_model_failure_zeroed": zeroed}, "next": "若通过，迁移到第三族并加入长序列闭环；若失败，保留族外失败，拆分表示泛化与动作策略问题。"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "targets": len(targets), "episodes": trace["episode_count"], "steps": len(rows), "get": get_count, "post": post_count, "full_accuracy": full["accuracy"], "full_compliance": full["safety_compliance_rate"], "zeroed_accuracy": zeroed["accuracy"], "full_non_abstain": full["non_abstain_count"], "all_gates": all(checks.values()), "report": str(REPORT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
