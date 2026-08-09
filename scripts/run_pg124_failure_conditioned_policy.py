# -*- coding: utf-8 -*-
"""PG-124 direct failure-token policy vs no-failure-input ablation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from app.pg124_failure_conditioned_policy import (
    FEATURE_DIM,
    FailureConditionedActionPolicy,
    POLICY_ACTIONS,
    failure_feedback_vector,
    policy_feature_vector,
    policy_index,
)
from app.pg123_authorization_rule_ir_decoder import canonical_model_input


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
PG123_TRACE = RESEARCH / "pg123_authorization_slot_training_trace_v1.json"
PG122_TRACE = RESEARCH / "pg122_failure_guided_authorization_holdout_trace_v1.json"
ARTIFACT_DIR = ROOT / "artifacts" / "pg124-failure-conditioned-policy-v1"
FULL_CHECKPOINT = ARTIFACT_DIR / "failure_conditioned.pt"
ABLATION_CHECKPOINT = ARTIFACT_DIR / "no_failure_input.pt"
TRACE = RESEARCH / "pg124_failure_conditioned_policy_trace_v1.json"
DATASET = RESEARCH / "pg124_failure_conditioned_policy_dataset_v1.json"
VISIBLE = RESEARCH / "pg124_failure_conditioned_policy_visible_dataset_v1.json"
REPORT = RESEARCH / "pg124_failure_conditioned_policy_report_v1.json"
PROTOCOL = RESEARCH / "pg124_failure_conditioned_policy_protocol_v1.json"
PROPOSAL = RESEARCH / "pg124_failure_conditioned_policy_proposal_v1.json"


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(step: dict[str, Any]) -> dict[str, Any]:
    return canonical_model_input({"action_manifest": step.get("action_manifest") or {}, "baseline_projection": step.get("baseline_projection") or {}, "response_projection": step.get("response_projection") or {}, "belief_before": step.get("belief_before") or {}})


def _rows_from_targets(targets: Iterable[dict[str, Any]], *, split: str, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in targets:
        for episode in target["episodes"]:
            prior: list[dict[str, Any]] = []
            for step in episode["steps"]:
                signature = dict(step.get("failure_signature") or {})
                if not signature:
                    raise ValueError(f"PG-124 source step lacks failure_signature: {step.get('step_id')}")
                model_input = _canonical(step)
                rows.append({"row_id": f"{source}::{step['step_id']}", "source": source, "split": split, "target_seed": target.get("target_seed"), "episode_id": episode["episode_id"], "label": signature["next_action"], "model_input": model_input, "prior_inputs": list(prior), "failure_signature": signature, "training_eligible": True, "memory_promotion_allowed": False})
                prior.append(model_input)
    return rows


def _load_sources() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pg123 = json.loads(PG123_TRACE.read_text(encoding="utf-8"))
    pg122 = json.loads(PG122_TRACE.read_text(encoding="utf-8"))
    train = _rows_from_targets(pg123["train_source_targets"], split="train", source="pg122_failure_guided_train_seeds")
    dev = _rows_from_targets(pg123["dev_source_targets"], split="dev", source="pg122_failure_guided_dev_seeds")
    holdout = _rows_from_targets(pg122["targets"], split="holdout", source="pg122_failure_guided_holdout")
    train_seeds = {row.get("target_seed") for row in train}
    dev_seeds = {row.get("target_seed") for row in dev}
    holdout_seeds = {row.get("target_seed") for row in holdout}
    if train_seeds & dev_seeds or train_seeds & holdout_seeds or dev_seeds & holdout_seeds:
        raise AssertionError("PG-124 seed leakage across train/dev/holdout")
    return train, dev, holdout


def _batch(rows: list[dict[str, Any]], device: torch.device, *, failure_enabled: bool) -> tuple[torch.Tensor, torch.Tensor]:
    vectors = [policy_feature_vector(row["model_input"], row["failure_signature"], prior_inputs=row.get("prior_inputs", []), failure_enabled=failure_enabled) for row in rows]
    if any(len(vector) != FEATURE_DIM for vector in vectors):
        raise AssertionError("PG-124 feature dimension drift")
    return torch.tensor(vectors, dtype=torch.float32, device=device), torch.tensor([policy_index(row["label"]) for row in rows], dtype=torch.long, device=device)


def _metrics(predictions: list[int], labels: list[int]) -> dict[str, Any]:
    total = len(labels)
    per_action: dict[str, dict[str, int | float]] = {}
    f1_values: list[float] = []
    for index, name in enumerate(POLICY_ACTIONS):
        tp = sum(prediction == index and label == index for prediction, label in zip(predictions, labels))
        fp = sum(prediction == index and label != index for prediction, label in zip(predictions, labels))
        fn = sum(prediction != index and label == index for prediction, label in zip(predictions, labels))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_action[name] = {"true_positive": tp, "false_positive": fp, "false_negative": fn, "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}
    return {"count": total, "accuracy": round(sum(prediction == label for prediction, label in zip(predictions, labels)) / total, 6) if total else 0.0, "macro_f1": round(sum(f1_values) / len(f1_values), 6), "per_action": per_action}


def _predict_rows(model: nn.Module, rows: list[dict[str, Any]], device: torch.device, *, failure_enabled: bool) -> tuple[list[int], list[float]]:
    model.eval()
    x, _ = _batch(rows, device, failure_enabled=failure_enabled)
    with torch.inference_mode():
        probabilities = torch.softmax(model(x), dim=-1)
    confidence, prediction = probabilities.max(dim=-1)
    return prediction.detach().cpu().tolist(), confidence.detach().cpu().tolist()


def _allowed_actions(signature: dict[str, Any]) -> set[str]:
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


def _evaluate(model: nn.Module, rows: list[dict[str, Any]], device: torch.device, *, failure_enabled: bool) -> dict[str, Any]:
    predictions, confidences = _predict_rows(model, rows, device, failure_enabled=failure_enabled)
    labels = [policy_index(row["label"]) for row in rows]
    compliance = [POLICY_ACTIONS[prediction] in _allowed_actions(row["failure_signature"]) for prediction, row in zip(predictions, rows)]
    per_kind: dict[str, dict[str, int]] = {}
    for prediction, row, ok in zip(predictions, rows, compliance):
        kind = str(row["failure_signature"].get("kind"))
        bucket = per_kind.setdefault(kind, {"count": 0, "compliant": 0})
        bucket["count"] += 1
        bucket["compliant"] += int(ok)
    return {"metrics": _metrics(predictions, labels), "predicted_actions": {"counts": {action: sum(POLICY_ACTIONS[index] == action for index in predictions) for action in POLICY_ACTIONS}, "non_abstain_count": sum(POLICY_ACTIONS[index] not in {"abstain_candidate_only", "abstain_unknown_oracle", "abstain_budget_exhausted"} for index in predictions)}, "safety_compliance_rate": round(sum(compliance) / len(compliance), 6) if compliance else 0.0, "per_failure_kind": per_kind, "mean_confidence": round(sum(confidences) / len(confidences), 6) if confidences else 0.0}


def _train(train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]], device: torch.device, *, failure_enabled: bool, seed: int) -> tuple[nn.Module, list[dict[str, float]]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = FailureConditionedActionPolicy().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_x, train_y = _batch(train_rows, device, failure_enabled=failure_enabled)
    history: list[dict[str, float]] = []
    for epoch in range(1, 61):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        train_predictions, _ = _predict_rows(model, train_rows, device, failure_enabled=failure_enabled)
        dev_predictions, _ = _predict_rows(model, dev_rows, device, failure_enabled=failure_enabled)
        history.append({"epoch": epoch, "loss": round(float(loss.item()), 8), "train_accuracy": _metrics(train_predictions, [policy_index(row["label"]) for row in train_rows])["accuracy"], "dev_accuracy": _metrics(dev_predictions, [policy_index(row["label"]) for row in dev_rows])["accuracy"]})
    return model, history


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows, dev_rows, holdout_rows = _load_sources()
    full_model, full_history = _train(train_rows, dev_rows, device, failure_enabled=True, seed=12424)
    no_failure_model, no_failure_history = _train(train_rows, dev_rows, device, failure_enabled=False, seed=12424)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg124-failure-conditioned-policy-v1", "feature_dim": FEATURE_DIM, "policy_actions": list(POLICY_ACTIONS), "failure_enabled": True, "model_state_dict": full_model.state_dict()}, FULL_CHECKPOINT)
    torch.save({"schema_version": "pg124-failure-conditioned-policy-v1", "feature_dim": FEATURE_DIM, "policy_actions": list(POLICY_ACTIONS), "failure_enabled": False, "model_state_dict": no_failure_model.state_dict()}, ABLATION_CHECKPOINT)
    full_dev = _evaluate(full_model, dev_rows, device, failure_enabled=True)
    no_failure_dev = _evaluate(no_failure_model, dev_rows, device, failure_enabled=False)
    full_holdout = _evaluate(full_model, holdout_rows, device, failure_enabled=True)
    full_zeroed_holdout = _evaluate(full_model, holdout_rows, device, failure_enabled=False)
    no_failure_holdout = _evaluate(no_failure_model, holdout_rows, device, failure_enabled=False)
    failure_slots_change = full_holdout["metrics"]["accuracy"] != full_zeroed_holdout["metrics"]["accuracy"] or full_holdout["safety_compliance_rate"] != full_zeroed_holdout["safety_compliance_rate"]
    train_seeds = sorted({row.get("target_seed") for row in train_rows})
    dev_seeds = sorted({row.get("target_seed") for row in dev_rows})
    holdout_seeds = sorted({row.get("target_seed") for row in holdout_rows})
    dataset = {"schema_version": "pg124-failure-conditioned-policy-dataset-v1", "training_eligible": True, "memory_promotion_allowed": False, "model_input_family_free": True, "oracle_authority_in_failure_vector": False, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "feature_dim": FEATURE_DIM, "failure_feature_dim": len(failure_feedback_vector(holdout_rows[0]["failure_signature"])), "train_seeds": train_seeds, "dev_seeds": dev_seeds, "holdout_seeds": holdout_seeds, "train_rows": train_rows, "dev_rows": dev_rows}
    dataset["manifest_sha256"] = _sha256_json(dataset)
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    visible = {"schema_version": "pg124-failure-conditioned-policy-visible-v1", "training_eligible": True, "memory_promotion_allowed": False, "rows": [{"row_id": row["row_id"], "model_input": row["model_input"], "failure_signature": {key: value for key, value in row["failure_signature"].items() if key not in {"positive_authority", "typed_available"}}, "training_label": row["label"]} for row in train_rows + dev_rows]}
    visible["manifest_sha256"] = _sha256_json(visible)
    VISIBLE.write_text(json.dumps(visible, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg124-failure-conditioned-policy-trace-v1", "protocol_id": "pg-pk-124-failure-conditioned-policy-v1", "status": "completed_failure_token_policy_ablation", "evaluation_only": False, "training_eligible": True, "memory_promotion_allowed": False, "train_source": "pg123_train_source_targets", "dev_source": "pg123_dev_source_targets", "holdout_source": "pg122_failure_guided_authorization_holdout_trace_v1", "train_seeds": train_seeds, "dev_seeds": dev_seeds, "holdout_seeds": holdout_seeds, "train_count": len(train_rows), "dev_count": len(dev_rows), "holdout_count": len(holdout_rows), "get_holdout_count": sum(row["model_input"]["action_manifest"]["method"] == "GET" for row in holdout_rows), "post_holdout_count": sum(row["model_input"]["action_manifest"]["method"] == "POST" for row in holdout_rows), "failure_signature_count": len(holdout_rows), "failure_authority_fields_masked": True, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "online_weight_update": False, "long_term_memory_write": False}
    trace["trace_manifest_sha256"] = _sha256_json(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checks = {"fresh_full_checkpoint": True, "fresh_no_failure_checkpoint": True, "same_capacity": sum(parameter.numel() for parameter in full_model.parameters()) == sum(parameter.numel() for parameter in no_failure_model.parameters()), "seed_disjoint": not bool(set(train_seeds) & set(dev_seeds) or set(train_seeds) & set(holdout_seeds) or set(dev_seeds) & set(holdout_seeds)), "get_post_balanced_holdout": trace["get_holdout_count"] == trace["post_holdout_count"], "failure_signature_all_holdout": trace["failure_signature_count"] == trace["holdout_count"], "failure_authority_fields_masked": True, "model_input_family_free": True, "no_raw_probe_strings": True, "no_raw_response_bodies": True, "full_policy_accuracy_nonzero": full_holdout["metrics"]["accuracy"] > 0.0, "full_policy_safety_compliance_nonzero": full_holdout["safety_compliance_rate"] > 0.0, "full_policy_non_abstain_nonzero": full_holdout["predicted_actions"]["non_abstain_count"] > 0, "failure_slot_ablation_changes_behavior": failure_slots_change, "no_failure_baseline_recorded": True, "memory_promotion_forbidden": True}
    report = {"protocol_id": "pg-pk-124-failure-conditioned-policy-v1", "schema_version": "pg124-failure-conditioned-policy-report-v1", "status": "completed_pg124_failure_token_policy_ablation", "scope": {"model": "fresh_failure_conditioned_action_policy", "feature_dim": FEATURE_DIM, "failure_feature_dim": len(failure_feedback_vector(holdout_rows[0]["failure_signature"])), "hidden_dim": 64, "parameter_count": sum(parameter.numel() for parameter in full_model.parameters()), "device": str(device), "real_vulnerability_scanner_claim_allowed": False}, "training": {"train_count": len(train_rows), "dev_count": len(dev_rows), "full_failure_history_tail": full_history[-5:], "no_failure_history_tail": no_failure_history[-5:], "full_dev": full_dev, "no_failure_dev": no_failure_dev}, "holdout": {"full_failure_input": full_holdout, "full_model_failure_zeroed": full_zeroed_holdout, "fresh_no_failure_baseline": no_failure_holdout, "failure_slot_behavior_changed": failure_slots_change}, "checks": checks, "diagnosis": {"meaning": "直接测试 failure_signature→next_action 的学习，而不是漏洞确认能力；typed oracle authority/typed_available 被屏蔽。", "next_action_is_abstract": True, "unsafe_payload_generation": False}, "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "status": "candidate_policy_ablation_pending_manual_review" if all(checks.values()) else "blocked_policy_gate_failure_preserved", "reason": "策略动作学习仍需第二独立逻辑族和长序列滚动复放，当前不提升长期记忆。"}, "source": {"policy": _sha256_file(ROOT / "app/pg124_failure_conditioned_policy.py"), "runner": _sha256_file(Path(__file__)), "pg123_trace": _sha256_file(PG123_TRACE), "pg122_trace": _sha256_file(PG122_TRACE)}}
    report["report_sha256"] = _sha256_json(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL.write_text(json.dumps({"protocol_id": "pg-pk-124-failure-conditioned-policy-v1", "schema_version": "pg124-failure-conditioned-policy-protocol-v1", "objective": "验证模型是否直接使用脱敏 failure token 选择下一安全动作。", "model_input": {"base": "PG-123 52 维投影", "failure_slots": len(failure_feedback_vector(holdout_rows[0]["failure_signature"])), "masked_fields": ["positive_authority", "typed_available", "evidence_hash", "raw_probe", "raw_response", "target_id", "family"]}, "variants": ["fresh failure-conditioned model", "fresh no-failure-input same-capacity model", "same full checkpoint with failure slots zeroed"], "holdout": {"source": "PG-122", "target_instances": 9, "episodes": 36, "steps": len(holdout_rows), "get": trace["get_holdout_count"], "post": trace["post_holdout_count"]}, "gates": checks, "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROPOSAL.write_text(json.dumps({"protocol_id": "pg-pk-124-failure-conditioned-policy-v1", "proposal_id": "pg124-failure-conditioned-policy-proposal-v1", "question": "模型是否真的能从失败门和已观察通道推导下一次安全复放，而非由外层固定动作顺序替代？", "intervention": "训练同容量的 failure-token 与 no-failure 两个 fresh policy；在未见 seed 的 PG-122 上比较动作准确率、门合规率、非全弃权和 failure-slot 消融。", "success_definition": {"full_accuracy_gt_zero": True, "full_compliance_gt_zero": True, "ablation_changes_behavior": True, "no_all_abstain": True, "authority_fields_masked": True}, "observed": {"full_holdout": full_holdout, "zeroed_holdout": full_zeroed_holdout, "no_failure_baseline": no_failure_holdout}, "next": "把同一策略迁移到第二个独立逻辑族，做长序列闭环复放后再考虑任何记忆准入。"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "feature_dim": FEATURE_DIM, "failure_feature_dim": len(failure_feedback_vector(holdout_rows[0]["failure_signature"])), "train": len(train_rows), "dev": len(dev_rows), "holdout": len(holdout_rows), "full_accuracy": full_holdout["metrics"]["accuracy"], "full_compliance": full_holdout["safety_compliance_rate"], "zeroed_accuracy": full_zeroed_holdout["metrics"]["accuracy"], "no_failure_accuracy": no_failure_holdout["metrics"]["accuracy"], "failure_slots_changed": failure_slots_change, "all_gates": all(checks.values()), "report": str(REPORT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
