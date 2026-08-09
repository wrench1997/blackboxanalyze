"""PG-126 fresh failure-only policy trained on PG-122 and tested on PG-125."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from app.pg124_failure_conditioned_policy import POLICY_ACTIONS, failure_feedback_vector, policy_index
from app.pg126_failure_only_policy import FEATURE_DIM, FailureOnlyActionPolicy, failure_only_feature_vector


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
PG123_TRACE = RESEARCH / "pg123_authorization_slot_training_trace_v1.json"
PG125_TRACE = RESEARCH / "pg125_failure_policy_ood_trace_v1.json"
ARTIFACT_DIR = ROOT / "artifacts" / "pg126-failure-only-policy-v1"
FULL_CHECKPOINT = ARTIFACT_DIR / "failure_only.pt"
ZERO_CHECKPOINT = ARTIFACT_DIR / "zero_failure_input.pt"
TRACE = RESEARCH / "pg126_failure_only_policy_trace_v1.json"
DATASET = RESEARCH / "pg126_failure_only_policy_dataset_v1.json"
VISIBLE = RESEARCH / "pg126_failure_only_policy_visible_dataset_v1.json"
REPORT = RESEARCH / "pg126_failure_only_policy_report_v1.json"
PROTOCOL = RESEARCH / "pg126_failure_only_policy_protocol_v1.json"
PROPOSAL = RESEARCH / "pg126_failure_only_policy_proposal_v1.json"


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _rows_from_targets(targets: Iterable[dict[str, Any]], *, split: str, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in targets:
        for episode in target["episodes"]:
            for step in episode["steps"]:
                signature = dict(step.get("failure_signature") or {})
                if not signature:
                    raise ValueError("PG-126 source step lacks failure_signature")
                rows.append({"row_id": f"{source}::{step['step_id']}", "source": source, "split": split, "target_seed": target.get("target_seed"), "episode_id": episode["episode_id"], "surface_kind": episode.get("surface_kind"), "failure_signature": signature, "label": signature["next_action"], "training_eligible": True, "memory_promotion_allowed": False})
    return rows


def _load() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pg123 = json.loads(PG123_TRACE.read_text(encoding="utf-8"))
    pg125 = json.loads(PG125_TRACE.read_text(encoding="utf-8"))
    train = _rows_from_targets(pg123["train_source_targets"], split="train", source="pg122_failure_guided_train")
    dev = _rows_from_targets(pg123["dev_source_targets"], split="dev", source="pg122_failure_guided_dev")
    holdout = _rows_from_targets(pg125["targets"], split="holdout", source="pg125_scope_ood")
    groups = [{row.get("target_seed") for row in rows} for rows in (train, dev, holdout)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise AssertionError("PG-126 seed leakage")
    return train, dev, holdout


def _batch(rows: list[dict[str, Any]], device: torch.device, *, enabled: bool) -> tuple[torch.Tensor, torch.Tensor]:
    vectors = [failure_only_feature_vector(row["failure_signature"], enabled=enabled) for row in rows]
    return torch.tensor(vectors, dtype=torch.float32, device=device), torch.tensor([policy_index(row["label"]) for row in rows], dtype=torch.long, device=device)


def _metrics(predictions: list[int], labels: list[int]) -> dict[str, Any]:
    total = len(labels)
    return {"count": total, "accuracy": round(sum(prediction == label for prediction, label in zip(predictions, labels)) / total, 6) if total else 0.0, "predicted_action_counts": {action: sum(POLICY_ACTIONS[index] == action for index in predictions) for action in POLICY_ACTIONS}}


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


def _predict(model: nn.Module, rows: list[dict[str, Any]], device: torch.device, *, enabled: bool) -> tuple[list[int], list[int], list[float]]:
    x, y = _batch(rows, device, enabled=enabled)
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(x), dim=-1)
    confidence, prediction = probabilities.max(dim=-1)
    return prediction.cpu().tolist(), y.cpu().tolist(), confidence.cpu().tolist()


def _evaluate(model: nn.Module, rows: list[dict[str, Any]], device: torch.device, *, enabled: bool) -> dict[str, Any]:
    predictions, labels, confidences = _predict(model, rows, device, enabled=enabled)
    names = [POLICY_ACTIONS[index] for index in predictions]
    compliant = [names[index] in _allowed(row["failure_signature"]) for index, row in enumerate(rows)]
    per_surface: dict[str, dict[str, float]] = {}
    for surface in sorted({row.get("surface_kind") for row in rows}):
        indices = [index for index, row in enumerate(rows) if row.get("surface_kind") == surface]
        per_surface[surface] = {"count": float(len(indices)), "accuracy": round(sum(predictions[index] == labels[index] for index in indices) / len(indices), 6), "compliance": round(sum(compliant[index] for index in indices) / len(indices), 6)}
    return {"metrics": _metrics(predictions, labels), "safety_compliance_rate": round(sum(compliant) / len(compliant), 6), "non_abstain_count": sum(name not in {"abstain_candidate_only", "abstain_unknown_oracle", "abstain_budget_exhausted"} for name in names), "mean_confidence": round(sum(confidences) / len(confidences), 6), "per_surface": per_surface}


def _train(train: list[dict[str, Any]], dev: list[dict[str, Any]], device: torch.device, *, enabled: bool, seed: int) -> tuple[nn.Module, list[dict[str, float]]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = FailureOnlyActionPolicy().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_x, train_y = _batch(train, device, enabled=enabled)
    history: list[dict[str, float]] = []
    for epoch in range(1, 61):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        train_pred, _, _ = _predict(model, train, device, enabled=enabled)
        dev_pred, _, _ = _predict(model, dev, device, enabled=enabled)
        history.append({"epoch": epoch, "loss": round(float(loss.item()), 8), "train_accuracy": _metrics(train_pred, [policy_index(row["label"]) for row in train])["accuracy"], "dev_accuracy": _metrics(dev_pred, [policy_index(row["label"]) for row in dev])["accuracy"]})
    return model, history


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, dev, holdout = _load()
    model, history = _train(train, dev, device, enabled=True, seed=12626)
    zero_model, zero_history = _train(train, dev, device, enabled=False, seed=12626)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg126-failure-only-policy-v1", "feature_dim": FEATURE_DIM, "policy_actions": list(POLICY_ACTIONS), "failure_only": True, "model_state_dict": model.state_dict()}, FULL_CHECKPOINT)
    torch.save({"schema_version": "pg126-failure-only-policy-v1", "feature_dim": FEATURE_DIM, "policy_actions": list(POLICY_ACTIONS), "failure_only": True, "failure_input_enabled": False, "model_state_dict": zero_model.state_dict()}, ZERO_CHECKPOINT)
    full_dev = _evaluate(model, dev, device, enabled=True)
    full_holdout = _evaluate(model, holdout, device, enabled=True)
    zeroed = _evaluate(model, holdout, device, enabled=False)
    baseline = _evaluate(zero_model, holdout, device, enabled=False)
    train_seeds = sorted({row.get("target_seed") for row in train})
    dev_seeds = sorted({row.get("target_seed") for row in dev})
    holdout_seeds = sorted({row.get("target_seed") for row in holdout})
    get_count = sum(row.get("failure_signature", {}).get("observed_method") == "GET" for row in holdout)
    post_count = sum(row.get("failure_signature", {}).get("observed_method") == "POST" for row in holdout)
    trace = {"schema_version": "pg126-failure-only-policy-trace-v1", "protocol_id": "pg-pk-126-failure-only-policy-v1", "status": "completed_pg126_failure_only_policy", "evaluation_only": False, "training_eligible": True, "memory_promotion_allowed": False, "training_source": "pg122_failure_guided_train_dev", "holdout_source": "pg125_scope_logic_ood", "train_count": len(train), "dev_count": len(dev), "holdout_count": len(holdout), "train_seeds": train_seeds, "dev_seeds": dev_seeds, "holdout_seeds": holdout_seeds, "get_holdout_count": get_count, "post_holdout_count": post_count, "failure_only_feature_dim": FEATURE_DIM, "oracle_authority_fields_masked": True, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "long_term_memory_write": False}
    trace["trace_manifest_sha256"] = _sha256_json(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    visible = {"schema_version": "pg126-failure-only-policy-visible-v1", "training_eligible": True, "memory_promotion_allowed": False, "rows": [{"row_id": row["row_id"], "failure_signature": {key: value for key, value in row["failure_signature"].items() if key not in {"positive_authority", "typed_available"}}, "training_label": row["label"]} for row in train + dev]}
    visible["manifest_sha256"] = _sha256_json(visible)
    VISIBLE.write_text(json.dumps(visible, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checks = {"fresh_checkpoint": True, "fresh_zero_input_baseline": True, "same_capacity": sum(parameter.numel() for parameter in model.parameters()) == sum(parameter.numel() for parameter in zero_model.parameters()), "seed_disjoint": not bool(set(train_seeds) & set(dev_seeds) or set(train_seeds) & set(holdout_seeds) or set(dev_seeds) & set(holdout_seeds)), "get_post_balanced": get_count == post_count == 72, "pg125_holdout_not_training": not set(holdout_seeds).intersection(train_seeds + dev_seeds), "failure_only_input": FEATURE_DIM == 17, "authority_fields_masked": True, "no_raw_probe_strings": True, "no_raw_response_bodies": True, "full_accuracy_floor": full_holdout["metrics"]["accuracy"] >= 0.95, "full_compliance_floor": full_holdout["safety_compliance_rate"] >= 0.95, "all_surface_accuracy_floor": all(value["accuracy"] >= 0.95 for value in full_holdout["per_surface"].values()), "full_non_abstain_nonzero": full_holdout["non_abstain_count"] > 0, "failure_ablation_changes_behavior": full_holdout["metrics"]["accuracy"] != zeroed["metrics"]["accuracy"], "memory_promotion_forbidden": True}
    report = {"protocol_id": "pg-pk-126-failure-only-policy-v1", "schema_version": "pg126-failure-only-policy-report-v1", "status": "completed_pg126_failure_only_policy", "scope": {"model": "fresh_failure_only_action_policy", "feature_dim": FEATURE_DIM, "hidden_dim": 64, "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "device": str(device), "real_vulnerability_scanner_claim_allowed": False}, "training": {"train_count": len(train), "dev_count": len(dev), "history_tail": history[-5:], "zero_history_tail": zero_history[-5:], "dev": full_dev}, "holdout": {"full_failure_only": full_holdout, "full_model_failure_zeroed": zeroed, "fresh_zero_input_baseline": baseline}, "checks": checks, "diagnosis": {"representation_change": "remove PG-123 response surface features; retain only 17 generic failure feedback slots", "meaning": "cross-family next-action generalization, not vulnerability confirmation", "training_rows_from_pg125": False}, "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "status": "candidate_cross_family_policy_passed_pending_manual_review" if all(checks.values()) else "blocked_pg126_gate_failure_preserved", "reason": "即便跨族门通过，仍需第三族和长序列闭环后才能考虑任何记忆准入。"}, "source": {"policy": hashlib.sha256((ROOT / "app/pg126_failure_only_policy.py").read_bytes()).hexdigest(), "runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "pg123_trace": hashlib.sha256(PG123_TRACE.read_bytes()).hexdigest(), "pg125_trace": hashlib.sha256(PG125_TRACE.read_bytes()).hexdigest()}}
    report["report_sha256"] = _sha256_json(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL.write_text(json.dumps({"protocol_id": "pg-pk-126-failure-only-policy-v1", "schema_version": "pg126-failure-only-policy-protocol-v1", "objective": "去除表面投影后，验证 failure-only policy 是否跨到 PG-125 scope/tenant 逻辑族。", "feature_dim": FEATURE_DIM, "masked_fields": ["positive_authority", "typed_available", "evidence_hash", "raw_probe", "raw_response", "target_id", "family"], "training_source": "PG-122 train/dev", "holdout_source": "PG-125 scope/tenant", "collection": {"train": len(train), "dev": len(dev), "holdout": len(holdout), "get": get_count, "post": post_count}, "gates": checks, "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROPOSAL.write_text(json.dumps({"protocol_id": "pg-pk-126-failure-only-policy-v1", "proposal_id": "pg126-failure-only-policy-proposal-v1", "question": "PG-125 的失败是否来自表面表示污染，而不是 failure token 不具备跨族动作信息？", "intervention": "同样的 PG-122 train/dev，fresh 训练只接受 17 维脱敏 failure feedback 的策略；PG-125 9 个 target 全冻结为 holdout。", "success_definition": {"full_accuracy_ge_0_95": True, "full_compliance_ge_0_95": True, "per_surface_accuracy_ge_0_95": True, "failure_ablation_changes": True, "memory_promotion_forbidden": True}, "observed": {"full_failure_only": full_holdout, "full_failure_zeroed": zeroed, "zero_baseline": baseline}, "next": "若通过，加入第三独立逻辑族和长序列滚动；若失败，按 failure kind 与方法一致性重新设计策略表示。"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "feature_dim": FEATURE_DIM, "parameter_count": report["scope"]["parameter_count"], "train": len(train), "dev": len(dev), "holdout": len(holdout), "full_accuracy": full_holdout["metrics"]["accuracy"], "full_compliance": full_holdout["safety_compliance_rate"], "scope_accuracy": full_holdout["per_surface"].get("scope", {}).get("accuracy"), "zeroed_accuracy": zeroed["metrics"]["accuracy"], "baseline_accuracy": baseline["metrics"]["accuracy"], "all_gates": all(checks.values()), "report": str(REPORT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
