"""Train/evaluate the fully tokenized failure-trajectory policy."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.failure_guided_scheduler import validate_failure_signature
from app.pg124_failure_conditioned_policy import POLICY_ACTIONS, policy_index
from app.pg125_scope_logic_replay import collect_target as collect_pg125_target
from app.pg127_resource_visibility_replay import collect_target as collect_pg127_target
from app.pg129_atomic_token_policy import (
    ATOMIC_GROUPS,
    ATOMIC_TOKEN_FEATURE_DIM,
    HIDDEN_DIM,
    MAX_ATOMIC_TOKENS,
    AtomicTokenActionPolicy,
    atomic_trajectory_matrix,
)


RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg129-atomic-token-policy-v1"
FULL_CHECKPOINT = ARTIFACT_DIR / "atomic_weighted.pt"
ZERO_CHECKPOINT = ARTIFACT_DIR / "atomic_zero.pt"
TRACE = RESEARCH / "pg129_atomic_token_trace_v1.json"
DATASET = RESEARCH / "pg129_atomic_token_dataset_v1.json"
VISIBLE = RESEARCH / "pg129_atomic_token_visible_dataset_v1.json"
REPORT = RESEARCH / "pg129_atomic_token_report_v1.json"
PROTOCOL = RESEARCH / "pg129_atomic_token_protocol_v1.json"
PROPOSAL = RESEARCH / "pg129_atomic_token_proposal_v1.json"

PG125_TRAIN_SEEDS = (12531, 12533, 12535)
PG127_TRAIN_SEEDS = (12911, 12913, 12915)
PG127_DEV_SEEDS = (12912, 12914, 12916)
PG127_HOLDOUT_SEEDS = (12901, 12903, 12905)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


async def _collect_targets() -> dict[str, list[dict[str, Any]]]:
    pg125 = [await collect_pg125_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG125_TRAIN_SEEDS)]
    pg127_train = [await collect_pg127_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG127_TRAIN_SEEDS)]
    pg127_dev = [await collect_pg127_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG127_DEV_SEEDS)]
    pg127_holdout = [await collect_pg127_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG127_HOLDOUT_SEEDS)]
    return {"pg125_fresh_train": pg125, "pg127_long_train": pg127_train, "pg127_long_dev": pg127_dev, "pg127_long_holdout": pg127_holdout}


def _rows_from_targets(targets: Iterable[dict[str, Any]], *, split: str, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in targets:
        for episode in target["episodes"]:
            if episode.get("episode_report", {}).get("status") != "accepted_evaluation":
                raise ValueError(f"PG-129 source episode was not accepted: {episode.get('episode_id')}")
            prefix: list[dict[str, Any]] = []
            for step in episode["steps"]:
                signature = dict(step.get("failure_signature") or {})
                validate_failure_signature(signature)
                prefix.append(signature)
                rows.append({"row_id": f"{source}::{step['step_id']}", "source": source, "split": split, "target_seed": target.get("target_seed"), "episode_id": episode["episode_id"], "surface_kind": episode.get("surface_kind"), "step_id": step["step_id"], "trajectory_signatures": [dict(item) for item in prefix], "failure_signature": signature, "label": signature["next_action"], "training_eligible": split in {"train", "dev"}, "memory_promotion_allowed": False})
    return rows


def _batch(rows: list[dict[str, Any]], device: torch.device, *, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    matrices = [atomic_trajectory_matrix(row["trajectory_signatures"], mode=mode) for row in rows]
    return torch.tensor(matrices, dtype=torch.float32, device=device), torch.tensor([policy_index(row["label"]) for row in rows], dtype=torch.long, device=device)


def _metrics(predictions: list[int], labels: list[int]) -> dict[str, Any]:
    total = len(labels)
    return {"count": total, "accuracy": round(sum(prediction == label for prediction, label in zip(predictions, labels)) / total, 6) if total else 0.0, "predicted_action_counts": {action: sum(POLICY_ACTIONS[index] == action for index in predictions) for action in POLICY_ACTIONS}}


def _allowed(signature: dict[str, Any]) -> set[str]:
    kind = str(signature.get("kind"))
    gate = str(signature.get("failed_gate"))
    methods = {str(item).upper() for item in signature.get("methods_seen", [])}
    remaining = int(signature.get("remaining_probe_budget", 0) or 0)
    if kind == "no_surface_delta" and gate == "matched_negative_control":
        return {"repeat_matched_negative_pair"}
    if kind in {"candidate_without_typed_effect", "no_surface_delta"}:
        return {"probe_candidate_other_method"} if len(methods) < 2 or remaining > 0 else {"abstain_candidate_only"}
    if kind == "oracle_unavailable":
        return {"replay_other_method"} if len(methods) < 2 or remaining > 0 else {"abstain_unknown_oracle"}
    if kind == "typed_positive":
        return {"probe_candidate_other_method"} if len(methods) < 2 else {"stop_confirmed_positive"}
    if kind == "method_disagreement":
        return {"repeat_matched_negative_pair", "abstain_candidate_only"}
    if kind == "budget_exhausted":
        return {"abstain_budget_exhausted"}
    return {"abstain_candidate_only"}


def _predict(model: nn.Module, rows: list[dict[str, Any]], device: torch.device, *, mode: str) -> tuple[list[int], list[int], list[float]]:
    x, y = _batch(rows, device, mode=mode)
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(x), dim=-1)
    confidence, prediction = probabilities.max(dim=-1)
    return prediction.cpu().tolist(), y.cpu().tolist(), confidence.cpu().tolist()


def _evaluate(model: nn.Module, rows: list[dict[str, Any]], device: torch.device, *, mode: str) -> dict[str, Any]:
    predictions, labels, confidences = _predict(model, rows, device, mode=mode)
    names = [POLICY_ACTIONS[index] for index in predictions]
    compliant = [names[index] in _allowed(row["failure_signature"]) for index, row in enumerate(rows)]
    per_surface: dict[str, dict[str, float]] = {}
    for surface in sorted({row.get("surface_kind") for row in rows}):
        indices = [index for index, row in enumerate(rows) if row.get("surface_kind") == surface]
        per_surface[surface] = {"count": float(len(indices)), "accuracy": round(sum(predictions[index] == labels[index] for index in indices) / len(indices), 6), "compliance": round(sum(compliant[index] for index in indices) / len(indices), 6)}
    budget_indices = [index for index, row in enumerate(rows) if int(row["failure_signature"].get("remaining_probe_budget", 0) or 0) > 0]
    negative_indices = [index for index, row in enumerate(rows) if row.get("surface_kind") in {"decoy", "steady", "blind"}]
    return {"mode": mode, "metrics": _metrics(predictions, labels), "safety_compliance_rate": round(sum(compliant) / len(compliant), 6), "non_abstain_count": sum(name not in {"abstain_candidate_only", "abstain_unknown_oracle", "abstain_budget_exhausted"} for name in names), "mean_confidence": round(sum(confidences) / len(confidences), 6), "per_surface": per_surface, "budget_rows": len(budget_indices), "budget_accuracy": round(sum(predictions[index] == labels[index] for index in budget_indices) / len(budget_indices), 6) if budget_indices else 0.0, "negative_false_stop_count": sum(names[index] == "stop_confirmed_positive" for index in negative_indices), "predictions": predictions, "labels": labels}


def _train(train: list[dict[str, Any]], dev: list[dict[str, Any]], device: torch.device, *, mode: str, seed: int) -> tuple[nn.Module, list[dict[str, float]]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = AtomicTokenActionPolicy().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_x, train_y = _batch(train, device, mode=mode)
    history: list[dict[str, float]] = []
    for epoch in range(1, 121):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        train_pred, _, _ = _predict(model, train, device, mode=mode)
        dev_pred, _, _ = _predict(model, dev, device, mode=mode)
        history.append({"epoch": epoch, "loss": round(float(loss.item()), 8), "train_accuracy": _metrics(train_pred, [policy_index(row["label"]) for row in train])["accuracy"], "dev_accuracy": _metrics(dev_pred, [policy_index(row["label"]) for row in dev])["accuracy"]})
    return model, history


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = asyncio.run(_collect_targets())
    train = _rows_from_targets(targets["pg125_fresh_train"], split="train", source="pg125_fresh_atomic_train") + _rows_from_targets(targets["pg127_long_train"], split="train", source="pg127_long_atomic_train")
    dev = _rows_from_targets(targets["pg127_long_dev"], split="dev", source="pg127_long_atomic_dev")
    holdout = _rows_from_targets(targets["pg127_long_holdout"], split="holdout", source="pg127_long_atomic_holdout")
    model, history = _train(train, dev, device, mode="weighted", seed=12929)
    zero_model, zero_history = _train(train, dev, device, mode="zero", seed=12929)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg129-atomic-token-policy-v1", "feature_dim": ATOMIC_TOKEN_FEATURE_DIM, "max_atomic_tokens": MAX_ATOMIC_TOKENS, "policy_actions": list(POLICY_ACTIONS), "atomic_mode": "weighted", "model_state_dict": model.state_dict()}, FULL_CHECKPOINT)
    torch.save({"schema_version": "pg129-atomic-token-policy-v1", "feature_dim": ATOMIC_TOKEN_FEATURE_DIM, "max_atomic_tokens": MAX_ATOMIC_TOKENS, "policy_actions": list(POLICY_ACTIONS), "atomic_mode": "zero", "model_state_dict": zero_model.state_dict()}, ZERO_CHECKPOINT)
    full_dev = _evaluate(model, dev, device, mode="weighted")
    weighted_holdout = _evaluate(model, holdout, device, mode="weighted")
    uniform_holdout = _evaluate(model, holdout, device, mode="uniform_tokens")
    zeroed_holdout = _evaluate(model, holdout, device, mode="zero")
    zero_baseline = _evaluate(zero_model, holdout, device, mode="zero")
    train_seeds = sorted({row.get("target_seed") for row in train})
    dev_seeds = sorted({row.get("target_seed") for row in dev})
    holdout_seeds = sorted({row.get("target_seed") for row in holdout})
    get_count = sum(row["failure_signature"].get("observed_method") == "GET" for row in holdout)
    post_count = sum(row["failure_signature"].get("observed_method") == "POST" for row in holdout)
    token_examples = []
    for row in holdout:
        if row["step_id"].endswith("-s06") and len(token_examples) < 6:
            matrix = atomic_trajectory_matrix(row["trajectory_signatures"], mode="weighted")
            token_examples.append({"step_id": row["step_id"], "token_count": len(row["trajectory_signatures"]) * len(ATOMIC_GROUPS), "last_step_token_weights": [round(float(matrix[index][14]), 6) for index in range((len(row["trajectory_signatures"])-1)*len(ATOMIC_GROUPS), len(row["trajectory_signatures"])*len(ATOMIC_GROUPS))]})
    trace = {"schema_version": "pg129-atomic-token-trace-v1", "protocol_id": "pg-pk-129-atomic-token-v1", "status": "completed_pg129_atomic_token_policy", "evaluation_only": False, "training_eligible": True, "memory_promotion_allowed": False, "sources": targets, "train_seeds": train_seeds, "dev_seeds": dev_seeds, "holdout_seeds": holdout_seeds, "train_count": len(train), "dev_count": len(dev), "holdout_count": len(holdout), "get_holdout_count": get_count, "post_holdout_count": post_count, "atomic_groups": list(ATOMIC_GROUPS), "max_atomic_tokens": MAX_ATOMIC_TOKENS, "token_feature_dim": ATOMIC_TOKEN_FEATURE_DIM, "token_examples": token_examples, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "long_term_memory_write": False}
    trace["trace_manifest_sha256"] = _sha256_json(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dataset = {"schema_version": "pg129-atomic-token-dataset-v1", "training_eligible": True, "memory_promotion_allowed": False, "rows": train + dev + holdout}
    dataset["manifest_sha256"] = _sha256_json(dataset)
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    visible = {"schema_version": "pg129-atomic-token-visible-v1", "training_eligible": True, "memory_promotion_allowed": False, "rows": [{"row_id": row["row_id"], "split": row["split"], "atomic_token_count": len(row["trajectory_signatures"]) * len(ATOMIC_GROUPS), "failure_signature": {key: value for key, value in row["failure_signature"].items() if key not in {"positive_authority", "typed_available"}}, "training_label": row["label"]} for row in train + dev]}
    visible["manifest_sha256"] = _sha256_json(visible)
    VISIBLE.write_text(json.dumps(visible, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    all_holdout_accepted = all(episode.get("episode_report", {}).get("status") == "accepted_evaluation" for target in targets["pg127_long_holdout"] for episode in target["episodes"])
    checks = {"fresh_checkpoint": True, "same_capacity_ablation": sum(parameter.numel() for parameter in model.parameters()) == sum(parameter.numel() for parameter in zero_model.parameters()), "atomic_feature_dim": ATOMIC_TOKEN_FEATURE_DIM == 16, "atomic_token_window": MAX_ATOMIC_TOKENS == 36, "all_groups_tokenized": len(ATOMIC_GROUPS) == 6, "seed_disjoint": not bool(set(train_seeds) & set(dev_seeds) or set(train_seeds) & set(holdout_seeds) or set(dev_seeds) & set(holdout_seeds)), "get_post_balanced": get_count == post_count == 36, "holdout_not_training": not set(holdout_seeds).intersection(train_seeds + dev_seeds), "six_step_replay_accepted": all_holdout_accepted, "full_accuracy_floor": weighted_holdout["metrics"]["accuracy"] >= 0.95, "full_compliance_floor": weighted_holdout["safety_compliance_rate"] >= 0.95, "all_surface_accuracy_floor": all(value["accuracy"] >= 0.95 for value in weighted_holdout["per_surface"].values()), "budget_accuracy_floor": weighted_holdout["budget_accuracy"] >= 0.95, "negative_false_stop_zero": weighted_holdout["negative_false_stop_count"] == 0, "full_non_abstain_nonzero": weighted_holdout["non_abstain_count"] > 0, "atomic_weight_ablation_changes_behavior": weighted_holdout["predictions"] != uniform_holdout["predictions"], "zero_ablation_changes_behavior": weighted_holdout["predictions"] != zeroed_holdout["predictions"], "memory_promotion_forbidden": True}
    report = {"protocol_id": "pg-pk-129-atomic-token-v1", "schema_version": "pg129-atomic-token-report-v1", "status": "completed_pg129_atomic_token_policy", "scope": {"model": "fresh_transformer_atomic_failure_trajectory_policy", "atomic_token_feature_dim": ATOMIC_TOKEN_FEATURE_DIM, "max_atomic_tokens": MAX_ATOMIC_TOKENS, "hidden_dim": HIDDEN_DIM, "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "device": str(device), "real_vulnerability_scanner_claim_allowed": False}, "training": {"train_count": len(train), "dev_count": len(dev), "history_tail": history[-5:], "zero_history_tail": zero_history[-5:], "dev": full_dev}, "holdout": {"weighted_atomic": weighted_holdout, "uniform_token_ablation": uniform_holdout, "weighted_atomic_zeroed": zeroed_holdout, "fresh_zero_baseline": zero_baseline}, "checks": checks, "tokenization": {"atomic_groups": list(ATOMIC_GROUPS), "per_token_fields": ["type_one_hot", "value_one_hot", "group_focus_weight", "trajectory_weight", "combined_token_weight", "position_current_scalar"], "one_failure_step_expands_to": len(ATOMIC_GROUPS), "future_tokens_masked": True, "raw_probe_or_response_tokenized": False, "oracle_authority_tokenized": False}, "diagnosis": {"representation_change": "把每个历史 failure step 拆为六个原子 token，并为每个 token 保留组权重、轨迹权重和位置。", "meaning": "fully tokenized failure-guided compositional safe replay action selection; not vulnerability confirmation", "training_sources": ["fresh PG-125 no-budget traces", "fresh PG-127 six-step traces"], "holdout_source": "fresh PG-127 unseen seeds and decoy strengths"}, "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "status": "candidate_atomic_token_holdout_passed_pending_manual_review_no_memory_promotion" if all(checks.values()) else "blocked_pg129_gate_failure_preserved", "reason": "所有字段 token 化不等于可以绕过 typed oracle；任何原子 token 消融失败都保留并阻断晋升。"}, "source": {"runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "policy": hashlib.sha256((ROOT / "app/pg129_atomic_token_policy.py").read_bytes()).hexdigest()}}
    report["report_sha256"] = _sha256_json(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": "pg-pk-129-atomic-token-v1", "schema_version": "pg129-atomic-token-protocol-v1", "objective": "将失败轨迹尽量细拆为原子 token，验证 token 权重和顺序对安全动作组装的贡献。", "token_contract": {"groups": list(ATOMIC_GROUPS), "tokens_per_step": len(ATOMIC_GROUPS), "max_steps": MAX_ATOMIC_TOKENS // len(ATOMIC_GROUPS), "feature_dim": ATOMIC_TOKEN_FEATURE_DIM, "weight_sources": ["group focus", "trajectory recency", "remaining probe budget"], "raw_fields_forbidden": ["raw_probe", "raw_response", "positive_authority", "typed_available", "evidence_hash", "target_id", "family"]}, "training": {"pg125_fresh_train_seeds": list(PG125_TRAIN_SEEDS), "pg127_long_train_seeds": list(PG127_TRAIN_SEEDS), "pg127_long_dev_seeds": list(PG127_DEV_SEEDS), "pg127_long_holdout_seeds": list(PG127_HOLDOUT_SEEDS), "holdout_never_in_training": True}, "evaluation": {"required_methods": ["GET", "POST"], "fresh_reset_per_action": True, "six_step_long_horizon": True, "positive_negative_unknown_surfaces": True, "weighted_vs_uniform_token_ablation": True, "gates": checks}, "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False}}
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    proposal = {"protocol_id": "pg-pk-129-atomic-token-v1", "proposal_id": "pg129-atomic-token-proposal-v1", "question": "把历史轨迹所有可用字段拆成原子 token 后，token 级权重能否帮助模型稳定拼装下一条探索路径？", "prediction": {"weighted_accuracy": ">=0.95", "weighted_budget_accuracy": ">=0.95", "negative_false_stop_count": 0, "uniform_token_ablation_changes": True}, "intervention": "six bounded failure groups become typed/value tokens; a Transformer uses group focus, trajectory weight, current-step marker and token position", "failure_rule": "若 token 化引入表面/标签泄漏或消融不改变行为，保留失败实验并禁止训练和记忆晋升。", "next": "若通过，接入第三独立实现、长于六步的滚动迷宫；若失败，做 token-level contribution 与工程/实验分诊。"}
    PROPOSAL.write_text(json.dumps(proposal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "atomic_token_feature_dim": ATOMIC_TOKEN_FEATURE_DIM, "max_atomic_tokens": MAX_ATOMIC_TOKENS, "parameter_count": report["scope"]["parameter_count"], "train": len(train), "dev": len(dev), "holdout": len(holdout), "weighted_accuracy": weighted_holdout["metrics"]["accuracy"], "weighted_compliance": weighted_holdout["safety_compliance_rate"], "budget_accuracy": weighted_holdout["budget_accuracy"], "uniform_token_accuracy": uniform_holdout["metrics"]["accuracy"], "zeroed_accuracy": zeroed_holdout["metrics"]["accuracy"], "all_gates": all(checks.values()), "report": str(REPORT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
