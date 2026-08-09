"""PG-66 train a utility/ranking trajectory head.

PG-65 used the action selected by a heuristic controller as a hard class
label, which made near-ties and candidate-order choices look like label noise.
PG-66 supervises every candidate with a continuous pre-oracle utility and a
pairwise ranking loss, then replays the frozen head on PG-65's independent
noise/layout fixture.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import statistics
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
PG64_SCRIPT = ROOT / "scripts" / "run_pg64_multistep_belief_regret.py"
PG65_SCRIPT = ROOT / "scripts" / "train_pg65_trajectory_policy_head.py"
PG64_TRACE_PATH = ROOT / "research" / "pg64_multistep_belief_regret_trace_v1.json"
REPORT_PATH = ROOT / "research" / "pg66_utility_ranking_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg66_utility_ranking_protocol_v1.json"
TRACE_PATH = ROOT / "research" / "pg66_utility_ranking_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg66_utility_ranking_report_v1.md"
OUTPUT_DIR = ROOT / "artifacts" / "pg66-utility-ranking"
CHECKPOINT_PATH = OUTPUT_DIR / "ranking_head.pt"
SEED = 20660803
EPOCHS = 100


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _groups(trace: dict[str, Any], split: str) -> list[dict[str, Any]]:
    pg65 = _load(PG65_SCRIPT, f"pg65_groups_{split}")
    return pg65._groups_from_trace(trace, split)


def _target_utilities(pg64: Any, group: dict[str, Any]) -> torch.Tensor:
    return torch.tensor([float(pg64._utility(action, group["belief"])) for action in group["candidate_order"]], dtype=torch.float32)


def _ranking_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = nn.functional.mse_loss(predicted, target)
    pair_losses: list[torch.Tensor] = []
    for left in range(len(target)):
        for right in range(len(target)):
            delta = float(target[left] - target[right])
            if abs(delta) < 1e-6:
                continue
            direction = 1.0 if delta > 0 else -1.0
            pair_losses.append(nn.functional.relu(0.03 - direction * (predicted[left] - predicted[right])))
    pair = torch.stack(pair_losses).mean() if pair_losses else predicted.sum() * 0.0
    return mse + 0.35 * pair


def _batch_tensors(pg64: Any, pg65: Any, groups: list[dict[str, Any]], device: torch.device, *, all_actions: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    action_order = list(pg65.ACTIONS)
    rows: list[list[list[float]]] = []
    targets: list[list[float]] = []
    masks: list[list[bool]] = []
    for group in groups:
        actions = action_order if all_actions else list(group["candidate_order"])
        rows.append([pg65._features(group["surface"], group["belief"], action) for action in actions])
        targets.append([float(pg64._utility(action, group["belief"])) for action in actions])
        masks.append([True] * len(actions))
    width = max((len(row) for row in rows), default=1)
    feature_dim = len(rows[0][0]) if rows and rows[0] else len(pg65._features(groups[0]["surface"], groups[0]["belief"], action_order[0]))
    feature_tensor = torch.zeros((len(rows), width, feature_dim), dtype=torch.float32)
    target_tensor = torch.zeros((len(rows), width), dtype=torch.float32)
    mask_tensor = torch.zeros((len(rows), width), dtype=torch.bool)
    for index, row in enumerate(rows):
        feature_tensor[index, :len(row)] = torch.tensor(row, dtype=torch.float32)
        target_tensor[index, :len(row)] = torch.tensor(targets[index], dtype=torch.float32)
        mask_tensor[index, :len(row)] = True
    return feature_tensor.to(device), target_tensor.to(device), mask_tensor.to(device)


def _batch_ranking_loss(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask
    mse = ((predicted - target) ** 2)[valid].mean()
    target_delta = target.unsqueeze(2) - target.unsqueeze(1)
    direction = torch.sign(target_delta)
    pair_mask = valid.unsqueeze(2) & valid.unsqueeze(1) & (direction.abs() > 0)
    predicted_delta = predicted.unsqueeze(2) - predicted.unsqueeze(1)
    pair = torch.relu(0.03 - direction * predicted_delta)[pair_mask].mean()
    return mse + 0.35 * pair


def _evaluate(model: nn.Module, pg64: Any, pg65: Any, groups: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    if not groups:
        return {"group_count": 0, "ranking_accuracy": 0.0, "mean_mse": 0.0, "ranking_loss": 0.0}
    features, targets, mask = _batch_tensors(pg64, pg65, groups, device, all_actions=False)
    with torch.inference_mode():
        predictions = model(features.reshape(-1, features.shape[-1])).reshape(features.shape[:2])
    masked_predictions = predictions.masked_fill(~mask, -1e9)
    masked_targets = targets.masked_fill(~mask, -1e9)
    predicted_index = torch.argmax(masked_predictions, dim=1)
    target_index = torch.argmax(masked_targets, dim=1)
    ranking_correct = int((predicted_index == target_index).sum().cpu())
    sorted_targets = torch.sort(masked_targets, dim=1, descending=True).values
    decisive = (sorted_targets[:, 0] - sorted_targets[:, 1]) >= 0.03
    decisive_count = int(decisive.sum().cpu())
    decisive_correct = int(((predicted_index == target_index) & decisive).sum().cpu())
    selected_target = masked_targets.gather(1, predicted_index.unsqueeze(1)).squeeze(1)
    regret = (sorted_targets[:, 0] - selected_target).clamp_min(0.0)
    mse = ((predictions - targets) ** 2)[mask].mean()
    return {"group_count": len(groups), "ranking_accuracy": round(ranking_correct / max(len(groups), 1), 6), "decisive_group_count": decisive_count, "decisive_ranking_accuracy": round(decisive_correct / max(decisive_count, 1), 6), "tie_margin": 0.03, "mean_utility_regret": round(float(regret.mean().cpu()), 6), "mean_mse": round(float(mse.cpu()), 6), "ranking_loss": round(float(_batch_ranking_loss(predictions, targets, mask).cpu()), 6)}


def main() -> int:
    pg64 = _load(PG64_SCRIPT, "pg64_for_pg66")
    pg65 = _load(PG65_SCRIPT, "pg65_for_pg66")
    trace = json.loads(PG64_TRACE_PATH.read_text(encoding="utf-8"))
    train_groups = _groups(trace, "train")
    dev_groups = _groups(trace, "dev")
    holdout_groups = _groups(trace, "holdout")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    first = train_groups[0]
    model = pg65.TrajectoryPolicyHead(len(pg65._features(first["surface"], first["belief"], first["candidate_order"][0]))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    best_state: dict[str, torch.Tensor] | None = None
    best_dev_loss = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        features, targets, mask = _batch_tensors(pg64, pg65, train_groups, device, all_actions=True)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(features.reshape(-1, features.shape[-1])).reshape(features.shape[:2])
        train_loss = _batch_ranking_loss(predictions, targets, mask)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch == 1 or epoch % 20 == 0:
            dev_metrics = _evaluate(model, pg64, pg65, dev_groups, device)
            history.append({"epoch": epoch, "train_loss": round(float(train_loss.detach().cpu()), 6), "dev_ranking_accuracy": dev_metrics["ranking_accuracy"], "dev_ranking_loss": dev_metrics["ranking_loss"]})
            if dev_metrics["ranking_loss"] < best_dev_loss:
                best_dev_loss = dev_metrics["ranking_loss"]
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    train_metrics = _evaluate(model, pg64, pg65, train_groups, device)
    dev_metrics = _evaluate(model, pg64, pg65, dev_groups, device)
    holdout_metrics = _evaluate(model, pg64, pg65, holdout_groups, device)
    independent_metrics, independent_episodes = pg65._simulate(model, pg65._independent_tasks(), device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg66-utility-ranking-checkpoint-v1", "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "input_contract": "pre_oracle_surface_plus_belief_plus_candidate_action", "target": "continuous_pre_oracle_utility", "seed": SEED}, CHECKPOINT_PATH)
    checkpoint_sha256 = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
    safety_reasons: list[str] = []
    for field, label in (("target_recall", "independent_target_recall_below_0.80"), ("negative_false_accept_count", "independent_negative_false_accept"), ("multi_step_episode_rate", "independent_policy_not_multistep")):
        if (independent_metrics[field] < 0.80 if field == "target_recall" else independent_metrics[field] != 0 if field == "negative_false_accept_count" else independent_metrics[field] < 0.50):
            safety_reasons.append(label)
    if not independent_metrics["unknown_strict_abstain"]:
        safety_reasons.append("independent_unknown_not_strict_abstain")
    capability_reasons: list[str] = []
    if dev_metrics["decisive_ranking_accuracy"] < 0.80:
        capability_reasons.append("dev_decisive_ranking_accuracy_below_0.80")
    if holdout_metrics["decisive_ranking_accuracy"] < 0.80:
        capability_reasons.append("pg64_holdout_decisive_ranking_accuracy_below_0.80")
    if dev_metrics["mean_utility_regret"] > 0.02:
        capability_reasons.append("dev_mean_utility_regret_above_0.02")
    if holdout_metrics["mean_utility_regret"] > 0.02:
        capability_reasons.append("pg64_holdout_mean_utility_regret_above_0.02")
    report = {"protocol_id": "pg-pk-66-utility-ranking-v1", "schema_version": "pg66-utility-ranking-report-v1", "status": "diagnostic_only", "training": {"source_trace": str(PG64_TRACE_PATH.relative_to(ROOT)), "train_group_count": len(train_groups), "dev_group_count": len(dev_groups), "holdout_group_count": len(holdout_groups), "epochs": EPOCHS, "device": str(device), "history_tail": history[-5:], "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "checkpoint_sha256": checkpoint_sha256, "oracle_in_input": False, "family_in_input": False, "target_in_input": False, "label_type": "continuous_pre_oracle_utility_plus_pairwise_ranking"}, "train_metrics": train_metrics, "dev_metrics": dev_metrics, "pg64_holdout_metrics": holdout_metrics, "independent_noise_holdout": independent_metrics, "safety_gate": {"status": "passed" if not safety_reasons else "blocked", "reasons": safety_reasons, "claim_allowed": False, "training_allowed": False, "memory_promotion_allowed": False}, "capability_gate": {"status": "passed" if not capability_reasons else "blocked", "reasons": capability_reasons, "tie_margin": 0.03, "minimum_decisive_ranking_accuracy": 0.80, "maximum_mean_utility_regret": 0.02, "claim_allowed": False, "training_allowed": False, "memory_promotion_allowed": False}, "promotion": {"status": "utility_ranking_safety_passed_capability_passed_no_promotion" if not safety_reasons and not capability_reasons else "utility_ranking_quarantined", "training_allowed": False, "memory_promotion_allowed": False, "formal_capability_claim_allowed": False}, "interpretation": "PG66 用连续 pre-oracle utility 替代单一选择标签；显式忽略 utility 差小于 0.03 的不可区分平局，避免把平局当作能力错误。"}
    protocol = {"protocol_id": "pg-pk-66-utility-ranking-v1", "schema_version": "pg66-utility-ranking-protocol-v1", "objective": "用每个候选动作的连续 pre-oracle utility 和 pairwise ranking 训练轨迹策略头，解决 PG65 选择标签噪声。", "authorized_scope": {"target_host": "127.0.0.1", "external_network": False, "fixture_replay_only": True, "state_mutation": False, "raw_probe_persistence": False, "raw_response_body_persistence": False}, "input_contract": {"model_reads": ["pre_oracle_surface_projection", "belief_before", "candidate_action"], "model_must_not_read": ["evaluator_target", "oracle_projection", "response_projection_after_action", "family_label", "layout_id", "task_id", "raw_probe", "raw_response"]}, "label_contract": {"continuous_utility_is_pre_oracle": True, "pairwise_ranking_loss": True, "selected_action_hard_label_not_used": True, "tie_margin": 0.03}, "required_gates": {"dev_only_checkpoint_selection": True, "dev_decisive_ranking_accuracy_min": 0.80, "pg64_holdout_decisive_ranking_accuracy_min": 0.80, "maximum_mean_utility_regret": 0.02, "independent_target_recall_min": 0.80, "independent_negative_false_accept_zero": True, "independent_unknown_strict_abstain": True, "independent_multi_step_rate_min": 0.50, "fresh_reset_per_action": True, "evidence_hash_per_action": True, "typed_oracle_after_action_only": True, "training_promotion_on_fixture": False, "memory_promotion_on_fixture": False}, "run_result": {"safety_status": report["safety_gate"]["status"], "capability_status": report["capability_gate"]["status"], "independent_target_recall": independent_metrics["target_recall"]}}
    trace_out = {"schema_version": "pg66-utility-ranking-trace-v1", "evaluation_only": True, "training_eligible": False, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "model_retrained_on_independent": False, "independent_noise_holdout": {"episodes": independent_episodes, "episode_count": len(independent_episodes), "trace_manifest_sha256": _sha256_json([step["evidence_hash"] for episode in independent_episodes for step in episode["steps"]])}}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps(trace_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-66 utility/ranking 轨迹策略头", "", "每个 candidate 使用连续 pre-oracle utility + pairwise ranking；不使用硬选择标签。", "", "| split | 全体 ranking | 明确优势 ranking | mean regret | MSE |", "|---|---:|---:|---:|---:|", f"| train | {train_metrics['ranking_accuracy']} | {train_metrics['decisive_ranking_accuracy']} | {train_metrics['mean_utility_regret']} | {train_metrics['mean_mse']} |", f"| dev | {dev_metrics['ranking_accuracy']} | {dev_metrics['decisive_ranking_accuracy']} | {dev_metrics['mean_utility_regret']} | {dev_metrics['mean_mse']} |", f"| PG64 holdout | {holdout_metrics['ranking_accuracy']} | {holdout_metrics['decisive_ranking_accuracy']} | {holdout_metrics['mean_utility_regret']} | {holdout_metrics['mean_mse']} |", f"| independent noise | — | — | — | — |", "", f"安全门：`{report['safety_gate']['status']}`；能力门：`{report['capability_gate']['status']}`；训练/记忆不晋升。", ""]
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "train_metrics": train_metrics, "dev_metrics": dev_metrics, "pg64_holdout_metrics": holdout_metrics, "independent_noise_holdout": independent_metrics, "safety_gate": report["safety_gate"], "capability_gate": report["capability_gate"], "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
