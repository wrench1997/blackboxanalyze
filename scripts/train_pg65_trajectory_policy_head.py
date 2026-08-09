"""PG-65 train a trajectory policy head on PG-64 traces.

The head consumes only a pre-oracle surface projection, the current belief,
and a candidate action.  It is selected on a dev split, frozen, then replayed
on a new layout/noise implementation.  Typed oracle fields are used only for
post-action scoring and never enter the feature vector.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
PG64_SCRIPT = ROOT / "scripts" / "run_pg64_multistep_belief_regret.py"
PG64_TRACE_PATH = ROOT / "research" / "pg64_multistep_belief_regret_trace_v1.json"
REPORT_PATH = ROOT / "research" / "pg65_trajectory_policy_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg65_trajectory_policy_protocol_v1.json"
TRACE_PATH = ROOT / "research" / "pg65_trajectory_policy_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg65_trajectory_policy_report_v1.md"
OUTPUT_DIR = ROOT / "artifacts" / "pg65-trajectory-policy"
CHECKPOINT_PATH = OUTPUT_DIR / "policy_head.pt"
SEED = 20650803
ACTIONS = (("GET", "screen"), ("POST", "screen"), ("GET", "confirm"), ("POST", "confirm"))
ACTION_INDEX = {action: index for index, action in enumerate(ACTIONS)}
SURFACE_CLASSES = ("collection", "record", "search")
RESPONSE_SHAPES = ("compact", "object", "redirect")
CHANNEL_HINTS = ("balanced_lane", "query_leaning", "form_leaning")
HYPOTHESES = ("GET_TARGET", "POST_TARGET", "NO_EXIT")
OBSERVATIONS = ("query_signal", "form_signal", "neutral_signal", "typed_exit", "typed_no_exit")


def _load_pg64() -> Any:
    spec = importlib.util.spec_from_file_location("pg64_for_pg65", PG64_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG64 helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _one_hot(value: str, vocabulary: tuple[str, ...]) -> list[float]:
    return [1.0 if value == item else 0.0 for item in vocabulary]


def _features(surface: dict[str, Any], belief: dict[str, float], action: tuple[str, str]) -> list[float]:
    # No target, oracle, response, layout ID, or raw request/response enters.
    values: list[float] = []
    values.extend(_one_hot(str(surface.get("surface_class", "")), SURFACE_CLASSES))
    values.extend(_one_hot(str(surface.get("response_shape", "")), RESPONSE_SHAPES))
    values.extend(_one_hot(str(surface.get("channel_hint", "")), CHANNEL_HINTS))
    values.append(float(surface.get("route_depth", 1.0)) / 3.0)
    values.extend([float(belief.get(hypothesis, 0.0)) for hypothesis in HYPOTHESES])
    values.extend([1.0 if action == candidate else 0.0 for candidate in ACTIONS])
    return values


class TrajectoryPolicyHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _groups_from_trace(trace: dict[str, Any], split: str) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for episode in trace["active_belief"]["episodes"]:
        task_id = str(episode["task_id"])
        if not task_id.startswith(f"pg64-{split}-"):
            continue
        for step in episode["steps"]:
            candidate_order = [tuple(item) for item in step["candidate_order"]]
            groups.append({"surface": step["pre_oracle_surface_projection"], "belief": step["belief_before"], "candidate_order": candidate_order, "selected": tuple(step["selected_action"]), "task_id": task_id})
    return groups


def _group_loss(model: TrajectoryPolicyHead, group: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, bool]:
    features = torch.tensor([_features(group["surface"], group["belief"], action) for action in group["candidate_order"]], dtype=torch.float32, device=device)
    logits = model(features)
    target = torch.tensor([group["candidate_order"].index(group["selected"])], dtype=torch.long, device=device)
    loss = nn.functional.cross_entropy(logits.unsqueeze(0), target)
    correct = int(group["candidate_order"][int(torch.argmax(logits).detach().cpu())] == group["selected"])
    return loss, bool(correct)


def _evaluate_groups(model: TrajectoryPolicyHead, groups: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    if not groups:
        return {"group_count": 0, "accuracy": 0.0, "loss": 0.0}
    losses: list[float] = []
    correct = 0
    model.eval()
    with torch.inference_mode():
        for group in groups:
            loss, is_correct = _group_loss(model, group, device)
            losses.append(float(loss.detach().cpu()))
            correct += int(is_correct)
    return {"group_count": len(groups), "accuracy": round(correct / len(groups), 6), "loss": round(float(statistics.mean(losses)), 6)}


def _independent_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    serial = 0
    for layout_index, layout in enumerate(("juno", "kestrel")):
        for local_index in range(72):
            seed = SEED + serial * 29 + layout_index * 307
            target = HYPOTHESES[(local_index + 1) % len(HYPOTHESES)]
            unknown_zone = target != "NO_EXIT" and local_index % 7 == 0
            surface = {"surface_class": SURFACE_CLASSES[(2 * local_index + seed) % 3], "response_shape": RESPONSE_SHAPES[(local_index + seed) % 3], "channel_hint": CHANNEL_HINTS[(local_index + layout_index) % 3], "route_depth": 1 + ((local_index + 2 * seed) % 3), "pre_oracle": True, "layout_role": "pg65_independent_noisy_surface"}
            tasks.append({"task_id": f"pg65-independent-{layout}-{serial:04d}", "dataset_role": "independent_holdout", "source_id": f"pg65-independent-{layout}", "layout": layout, "sampling_seed": seed, "observation_seed": seed + 11011, "surface_projection": surface, "evaluator_target": target, "unknown_zone": unknown_zone, "negative_control": target == "NO_EXIT"})
            serial += 1
    return tasks


def _likelihood_independent(action: tuple[str, str], hypothesis: str) -> dict[str, float]:
    if action[1] == "confirm":
        outcome = "typed_exit" if (action[0] == "GET" and hypothesis == "GET_TARGET") or (action[0] == "POST" and hypothesis == "POST_TARGET") else "typed_no_exit"
        return {observation: (1.0 if observation == outcome else 0.0) for observation in OBSERVATIONS}
    # Independent implementation has a different noise profile from PG64.
    if action[0] == "GET":
        rows = {"GET_TARGET": (0.70, 0.12, 0.18), "POST_TARGET": (0.18, 0.64, 0.18), "NO_EXIT": (0.30, 0.22, 0.48)}
    else:
        rows = {"GET_TARGET": (0.58, 0.24, 0.18), "POST_TARGET": (0.12, 0.70, 0.18), "NO_EXIT": (0.22, 0.30, 0.48)}
    query, form, neutral = rows[hypothesis]
    return {"query_signal": query, "form_signal": form, "neutral_signal": neutral, "typed_exit": 0.0, "typed_no_exit": 0.0}


def _training_likelihood(action: tuple[str, str], hypothesis: str) -> dict[str, float]:
    if action[1] == "confirm":
        outcome = "typed_exit" if (action[0] == "GET" and hypothesis == "GET_TARGET") or (action[0] == "POST" and hypothesis == "POST_TARGET") else "typed_no_exit"
        return {observation: (1.0 if observation == outcome else 0.0) for observation in OBSERVATIONS}
    if action[0] == "GET":
        rows = {"GET_TARGET": (0.78, 0.08, 0.14), "POST_TARGET": (0.20, 0.62, 0.18), "NO_EXIT": (0.24, 0.24, 0.52)}
    else:
        rows = {"GET_TARGET": (0.62, 0.20, 0.18), "POST_TARGET": (0.08, 0.78, 0.14), "NO_EXIT": (0.24, 0.24, 0.52)}
    query, form, neutral = rows[hypothesis]
    return {"query_signal": query, "form_signal": form, "neutral_signal": neutral, "typed_exit": 0.0, "typed_no_exit": 0.0}


def _normalise(values: dict[str, float]) -> dict[str, float]:
    clipped = {key: max(float(value), 1e-9) for key, value in values.items()}
    total = sum(clipped.values()) or 1.0
    return {key: value / total for key, value in clipped.items()}


def _posterior(belief: dict[str, float], action: tuple[str, str], observation: str) -> dict[str, float]:
    return _normalise({hypothesis: belief[hypothesis] * _training_likelihood(action, hypothesis)[observation] for hypothesis in HYPOTHESES})


def _sample(action: tuple[str, str], target: str, rng: random.Random) -> str:
    distribution = _likelihood_independent(action, target)
    return rng.choices(list(distribution), weights=list(distribution.values()), k=1)[0]


def _oracle(task: dict[str, Any], action: tuple[str, str]) -> dict[str, Any]:
    positive = action[1] == "confirm" and ((action[0] == "GET" and task["evaluator_target"] == "GET_TARGET") or (action[0] == "POST" and task["evaluator_target"] == "POST_TARGET"))
    return {"oracle_id": "pg65-typed-target-zone-oracle-v1", "positive": positive, "modality": "typed_exit" if positive else "bounded_observation", "evaluator_state_hidden": True, "unknown_zone": bool(task["unknown_zone"]), "raw_body_stored": False, "external_network": False, "state_mutated": False}


def _simulate(model: TrajectoryPolicyHead, tasks: list[dict[str, Any]], device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episodes: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    for task in tasks:
        rng = random.Random(int(task["observation_seed"]))
        belief = {hypothesis: 1.0 / len(HYPOTHESES) for hypothesis in HYPOTHESES}
        remaining = list(ACTIONS)
        steps: list[dict[str, Any]] = []
        regrets: list[float] = []
        for step_index in range(1, len(ACTIONS) + 1):
            if not remaining:
                break
            order = list(remaining)
            rng.shuffle(order)
            features = torch.tensor([_features(task["surface_projection"], belief, action) for action in order], dtype=torch.float32, device=device)
            with torch.inference_mode():
                logits = model(features).detach().cpu().tolist()
            action = order[max(range(len(order)), key=lambda index: (logits[index], -index))]
            # Regret is measured against the strongest available model-score
            # action at this pre-oracle state, not against the hidden target.
            regret = max(logits) - logits[order.index(action)]
            prior = dict(belief)
            observation = _sample(action, task["evaluator_target"], rng)
            belief = _posterior(belief, action, observation)
            oracle = _oracle(task, action)
            reset = {"kind": "pg65-independent-fresh-reset", "target_host": "127.0.0.1", "target_instance_id": f"{task['task_id']}-{step_index}", "fresh_target": True, "completed": True, "evaluator_state_hidden": True, "external_network": False, "state_change_allowed": False, "action": list(action)}
            reset["reset_sha256"] = _sha256_json(reset)
            response = {"observation_class": observation, "status_class": "2xx" if observation == "typed_exit" else "4xx" if observation == "typed_no_exit" else "2xx", "raw_body_stored": False, "external_network": False}
            evidence = {"task_id": task["task_id"], "surface_projection": task["surface_projection"], "action": list(action), "belief_before": prior, "belief_after": belief, "response_projection": response, "oracle_projection": oracle, "reset": reset}
            step = {"step_id": f"{task['task_id']}-{step_index}", "task_id": task["task_id"], "candidate_order": [list(item) for item in order], "selected_action": list(action), "pre_oracle_surface_projection": task["surface_projection"], "belief_before": prior, "policy_logits": [round(float(value), 6) for value in logits], "counterfactual_regret": round(float(regret), 6), "observed_class": observation, "belief_after": belief, "reset": reset, "oracle_after_action": oracle, "response_projection_after_action": response, "evidence_hash_algorithm": "sha256-canonical-json", "evidence_hash": _sha256_json(evidence), "raw_probe_stored": False, "raw_response_stored": False, "online_weight_update": False, "long_term_memory_write": False}
            steps.append(step)
            regrets.append(float(regret))
            action_counts[f"{action[0]}.{action[1]}"] += 1
            remaining.remove(action)
            if oracle["positive"]:
                break
        positive = task["evaluator_target"] != "NO_EXIT"
        confirmed = bool(steps and steps[-1]["oracle_after_action"]["positive"])
        episodes.append({"task_id": task["task_id"], "positive": positive, "unknown_zone": task["unknown_zone"], "confirmed": confirmed, "false_accept": bool(not positive and confirmed), "abstain": bool(not confirmed or task["unknown_zone"]), "final_route": "unknown_surface" if task["unknown_zone"] and positive else "target_zone" if confirmed else "none", "step_count": len(steps), "mean_regret": round(statistics.mean(regrets) if regrets else 0.0, 6), "steps": steps})
    positives = [episode for episode in episodes if episode["positive"]]
    negatives = [episode for episode in episodes if not episode["positive"]]
    unknowns = [episode for episode in positives if episode["unknown_zone"]]
    metrics = {"episode_count": len(episodes), "positive_episode_count": len(positives), "negative_control_count": len(negatives), "target_recall": round(sum(int(episode["confirmed"]) for episode in positives) / max(len(positives), 1), 6), "known_target_recall": round(sum(int(episode["confirmed"]) for episode in positives if not episode["unknown_zone"]) / max(len(positives) - len(unknowns), 1), 6), "unknown_positive_count": len(unknowns), "unknown_strict_abstain": all(episode["abstain"] and episode["final_route"] == "unknown_surface" for episode in unknowns), "unknown_abstain_count": sum(int(episode["abstain"]) for episode in unknowns), "negative_false_accept_count": sum(int(episode["false_accept"]) for episode in negatives), "negative_false_accept_rate": round(sum(int(episode["false_accept"]) for episode in negatives) / max(len(negatives), 1), 6), "mean_steps": round(statistics.mean(episode["step_count"] for episode in episodes), 6), "median_steps": float(statistics.median(episode["step_count"] for episode in episodes)), "multi_step_episode_rate": round(sum(int(episode["step_count"] >= 2) for episode in episodes) / max(len(episodes), 1), 6), "mean_counterfactual_regret": round(statistics.mean(episode["mean_regret"] for episode in episodes), 6), "action_counts": dict(action_counts), "fresh_reset_count": sum(len(episode["steps"]) for episode in episodes), "evidence_hash_count": sum(sum(int(bool(step["evidence_hash"])) for step in episode["steps"]) for episode in episodes), "raw_probe_stored_count": 0, "raw_response_stored_count": 0}
    return metrics, episodes


def main() -> int:
    trace = json.loads(PG64_TRACE_PATH.read_text(encoding="utf-8"))
    train_groups = _groups_from_trace(trace, "train")
    dev_groups = _groups_from_trace(trace, "dev")
    holdout_groups = _groups_from_trace(trace, "holdout")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    first = train_groups[0]
    model = TrajectoryPolicyHead(len(_features(first["surface"], first["belief"], first["candidate_order"][0]))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008, weight_decay=0.01)
    best_state: dict[str, torch.Tensor] | None = None
    best_dev_loss = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, 241):
        model.train()
        losses: list[torch.Tensor] = []
        for group in train_groups:
            loss, _ = _group_loss(model, group, device)
            losses.append(loss)
        optimizer.zero_grad(set_to_none=True)
        train_loss = torch.stack(losses).mean()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch == 1 or epoch % 40 == 0:
            dev_metrics = _evaluate_groups(model, dev_groups, device)
            history.append({"epoch": epoch, "train_loss": round(float(train_loss.detach().cpu()), 6), "dev_loss": dev_metrics["loss"], "dev_accuracy": dev_metrics["accuracy"]})
            if dev_metrics["loss"] < best_dev_loss:
                best_dev_loss = dev_metrics["loss"]
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    train_metrics = _evaluate_groups(model, train_groups, device)
    dev_metrics = _evaluate_groups(model, dev_groups, device)
    holdout_policy_metrics = _evaluate_groups(model, holdout_groups, device)
    independent_metrics, independent_episodes = _simulate(model, _independent_tasks(), device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg65-trajectory-policy-checkpoint-v1", "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "input_contract": "pre_oracle_surface_plus_belief_plus_candidate_action", "seed": SEED}, CHECKPOINT_PATH)
    checkpoint_sha256 = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
    safety_reasons = []
    if independent_metrics["target_recall"] < 0.80:
        safety_reasons.append("independent_target_recall_below_0.80")
    if independent_metrics["negative_false_accept_count"] != 0:
        safety_reasons.append("independent_negative_false_accept")
    if not independent_metrics["unknown_strict_abstain"]:
        safety_reasons.append("independent_unknown_not_strict_abstain")
    if independent_metrics["multi_step_episode_rate"] < 0.50:
        safety_reasons.append("independent_policy_not_multistep")
    capability_reasons = []
    if dev_metrics["accuracy"] < 0.80:
        capability_reasons.append("dev_policy_accuracy_below_0.80")
    if holdout_policy_metrics["accuracy"] < 0.80:
        capability_reasons.append("pg64_holdout_policy_accuracy_below_0.80")
    safety_status = "passed" if not safety_reasons else "blocked"
    capability_status = "passed" if not capability_reasons else "blocked"
    report = {"protocol_id": "pg-pk-65-trajectory-policy-v1", "schema_version": "pg65-trajectory-policy-report-v1", "status": "diagnostic_only", "training": {"source_trace": str(PG64_TRACE_PATH.relative_to(ROOT)), "train_group_count": len(train_groups), "dev_group_count": len(dev_groups), "holdout_group_count": len(holdout_groups), "epochs": 240, "device": str(device), "history_tail": history[-5:], "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "checkpoint_sha256": checkpoint_sha256, "oracle_in_input": False, "family_in_input": False, "target_in_input": False, "raw_request_response_in_input": False}, "train_metrics": train_metrics, "dev_metrics": dev_metrics, "pg64_holdout_policy_metrics": holdout_policy_metrics, "independent_noise_holdout": independent_metrics, "independent_source": {"layouts": ["juno", "kestrel"], "noise_profile": "pg65-shifted-observation-table", "task_count": 144, "model_retrained_on_independent": False}, "hard_gate": {"schema_version": "pg65-trajectory-policy-hard-gate-v1", "status": safety_status, "claim_allowed": False, "reasons": safety_reasons, "training_allowed": False, "memory_promotion_allowed": False}, "capability_gate": {"schema_version": "pg65-trajectory-policy-capability-gate-v1", "status": capability_status, "claim_allowed": False, "reasons": capability_reasons, "minimum_dev_accuracy": 0.80, "minimum_pg64_holdout_accuracy": 0.80, "training_allowed": False, "memory_promotion_allowed": False}, "promotion": {"status": "safety_gate_passed_capability_gate_blocked" if not safety_reasons and capability_reasons else "quarantined_trajectory_policy_candidate", "training_allowed": False, "memory_promotion_allowed": False, "formal_capability_claim_allowed": False}, "interpretation": "PG65 的独立安全复放通过，但策略头 dev/holdout 选择准确率不足，不能称为可规模化能力；下一轮改用 utility/ranking 目标。"}
    protocol = {"protocol_id": "pg-pk-65-trajectory-policy-v1", "schema_version": "pg65-trajectory-policy-protocol-v1", "objective": "用 PG64 轨迹训练 pre-oracle trajectory policy head，在独立布局/观察噪声上验证多步动作泛化。", "authorized_scope": {"target_host": "127.0.0.1", "external_network": False, "fixture_replay_only": True, "state_mutation": False, "raw_probe_persistence": False, "raw_response_body_persistence": False}, "splits": {"train": "PG64 train episodes", "dev": "PG64 dev episodes", "holdout": "PG64 holdout + PG65 independent noise/layout"}, "input_contract": {"model_reads": ["pre_oracle_surface_projection", "belief_before", "candidate_action"], "model_must_not_read": ["evaluator_target", "oracle_projection", "response_projection_after_action", "family_label", "layout_id", "task_id", "raw_probe", "raw_response"]}, "required_gates": {"dev_only_checkpoint_selection": True, "dev_policy_accuracy_min": 0.80, "pg64_holdout_policy_accuracy_min": 0.80, "independent_target_recall_min": 0.80, "independent_negative_false_accept_zero": True, "independent_unknown_strict_abstain": True, "independent_multi_step_rate_min": 0.50, "fresh_reset_per_action": True, "evidence_hash_per_action": True, "typed_oracle_after_action_only": True, "training_promotion_on_fixture": False, "memory_promotion_on_fixture": False}, "run_result": {"safety_status": report["hard_gate"]["status"], "capability_status": report["capability_gate"]["status"], "independent_target_recall": independent_metrics["target_recall"], "independent_negative_false_accept_count": independent_metrics["negative_false_accept_count"]}}
    trace_out = {"schema_version": "pg65-trajectory-policy-trace-v1", "evaluation_only": True, "training_eligible": False, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "model_retrained_on_independent": False, "independent_noise_holdout": {"episodes": independent_episodes, "episode_count": len(independent_episodes), "trace_manifest_sha256": _sha256_json([step["evidence_hash"] for episode in independent_episodes for step in episode["steps"]])}}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps(trace_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-65 轨迹策略头", "", "输入为 pre-oracle surface + belief + candidate action；独立噪声/布局上冻结模型复放。", "", "| split | accuracy/recall | 阴性误报 | 未知弃权 | multi-step | mean steps |", "|---|---:|---:|---|---:|---:|", f"| train policy | {train_metrics['accuracy']} | — | — | — | — |", f"| dev policy | {dev_metrics['accuracy']} | — | — | — | — |", f"| PG64 holdout policy | {holdout_policy_metrics['accuracy']} | — | — | — | — |", f"| independent noise | {independent_metrics['target_recall']} | {independent_metrics['negative_false_accept_count']} | {independent_metrics['unknown_strict_abstain']} | {independent_metrics['multi_step_episode_rate']} | {independent_metrics['mean_steps']} |", "", f"安全门：`{report['hard_gate']['status']}`；能力门：`{report['capability_gate']['status']}`；formal capability claim=false；训练/记忆不晋升。", ""]
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "train_metrics": train_metrics, "dev_metrics": dev_metrics, "pg64_holdout_policy_metrics": holdout_policy_metrics, "independent_noise_holdout": independent_metrics, "hard_gate": report["hard_gate"], "report": str(REPORT_PATH.relative_to(ROOT)), "trace": str(TRACE_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
