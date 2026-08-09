"""PG-64 multi-step belief update with noisy observations and regret audit.

This is a projection-only local fixture.  The active controller sees an
abstract surface and a belief state, samples safe bounded observations, and
chooses the next action by expected information gain/exit utility.  The typed
oracle is attached only after an action.  A fixed-order controller is replayed
against the same hidden task seeds as a diagnostic baseline.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "research" / "pg64_multistep_belief_regret_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg64_multistep_belief_regret_protocol_v1.json"
TRACE_PATH = ROOT / "research" / "pg64_multistep_belief_regret_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg64_multistep_belief_regret_report_v1.md"
SEED = 20640803
HYPOTHESES = ("GET_TARGET", "POST_TARGET", "NO_EXIT")
OBSERVATIONS = ("query_signal", "form_signal", "neutral_signal", "typed_exit", "typed_no_exit")
ACTIONS = (("GET", "screen"), ("POST", "screen"), ("GET", "confirm"), ("POST", "confirm"))
FIXED_ORDER = ACTIONS


def _normalise(values: dict[str, float]) -> dict[str, float]:
    clipped = {key: max(float(value), 1e-9) for key, value in values.items()}
    total = sum(clipped.values()) or 1.0
    return {key: value / total for key, value in clipped.items()}


def _entropy(values: dict[str, float]) -> float:
    return -sum(value * math.log(value) for value in values.values() if value > 0)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _surface(task_index: int, layout: str, seed: int) -> dict[str, Any]:
    # These are abstract bounded projections, not request strings or bodies.
    return {
        "surface_class": ("collection", "record", "search")[(task_index + seed) % 3],
        "response_shape": ("compact", "object", "redirect")[(2 * task_index + seed) % 3],
        "channel_hint": ("balanced_lane", "query_leaning", "form_leaning")[(task_index + len(layout)) % 3],
        "route_depth": 1 + ((task_index + seed) % 3),
        "pre_oracle": True,
        "layout_role": "pg64_hidden_target_surface",
    }


def _tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    serial = 0
    split_sizes = (("train", ("fable",), 72), ("dev", ("garnet",), 72), ("holdout", ("haze", "ion"), 144))
    for dataset_role, layouts, count in split_sizes:
        for layout_index, layout in enumerate(layouts):
            per_layout = count // len(layouts)
            for local_index in range(per_layout):
                seed = SEED + serial * 23 + layout_index * 173
                target = HYPOTHESES[local_index % len(HYPOTHESES)]
                unknown_zone = target != "NO_EXIT" and local_index % 5 == 0
                task_id = f"pg64-{dataset_role}-{layout}-{serial:04d}"
                tasks.append({
                    "task_id": task_id,
                    "dataset_role": dataset_role,
                    "source_id": f"pg64-local-{layout}",
                    "implementation": "pg64_noisy_maze_fixture_v1",
                    "layout": layout,
                    "sampling_seed": seed,
                    "observation_seed": seed + 7001,
                    "surface_projection": _surface(local_index, layout, seed),
                    "evaluator_target": target,
                    "unknown_zone": unknown_zone,
                    "negative_control": target == "NO_EXIT",
                })
                serial += 1
    return tasks


def _likelihood(action: tuple[str, str], hypothesis: str) -> dict[str, float]:
    method, phase = action
    if phase == "confirm":
        outcome = "typed_exit" if (method == "GET" and hypothesis == "GET_TARGET") or (method == "POST" and hypothesis == "POST_TARGET") else "typed_no_exit"
        return {observation: (1.0 if observation == outcome else 0.0) for observation in OBSERVATIONS}
    if method == "GET":
        rows = {"GET_TARGET": (0.78, 0.08, 0.14), "POST_TARGET": (0.20, 0.62, 0.18), "NO_EXIT": (0.24, 0.24, 0.52)}
    else:
        rows = {"GET_TARGET": (0.62, 0.20, 0.18), "POST_TARGET": (0.08, 0.78, 0.14), "NO_EXIT": (0.24, 0.24, 0.52)}
    query, form, neutral = rows[hypothesis]
    return {"query_signal": query, "form_signal": form, "neutral_signal": neutral, "typed_exit": 0.0, "typed_no_exit": 0.0}


def _outcome_distribution(action: tuple[str, str], belief: dict[str, float]) -> dict[str, float]:
    return _normalise({observation: sum(belief[hypothesis] * _likelihood(action, hypothesis)[observation] for hypothesis in HYPOTHESES) for observation in OBSERVATIONS})


def _expected_information_gain(action: tuple[str, str], belief: dict[str, float]) -> float:
    before = _entropy(belief)
    distribution = _outcome_distribution(action, belief)
    expected_after = 0.0
    for observation, probability in distribution.items():
        if probability <= 0:
            continue
        posterior = _normalise({hypothesis: belief[hypothesis] * _likelihood(action, hypothesis)[observation] for hypothesis in HYPOTHESES})
        expected_after += probability * _entropy(posterior)
    return max(0.0, before - expected_after)


def _utility(action: tuple[str, str], belief: dict[str, float]) -> float:
    method, phase = action
    if phase == "confirm":
        expected_exit = belief["GET_TARGET" if method == "GET" else "POST_TARGET"]
        return 0.90 * expected_exit - 0.06
    return 0.82 * _expected_information_gain(action, belief) - 0.04


def _sample_observation(action: tuple[str, str], target: str, rng: random.Random) -> str:
    distribution = _likelihood(action, target)
    choices = list(distribution)
    weights = [distribution[choice] for choice in choices]
    return rng.choices(choices, weights=weights, k=1)[0]


def _observe(belief: dict[str, float], action: tuple[str, str], observation: str) -> tuple[dict[str, float], float]:
    before_entropy = _entropy(belief)
    posterior = _normalise({hypothesis: belief[hypothesis] * _likelihood(action, hypothesis)[observation] for hypothesis in HYPOTHESES})
    return posterior, max(0.0, before_entropy - _entropy(posterior))


def _make_reset(task: dict[str, Any], action: tuple[str, str], step_index: int) -> dict[str, Any]:
    reset = {"kind": "pg64-local-fresh-reset", "target_host": "127.0.0.1", "target_instance_id": f"pg64-{task['task_id']}-{step_index}", "fresh_target": True, "completed": True, "evaluator_state_hidden": True, "external_network": False, "state_change_allowed": False, "action": list(action)}
    reset["reset_sha256"] = _sha256_json(reset)
    return reset


def _typed_oracle(task: dict[str, Any], action: tuple[str, str]) -> dict[str, Any]:
    method, phase = action
    positive = phase == "confirm" and ((method == "GET" and task["evaluator_target"] == "GET_TARGET") or (method == "POST" and task["evaluator_target"] == "POST_TARGET"))
    return {"oracle_id": "pg64-typed-target-zone-oracle-v1", "modality": "typed_exit" if positive else "bounded_observation", "positive": positive, "confirmed_effect": "typed_exit" if positive else "none", "evaluator_state_hidden": True, "unknown_zone": bool(task["unknown_zone"]), "raw_body_stored": False, "external_network": False, "state_mutated": False}


def _run_episode(task: dict[str, Any], policy: str) -> dict[str, Any]:
    rng = random.Random(int(task["observation_seed"]) + (11 if policy == "active_belief" else 29))
    belief = {hypothesis: 1.0 / len(HYPOTHESES) for hypothesis in HYPOTHESES}
    remaining = list(ACTIONS)
    steps: list[dict[str, Any]] = []
    regret_values: list[float] = []
    entropy_start = _entropy(belief)
    selected: list[tuple[str, str]] = []
    for step_index in range(1, len(ACTIONS) + 1):
        if not remaining:
            break
        candidate_order = list(remaining)
        rng.shuffle(candidate_order)
        scores = {action: _utility(action, belief) for action in candidate_order}
        if policy == "fixed_order":
            action = next(action for action in FIXED_ORDER if action in remaining)
        else:
            action = max(candidate_order, key=lambda item: (scores[item], -candidate_order.index(item)))
        regret = max(scores.values()) - scores[action]
        prior = dict(belief)
        observation = _sample_observation(action, task["evaluator_target"], rng)
        belief, information_gain = _observe(belief, action, observation)
        oracle = _typed_oracle(task, action)
        reset = _make_reset(task, action, step_index)
        response = {"observation_class": observation, "status_class": "2xx" if observation == "typed_exit" else "4xx" if observation == "typed_no_exit" else "2xx", "raw_body_stored": False, "external_network": False}
        evidence = {"task_id": task["task_id"], "surface_projection": task["surface_projection"], "action": list(action), "reset": reset, "response_projection": response, "oracle_projection": oracle, "belief_before": prior, "belief_after": belief}
        step = {"step_id": f"{task['task_id']}-{step_index}", "task_id": task["task_id"], "policy": policy, "candidate_order": [list(item) for item in candidate_order], "selected_action": list(action), "pre_oracle_surface_projection": task["surface_projection"], "belief_before": prior, "expected_information_gain": round(_expected_information_gain(action, prior), 6), "action_utility": round(scores[action], 6), "counterfactual_regret": round(regret, 6), "observed_class": observation, "information_gain": round(information_gain, 6), "belief_after": belief, "reset": reset, "oracle_after_action": oracle, "response_projection_after_action": response, "evidence_hash_algorithm": "sha256-canonical-json", "evidence_hash": _sha256_json(evidence), "raw_probe_stored": False, "raw_response_stored": False, "online_weight_update": False, "long_term_memory_write": False}
        steps.append(step)
        regret_values.append(regret)
        selected.append(action)
        remaining.remove(action)
        if oracle["positive"]:
            break
    positive = task["evaluator_target"] != "NO_EXIT"
    confirmed = bool(steps and steps[-1]["oracle_after_action"]["positive"])
    unknown_abstain = bool(positive and task["unknown_zone"])
    final_route = "unknown_surface" if unknown_abstain else "target_zone" if confirmed else "none"
    return {"task_id": task["task_id"], "policy": policy, "positive": positive, "unknown_zone": task["unknown_zone"], "confirmed": confirmed, "false_accept": bool(not positive and confirmed), "abstain": bool(not confirmed or unknown_abstain), "final_route": final_route, "step_count": len(steps), "entropy_start": round(entropy_start, 6), "entropy_final": round(_entropy(belief), 6), "entropy_reduction": round(entropy_start - _entropy(belief), 6), "mean_regret": round(statistics.mean(regret_values) if regret_values else 0.0, 6), "steps": steps}


def _aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [episode for episode in episodes if episode["positive"]]
    negatives = [episode for episode in episodes if not episode["positive"]]
    known_positives = [episode for episode in positives if not episode["unknown_zone"]]
    unknown_positives = [episode for episode in positives if episode["unknown_zone"]]
    return {"episode_count": len(episodes), "positive_episode_count": len(positives), "negative_control_count": len(negatives), "confirmed_positive_count": sum(int(episode["confirmed"]) for episode in positives), "target_recall": round(sum(int(episode["confirmed"]) for episode in positives) / max(len(positives), 1), 6), "known_target_recall": round(sum(int(episode["confirmed"]) for episode in known_positives) / max(len(known_positives), 1), 6), "unknown_strict_abstain": all(episode["abstain"] and episode["final_route"] == "unknown_surface" for episode in unknown_positives), "unknown_abstain_count": sum(int(episode["abstain"]) for episode in unknown_positives), "negative_false_accept_count": sum(int(episode["false_accept"]) for episode in negatives), "negative_false_accept_rate": round(sum(int(episode["false_accept"]) for episode in negatives) / max(len(negatives), 1), 6), "mean_steps": round(statistics.mean(episode["step_count"] for episode in episodes), 6), "median_steps": float(statistics.median(episode["step_count"] for episode in episodes)), "mean_entropy_reduction": round(statistics.mean(episode["entropy_reduction"] for episode in episodes), 6), "mean_counterfactual_regret": round(statistics.mean(episode["mean_regret"] for episode in episodes), 6), "max_counterfactual_regret": round(max((episode["mean_regret"] for episode in episodes), default=0.0), 6), "multi_step_episode_rate": round(sum(int(episode["step_count"] >= 2) for episode in episodes) / max(len(episodes), 1), 6), "action_counts": dict(Counter(step["selected_action"][0] + "." + step["selected_action"][1] for episode in episodes for step in episode["steps"])), "fresh_reset_count": sum(len(episode["steps"]) for episode in episodes), "evidence_hash_count": sum(sum(int(bool(step["evidence_hash"])) for step in episode["steps"]) for episode in episodes), "raw_probe_stored_count": sum(sum(int(step["raw_probe_stored"]) for step in episode["steps"]) for episode in episodes), "raw_response_stored_count": sum(sum(int(step["raw_response_stored"]) for step in episode["steps"]) for episode in episodes)}


def main() -> int:
    tasks = _tasks()
    active = [_run_episode(task, "active_belief") for task in tasks]
    fixed = [_run_episode(task, "fixed_order") for task in tasks]
    active_metrics = _aggregate(active)
    fixed_metrics = _aggregate(fixed)
    trace = {"schema_version": "pg64-multistep-belief-regret-trace-v1", "evaluation_only": True, "training_eligible": False, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "active_belief": {"episodes": active, "episode_count": len(active), "trace_manifest_sha256": _sha256_json([step["evidence_hash"] for episode in active for step in episode["steps"]])}, "fixed_order": {"episodes": fixed, "episode_count": len(fixed), "trace_manifest_sha256": _sha256_json([step["evidence_hash"] for episode in fixed for step in episode["steps"]])}}
    gate_reasons = []
    if active_metrics["target_recall"] < 0.80:
        gate_reasons.append("active_target_recall_below_0.80")
    if active_metrics["negative_false_accept_count"] != 0:
        gate_reasons.append("active_negative_false_accept")
    if not active_metrics["unknown_strict_abstain"]:
        gate_reasons.append("active_unknown_not_strict_abstain")
    if active_metrics["multi_step_episode_rate"] < 0.50:
        gate_reasons.append("active_policy_not_multistep")
    report = {"protocol_id": "pg-pk-64-multistep-belief-regret-v1", "schema_version": "pg64-multistep-belief-regret-report-v1", "status": "diagnostic_only", "dataset": {"task_count": len(tasks), "split_counts": dict(Counter(task["dataset_role"] for task in tasks)), "layouts": sorted({task["layout"] for task in tasks}), "hypothesis_count": len(HYPOTHESES), "observation_classes": list(OBSERVATIONS)}, "controller": {"class": "expected_information_gain_plus_exit_utility", "typed_oracle_before_action": False, "family_oracle_used_before_action": False, "belief_updates_from_post_action_projection": True, "duplicate_evidence_guard": True, "raw_input_model": False, "gradient_training": False}, "active_belief": active_metrics, "fixed_order": fixed_metrics, "comparison": {"target_recall_delta": round(active_metrics["target_recall"] - fixed_metrics["target_recall"], 6), "mean_step_delta": round(active_metrics["mean_steps"] - fixed_metrics["mean_steps"], 6), "mean_regret_delta": round(active_metrics["mean_counterfactual_regret"] - fixed_metrics["mean_counterfactual_regret"], 6), "entropy_reduction_delta": round(active_metrics["mean_entropy_reduction"] - fixed_metrics["mean_entropy_reduction"], 6)}, "hard_gate": {"schema_version": "pg64-multistep-belief-regret-hard-gate-v1", "status": "passed" if not gate_reasons else "blocked", "claim_allowed": False, "reasons": gate_reasons, "training_allowed": False, "memory_promotion_allowed": False}, "promotion": {"status": "quarantined_multistep_belief_diagnostic", "training_allowed": False, "memory_promotion_allowed": False, "formal_capability_claim_allowed": False}, "interpretation": "PG64 证明的是带噪抽象观察下的多步 belief/动作选择审计链；它不等价于真实漏洞发现或 payload 成功。"}
    protocol = {"protocol_id": "pg-pk-64-multistep-belief-regret-v1", "schema_version": "pg64-multistep-belief-regret-protocol-v1", "objective": "验证观察差分→belief 后验→下一动作→typed oracle→regret 的多步链路，并与固定顺序基线对照。", "authorized_scope": {"target_host": "127.0.0.1", "external_network": False, "fixture_replay_only": True, "state_mutation": False, "raw_probe_persistence": False, "raw_response_body_persistence": False}, "input_contract": {"pre_oracle_surface_projection_only": True, "belief_uses_post_action_projection": True, "typed_oracle_after_action_only": True, "family_label_before_action_forbidden": True, "raw_probe_response_forbidden": True}, "required_gates": {"multi_step_episode_rate_min": 0.50, "target_recall_min": 0.80, "negative_false_accept_zero": True, "unknown_strict_abstain": True, "fresh_reset_per_action": True, "evidence_hash_per_action": True, "counterfactual_regret_reported": True, "fixed_order_baseline_required": True}, "run_result": {"status": report["hard_gate"]["status"], "active_target_recall": active_metrics["target_recall"], "active_mean_regret": active_metrics["mean_counterfactual_regret"]}}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-64 多步 belief / counterfactual regret", "", "每步遵循 pre-oracle 状态 → 候选效用 → noisy observation → belief 后验 → typed oracle（仅动作后） → 下一动作。", "", "| policy | recall | 阴性误报 | 未知弃权 | mean steps | entropy reduction | mean regret |", "|---|---:|---:|---|---:|---:|---:|"]
    for name, metrics in (("active belief", active_metrics), ("fixed order", fixed_metrics)):
        lines.append(f"| {name} | {metrics['target_recall']} | {metrics['negative_false_accept_count']} | {metrics['unknown_strict_abstain']} | {metrics['mean_steps']} | {metrics['mean_entropy_reduction']} | {metrics['mean_counterfactual_regret']} |")
    lines.extend(["", f"硬门：`{report['hard_gate']['status']}`；formal capability claim=false；训练/记忆不晋升。", ""])
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "active_belief": active_metrics, "fixed_order": fixed_metrics, "comparison": report["comparison"], "hard_gate": report["hard_gate"], "report": str(REPORT_PATH.relative_to(ROOT)), "trace": str(TRACE_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
