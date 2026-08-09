"""PG-67 independent Rule IR binding under typed-oracle noise.

The PG-66 action head is frozen.  This runner changes target family/layout
metadata and injects controlled post-action oracle qualities.  Family/Rule IR
binding is evaluated only after a typed exit; contradictory and unknown-family
signals must abstain rather than misname a family.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
PG65_SCRIPT = ROOT / "scripts" / "train_pg65_trajectory_policy_head.py"
PG66_CHECKPOINT = ROOT / "artifacts" / "pg66-utility-ranking" / "ranking_head.pt"
REPORT_PATH = ROOT / "research" / "pg67_independent_rule_ir_oracle_noise_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg67_independent_rule_ir_oracle_noise_protocol_v1.json"
TRACE_PATH = ROOT / "research" / "pg67_independent_rule_ir_oracle_noise_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg67_independent_rule_ir_oracle_noise_report_v1.md"
SEED = 20670803
ACTIONS = (("GET", "screen"), ("POST", "screen"), ("GET", "confirm"), ("POST", "confirm"))
HYPOTHESES = ("GET_TARGET", "POST_TARGET", "NO_EXIT")
KNOWN_FAMILIES = ("markup", "operator", "access", "logic")
UNKNOWN_FAMILY = "template"


def _load_pg65() -> Any:
    spec = importlib.util.spec_from_file_location("pg65_for_pg67", PG65_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG65 model helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    serial = 0
    for layout_index, layout in enumerate(("lumen", "mosaic")):
        for local_index in range(96):
            seed = SEED + serial * 31 + layout_index * 401
            target = HYPOTHESES[(local_index + layout_index) % len(HYPOTHESES)]
            negative = target == "NO_EXIT"
            unknown_zone = not negative and local_index % 7 == 0
            family = UNKNOWN_FAMILY if unknown_zone else KNOWN_FAMILIES[(local_index + 2 * layout_index) % len(KNOWN_FAMILIES)]
            noise = "ambiguous" if serial % 7 == 0 else "contradictory" if serial % 13 == 0 else "exact"
            tasks.append({"task_id": f"pg67-{layout}-{serial:04d}", "source_id": f"pg67-independent-{layout}", "layout": layout, "sampling_seed": seed, "observation_seed": seed + 17001, "surface_projection": {"surface_class": ("collection", "record", "search")[(local_index + seed) % 3], "response_shape": ("compact", "object", "redirect")[(2 * local_index + seed) % 3], "channel_hint": ("balanced_lane", "query_leaning", "form_leaning")[(local_index + layout_index) % 3], "route_depth": 1 + ((local_index + seed) % 3), "pre_oracle": True, "layout_role": "pg67_independent_rule_ir_surface"}, "evaluator_target": target, "family": family, "unknown_zone": unknown_zone, "negative_control": negative, "oracle_quality": noise})
            serial += 1
    return tasks


def _oracle(task: dict[str, Any], action: tuple[str, str]) -> dict[str, Any]:
    positive = action[1] == "confirm" and ((action[0] == "GET" and task["evaluator_target"] == "GET_TARGET") or (action[0] == "POST" and task["evaluator_target"] == "POST_TARGET"))
    signal = task["family"]
    consistency = True
    if task["oracle_quality"] == "ambiguous":
        signal = "ambiguous"
    elif task["oracle_quality"] == "contradictory":
        signal = KNOWN_FAMILIES[(KNOWN_FAMILIES.index(task["family"]) + 1) % len(KNOWN_FAMILIES)] if task["family"] in KNOWN_FAMILIES else "markup"
        consistency = False
    return {"oracle_id": "pg67-independent-typed-oracle-v1", "positive": positive, "modality": "typed_exit" if positive else "bounded_observation", "family_signal": signal if positive else "none", "family_signal_consistent": consistency if positive else True, "oracle_quality": task["oracle_quality"], "evaluator_state_hidden": True, "raw_body_stored": False, "external_network": False, "state_mutated": False}


def _bind_rule_ir(oracle: dict[str, Any], action: tuple[str, str]) -> dict[str, Any]:
    signal = str(oracle.get("family_signal", "none"))
    bound = bool(oracle.get("positive") and oracle.get("family_signal_consistent") and signal in KNOWN_FAMILIES)
    status = "bound" if bound else "abstain"
    family = signal if bound else "unknown_surface"
    slots = {"surface": "independent_abstract_surface", "transport": action[0], "oracle": oracle.get("modality", "none"), "family": family}
    rule = {"schema_version": "rule-ir-v1", "binding_status": status, "family_candidate": family, "bound_slots": [key for key in ("surface", "transport", "oracle", "family") if bound or key != "family"], "required_slots": ["surface", "transport", "oracle", "family"], "executable": False, "operator_set": ["and", "eq", "present"], "binding_sha256": _sha256_json(slots), "slots": slots}
    return rule


def main() -> int:
    pg65 = _load_pg65()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = _tasks()
    model = pg65.TrajectoryPolicyHead(len(pg65._features(tasks[0]["surface_projection"], {hypothesis: 1 / 3 for hypothesis in HYPOTHESES}, ACTIONS[0]))).to(device)
    checkpoint = torch.load(PG66_CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    episodes: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    for task in tasks:
        rng = random.Random(int(task["observation_seed"]))
        belief = {hypothesis: 1 / 3 for hypothesis in HYPOTHESES}
        remaining = list(ACTIONS)
        steps: list[dict[str, Any]] = []
        for step_index in range(1, len(ACTIONS) + 1):
            if not remaining:
                break
            order = list(remaining)
            rng.shuffle(order)
            features = torch.tensor([pg65._features(task["surface_projection"], belief, action) for action in order], dtype=torch.float32, device=device)
            with torch.inference_mode():
                logits = model(features).detach().cpu().tolist()
            action = order[max(range(len(order)), key=lambda index: (logits[index], -index))]
            observation = pg65._sample(action, task["evaluator_target"], rng)
            prior = dict(belief)
            belief = pg65._posterior(belief, action, observation)
            oracle = _oracle(task, action)
            rule_ir = _bind_rule_ir(oracle, action)
            reset = {"kind": "pg67-independent-fresh-reset", "target_host": "127.0.0.1", "target_instance_id": f"{task['task_id']}-{step_index}", "fresh_target": True, "completed": True, "evaluator_state_hidden": True, "external_network": False, "state_change_allowed": False, "action": list(action)}
            reset["reset_sha256"] = _sha256_json(reset)
            response = {"observation_class": observation, "status_class": "2xx" if observation == "typed_exit" else "4xx" if observation == "typed_no_exit" else "2xx", "raw_body_stored": False, "external_network": False}
            evidence = {"task_id": task["task_id"], "surface_projection": task["surface_projection"], "action": list(action), "belief_before": prior, "belief_after": belief, "oracle": oracle, "rule_ir": rule_ir, "response": response, "reset": reset}
            steps.append({"step_id": f"{task['task_id']}-{step_index}", "candidate_order": [list(item) for item in order], "selected_action": list(action), "pre_oracle_surface_projection": task["surface_projection"], "belief_before": prior, "policy_logits": [round(float(value), 6) for value in logits], "observed_class": observation, "belief_after": belief, "reset": reset, "oracle_after_action": oracle, "rule_ir_after_action": rule_ir, "response_projection_after_action": response, "evidence_hash_algorithm": "sha256-canonical-json", "evidence_hash": _sha256_json(evidence), "raw_probe_stored": False, "raw_response_stored": False, "online_weight_update": False, "long_term_memory_write": False})
            action_counts[f"{action[0]}.{action[1]}"] += 1
            remaining.remove(action)
            if oracle["positive"]:
                break
        positive = not task["negative_control"]
        confirmed = bool(steps and steps[-1]["oracle_after_action"]["positive"])
        final_rule = steps[-1]["rule_ir_after_action"] if steps else _bind_rule_ir({"positive": False}, ("GET", "screen"))
        named = bool(confirmed and final_rule["binding_status"] == "bound")
        episodes.append({"task_id": task["task_id"], "layout": task["layout"], "positive": positive, "family": task["family"], "unknown_zone": task["unknown_zone"], "oracle_quality": task["oracle_quality"], "confirmed": confirmed, "family_named": named, "false_accept": bool(not positive and confirmed), "unknown_misname": bool(task["unknown_zone"] and final_rule["binding_status"] == "bound"), "abstain": bool(task["unknown_zone"] and final_rule["binding_status"] != "bound" or not confirmed), "final_rule_ir": final_rule, "step_count": len(steps), "steps": steps})
    positives = [episode for episode in episodes if episode["positive"]]
    known = [episode for episode in positives if not episode["unknown_zone"]]
    unknown = [episode for episode in positives if episode["unknown_zone"]]
    negatives = [episode for episode in episodes if not episode["positive"]]
    known_named = sum(int(episode["family_named"]) for episode in known)
    metrics = {"episode_count": len(episodes), "positive_episode_count": len(positives), "known_positive_count": len(known), "unknown_positive_count": len(unknown), "negative_control_count": len(negatives), "effect_recall": round(sum(int(episode["confirmed"]) for episode in positives) / max(len(positives), 1), 6), "known_family_recall": round(known_named / max(len(known), 1), 6), "unknown_misname_count": sum(int(episode["unknown_misname"]) for episode in unknown), "unknown_strict_abstain": all(episode["abstain"] and not episode["unknown_misname"] for episode in unknown), "negative_false_accept_count": sum(int(episode["false_accept"]) for episode in negatives), "oracle_quality_counts": dict(Counter(episode["oracle_quality"] for episode in positives)), "oracle_ambiguous_known_count": sum(int(episode["oracle_quality"] == "ambiguous" and not episode["unknown_zone"]) for episode in positives), "oracle_contradictory_known_count": sum(int(episode["oracle_quality"] == "contradictory" and not episode["unknown_zone"]) for episode in positives), "rule_ir_bound_count": sum(int(episode["family_named"]) for episode in positives), "rule_ir_abstain_count": sum(int(not episode["family_named"]) for episode in positives), "mean_steps": round(statistics.mean(episode["step_count"] for episode in episodes), 6), "action_counts": dict(action_counts), "fresh_reset_count": sum(len(episode["steps"]) for episode in episodes), "evidence_hash_count": sum(sum(int(bool(step["evidence_hash"])) for step in episode["steps"]) for episode in episodes), "raw_probe_stored_count": 0, "raw_response_stored_count": 0}
    reasons = []
    if metrics["effect_recall"] < 0.80:
        reasons.append("effect_recall_below_0.80")
    if metrics["known_family_recall"] < 0.80:
        reasons.append("known_family_recall_below_0.80")
    if metrics["unknown_misname_count"] != 0:
        reasons.append("unknown_family_misname")
    if metrics["negative_false_accept_count"] != 0:
        reasons.append("negative_false_accept")
    if not metrics["unknown_strict_abstain"]:
        reasons.append("unknown_not_strict_abstain")
    report = {"protocol_id": "pg-pk-67-independent-rule-ir-oracle-noise-v1", "schema_version": "pg67-independent-rule-ir-oracle-noise-report-v1", "status": "diagnostic_only", "source": {"implementation": "pg67-independent-rule-ir-v1", "layouts": ["lumen", "mosaic"], "model_checkpoint": str(PG66_CHECKPOINT.relative_to(ROOT)), "model_retrained_on_pg67": False, "family_in_pre_oracle_input": False}, "metrics": metrics, "oracle_noise_contract": {"exact": True, "ambiguous": True, "contradictory": True, "unknown_family": UNKNOWN_FAMILY, "contradictory_signal_must_abstain": True}, "hard_gate": {"schema_version": "pg67-independent-rule-ir-oracle-noise-hard-gate-v1", "status": "passed" if not reasons else "blocked", "claim_allowed": False, "reasons": reasons, "training_allowed": False, "memory_promotion_allowed": False}, "promotion": {"status": "quarantined_independent_rule_ir_oracle_noise", "training_allowed": False, "memory_promotion_allowed": False, "formal_capability_claim_allowed": False}, "interpretation": "PG67 证明的是动作确认后 Rule IR 绑定在可控 oracle 噪声下的拒绝/弃权边界，不是 pre-oracle 漏洞族发现证明。"}
    protocol = {"protocol_id": "pg-pk-67-independent-rule-ir-oracle-noise-v1", "schema_version": "pg67-independent-rule-ir-oracle-noise-protocol-v1", "objective": "在独立目标族与 typed-oracle 噪声下验证 effect confirmation、Rule IR family binding、矛盾拒绝和未知族弃权的分离。", "authorized_scope": {"target_host": "127.0.0.1", "external_network": False, "fixture_replay_only": True, "state_mutation": False, "raw_probe_persistence": False, "raw_response_body_persistence": False}, "input_contract": {"pre_oracle_model_input": ["surface_projection", "belief_before", "candidate_action"], "family_before_action_forbidden": True, "typed_oracle_after_action_only": True, "rule_ir_binding_after_typed_exit_only": True, "raw_probe_response_forbidden": True}, "required_gates": {"effect_recall_min": 0.80, "known_family_recall_min": 0.80, "unknown_misname_zero": True, "negative_false_accept_zero": True, "unknown_strict_abstain": True, "contradictory_oracle_must_abstain": True, "fresh_reset_per_action": True, "evidence_hash_per_action": True, "training_promotion_on_fixture": False, "memory_promotion_on_fixture": False}, "run_result": {"status": report["hard_gate"]["status"], "known_family_recall": metrics["known_family_recall"], "unknown_misname_count": metrics["unknown_misname_count"], "negative_false_accept_count": metrics["negative_false_accept_count"]}}
    trace = {"schema_version": "pg67-independent-rule-ir-oracle-noise-trace-v1", "evaluation_only": True, "training_eligible": False, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "model_retrained_on_pg67": False, "episodes": episodes, "episode_count": len(episodes), "trace_manifest_sha256": _sha256_json([step["evidence_hash"] for episode in episodes for step in episode["steps"]])}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-67 独立 Rule IR / typed oracle 噪声", "", "模型只在动作前看 surface + belief；Rule IR 只在动作后的 typed oracle 上绑定。", "", "| 指标 | 值 |", "|---|---:|"]
    for key in ("effect_recall", "known_family_recall", "unknown_misname_count", "negative_false_accept_count", "unknown_strict_abstain", "rule_ir_bound_count", "rule_ir_abstain_count"):
        lines.append(f"| {key} | {metrics[key]} |")
    lines.extend(["", f"硬门：`{report['hard_gate']['status']}`；formal capability claim=false；训练/记忆不晋升。", ""])
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "metrics": metrics, "hard_gate": report["hard_gate"], "report": str(REPORT_PATH.relative_to(ROOT)), "trace": str(TRACE_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
