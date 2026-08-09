"""Run PG-46's active GET/POST controller on the PG-42 replay catalog.

The controller uses only the frozen PG-43 effect head, bounded projections,
and an abstract semantic reference.  It updates a small belief posterior after
each action and stops on model evidence; evaluator labels are attached only
after the episode for scoring and trace validation.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CATALOG_PATH = ROOT / "research" / "pg42_independent_semantic_catalog_v1.json"
TRACE_PATH = ROOT / "research" / "pg46_active_probe_trace_v1.json"
REPORT_PATH = ROOT / "research" / "pg46_active_probe_report_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg46_active_probe_report_v1.md"
EFFECT_CHECKPOINT_PATH = ROOT / "artifacts" / "pg43-invariant-effect" / "invariant_effect.pt"
PG38_SCRIPT = ROOT / "scripts" / "train_pg38_effect_pair_candidate.py"
PG39_SCRIPT = ROOT / "scripts" / "train_pg39_coarse_delta_candidate.py"
PG43_SCRIPT = ROOT / "scripts" / "train_pg43_invariant_effect_candidate.py"
EFFECT_THRESHOLD = 0.60
ACTION_ORDER = (("GET", "screen"), ("POST", "screen"), ("GET", "confirm"), ("POST", "confirm"))
KNOWN_BINDINGS = {
    "markup-context": "xss",
    "operator-context": "injection",
    "auth-boundary": "authentication",
    "subject-boundary": "access_control",
    "state-invariant": "logic",
    "url-target": "url_redirect",
    "scalar-boundary": "input_validation",
    "local-canary": "command_injection",
    "ordinary-surface": "ordinary_response",
}


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {path.name}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def _semantic(row: dict[str, Any]) -> str:
    ref = str((row.get("payload_manifest") or {}).get("probe_ref", ""))
    prefix = "pg42-semantic-"
    return ref[len(prefix) :] if ref.startswith(prefix) else ref


def _features(pg39: Any, pg43: Any, pairs: list[dict[str, Any]], indices: tuple[int, ...]) -> torch.Tensor:
    if not pairs:
        return torch.empty((0, len(indices)), dtype=torch.float32)
    return torch.sign(torch.stack([pg39._coarse_pair(pair) for pair in pairs])[:, indices])


def _action_manifest(row: dict[str, Any]) -> dict[str, Any]:
    manifest = row["payload_manifest"]
    action = {"method": manifest["method"], "route_template_id": manifest["route_template_id"], "placement": manifest["placement"], "encoding_chain": manifest["encoding_chain"], "probe_ref": manifest["probe_ref"], "probe_sha256": manifest["payload_sha256"], "safety": {"no_external_network": True, "does_not_execute": True, "no_database_write": True, "no_credential_access": True}}
    if manifest["method"] == "POST":
        action["form_field_names"] = manifest["form_field_names"]
    return action


def _belief_update(before: dict[str, float], semantic: str, effect_probability: float) -> dict[str, float]:
    if effect_probability < EFFECT_THRESHOLD:
        return {"unknown": 1.0}
    family = KNOWN_BINDINGS.get(semantic)
    if family is None:
        return {"unknown": 1.0}
    confidence = max(EFFECT_THRESHOLD, min(0.99, float(effect_probability)))
    return {"unknown": round(1.0 - confidence, 6), family: round(confidence, 6)}


def _make_trace_step(trace_module: Any, row: dict[str, Any], control: dict[str, Any], episode_id: str, step_id: str, parent: str | None, next_action: str, belief_before: dict[str, float], belief_after: dict[str, float], effect_probability: float, route_family: str, abstain: bool) -> dict[str, Any]:
    oracle = copy.deepcopy(row["oracle_projection"])
    oracle["negative_control_pair_id"] = control["sample_id"]
    action = _action_manifest(row)
    positive = bool(oracle.get("positive", False))
    if abstain:
        decision = "abstain"
    elif positive and effect_probability >= EFFECT_THRESHOLD:
        decision = "confirmed_positive"
    elif next_action == "stop_episode":
        decision = "confirmed_negative"
    else:
        decision = "candidate"
    echo_body = {"action_manifest": action, "baseline_projection": control["response_projection"], "response_projection": row["response_projection"], "oracle_projection": oracle, "belief_before": belief_before, "belief_after": belief_after, "decision": decision, "next_action": next_action}
    hypothesis = route_family if route_family != "unknown_surface" else "unknown_surface"
    return trace_module.validate_trace_step({"episode_id": episode_id, "step_id": step_id, "parent_step_id": parent, "sampling_seed": int(row["sampling_seed"]), "target_instance_id": row["target_instance_id"], "hypothesis": hypothesis, "belief_before": belief_before, "action_manifest": action, "baseline_projection": control["response_projection"], "response_projection": row["response_projection"], "oracle_projection": oracle, "belief_after": belief_after, "decision": decision, "next_action": next_action, "fresh_reset": row["reset"], "evidence_sha256": row["evidence"]["evidence_hash"], "dataset_stage": "trace_only", "online_weight_update": False, "long_term_memory_write": False, "echo": {"sha256": trace_module.sha256_json(echo_body)}})


def main() -> int:
    pg38 = _load(PG38_SCRIPT, "pg38_for_pg46")
    pg39 = _load(PG39_SCRIPT, "pg39_for_pg46")
    pg43 = _load(PG43_SCRIPT, "pg43_for_pg46")
    trace_module = _load(ROOT / "app" / "trace_aligned_dataset.py", "trace_for_pg46")
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    pairs = pg38._pair_rows(list(catalog["samples"]))
    checkpoint = torch.load(EFFECT_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    effect_model = pg43.InvariantEffectModel(); effect_model.load_state_dict(checkpoint["model_state"]); effect_model.eval()
    invariant_indices = tuple(int(index) for index in checkpoint["invariant_indices"])
    pair_features = _features(pg39, pg43, pairs, invariant_indices)
    with torch.inference_mode(): probabilities = torch.sigmoid(effect_model(pair_features)).cpu()
    probability_by_key: dict[tuple[str, str, str, int, str, str], float] = {}
    pair_by_action: dict[tuple[str, str, str, int, str, str], dict[str, Any]] = {}
    groups: dict[tuple[str, str, str, int], dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for index, pair in enumerate(pairs):
        candidate = pair["candidate"]
        task_key = (str(candidate["implementation"]), str(candidate["surface_id"]), str(candidate["surface_variant"]), int(candidate["sampling_seed"]))
        action_key = (str(candidate["method"]), str(candidate["phase"]))
        groups[task_key][action_key] = pair
        full_key = task_key + action_key
        pair_by_action[full_key] = pair
        probability_by_key[full_key] = float(probabilities[index])
    episodes: list[dict[str, Any]] = []
    all_steps: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    query_counts: list[int] = []
    positive_queries: list[int] = []
    negative_queries: list[int] = []
    effect_success = known_named_success = unknown_safe_count = negative_false_accept = 0
    positive_episode_count = negative_episode_count = 0
    belief_updates = 0
    for task_key in sorted(groups):
        task = groups[task_key]
        first = next(iter(task.values()))["candidate"]
        positive_episode = any(bool(pair["candidate"]["oracle_projection"].get("positive", False)) for pair in task.values())
        semantic = _semantic(first)
        episode_id = f"pg46-episode-{task_key[0]}-{task_key[1]}-{task_key[2]}-s{task_key[3]}"
        previous: str | None = None
        belief = {"unknown": 1.0}
        steps: list[dict[str, Any]] = []
        stopped = False
        route_family = "unknown_surface"
        for action_index, (method, phase) in enumerate(ACTION_ORDER):
            pair = task[(method, phase)]
            candidate = pair["candidate"]
            control = pair["control"]
            full_key = task_key + (method, phase)
            effect_probability = probability_by_key[full_key]
            effect_accepted = effect_probability >= EFFECT_THRESHOLD
            route_family = KNOWN_BINDINGS.get(semantic, "unknown_surface") if effect_accepted else "unknown_surface"
            abstain = bool(effect_accepted and semantic not in KNOWN_BINDINGS)
            belief_after = _belief_update(belief, semantic, effect_probability)
            next_action = "stop_episode" if effect_accepted else "probe_" + (ACTION_ORDER[action_index + 1][0] + "_" + ACTION_ORDER[action_index + 1][1] if action_index + 1 < len(ACTION_ORDER) else "exhausted")
            step = _make_trace_step(trace_module, candidate, control, episode_id, f"{episode_id}-{method.casefold()}-{phase}", previous, next_action, belief, belief_after, effect_probability, route_family, abstain)
            steps.append(step); all_steps.append(step); previous = step["step_id"]; belief = belief_after; belief_updates += 1
            action_counts[f"{method}.{phase}"] += 1; method_counts[method] += 1
            if effect_accepted:
                stopped = True
                break
        episode_report = trace_module.evaluate_episode(steps)
        query_count = len(steps); query_counts.append(query_count)
        if positive_episode:
            positive_episode_count += 1; positive_queries.append(query_count)
        else:
            negative_episode_count += 1; negative_queries.append(query_count)
        typed_positive = positive_episode and stopped
        effect_success += int(typed_positive)
        known_named_success += int(typed_positive and semantic in KNOWN_BINDINGS)
        unknown_safe_count += int(typed_positive and semantic not in KNOWN_BINDINGS and route_family == "unknown_surface")
        negative_false_accept += int((not positive_episode) and stopped)
        final_decision = {"effect_confirmed": bool(stopped), "route_family": route_family if stopped else "unknown_surface", "abstain": bool((not stopped) or route_family == "unknown_surface"), "reason": "known_semantic_binding" if stopped and route_family != "unknown_surface" else "semantic_outside_ontology_or_effect_unconfirmed"}
        rule_ir = {"schema_version": "sift-cross-lab-rule-ir-manifest-v1", "rule_key": f"{final_decision['route_family']}.pg46.active", "grammar_version": "rule-ir-v1", "family_candidate": final_decision["route_family"], "operator_set": ["and", "present", "threshold"], "required_slots": ["surface", "transport", "oracle"], "bound_slots": ["surface", "transport", "oracle"], "executable": False}
        episode_records.append({"episode_id": episode_id, "implementation": task_key[0], "surface_id": task_key[1], "surface_variant": task_key[2], "sampling_seed": task_key[3], "semantic_reference": semantic, "positive": positive_episode, "step_count": query_count, "status": episode_report["status"], "trace_sha256": episode_report["trace_sha256"], "belief_update_count": query_count, "final_decision": final_decision, "rule_ir_sha256": hashlib.sha256(json.dumps(rule_ir, sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
        episodes.append(episode_report)
    positive_semantics = Counter(item["semantic_reference"] for item in episode_records if item["positive"])
    known_positive_count = sum(count for semantic, count in positive_semantics.items() if semantic in KNOWN_BINDINGS)
    unknown_positive_count = sum(count for semantic, count in positive_semantics.items() if semantic not in KNOWN_BINDINGS)
    safe_gate_reasons: list[str] = []
    if negative_false_accept != 0: safe_gate_reasons.append("negative_effect_false_accept")
    if known_named_success != known_positive_count: safe_gate_reasons.append("known_family_recall_below_1")
    if unknown_safe_count != unknown_positive_count: safe_gate_reasons.append("unknown_effect_not_strict_abstain")
    if any(item["status"] != "accepted_evaluation" for item in episodes): safe_gate_reasons.append("trace_episode_not_accepted")
    safe_gate = {"schema_version": "sift-pg46-active-probe-gate-v1", "status": "passed" if not safe_gate_reasons else "blocked", "claim_allowed": not safe_gate_reasons, "reasons": safe_gate_reasons, "training_allowed": False, "memory_promotion_allowed": False}
    query_mean = round(float(statistics.mean(query_counts)), 6); query_median = float(statistics.median(query_counts))
    metrics = {"episode_count": len(episode_records), "positive_episode_count": positive_episode_count, "negative_episode_count": negative_episode_count, "effect_success_count": effect_success, "effect_success_rate": round(effect_success / max(positive_episode_count, 1), 6), "known_positive_count": known_positive_count, "known_named_success_count": known_named_success, "known_family_recall": round(known_named_success / max(known_positive_count, 1), 6), "unknown_positive_count": unknown_positive_count, "unknown_safe_abstain_count": unknown_safe_count, "unknown_strict_abstain": unknown_safe_count == unknown_positive_count, "negative_false_accept_count": negative_false_accept, "negative_false_accept_rate": round(negative_false_accept / max(negative_episode_count, 1), 6), "mean_queries": query_mean, "median_queries": query_median, "positive_mean_queries": round(float(statistics.mean(positive_queries)), 6), "negative_mean_queries": round(float(statistics.mean(negative_queries)), 6), "fixed_probe_baseline_queries": 4.0, "mean_query_reduction_rate": round((4.0 - query_mean) / 4.0, 6), "get_post_covered": set(method_counts) == {"GET", "POST"}, "belief_update_count": belief_updates, "accepted_trace_episode_count": sum(int(item["status"] == "accepted_evaluation") for item in episodes)}
    trace = {"schema_version": "pg-pk-46-active-probe-trace-v1", "purpose": "effect-gated active GET/POST probe trace", "evaluation_only": True, "training_eligible": False, "methods": ["GET", "POST"], "action_order": [f"{method}.{phase}" for method, phase in ACTION_ORDER], "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "episodes": episodes, "episode_records": episode_records, "steps": all_steps, "episode_count": len(episodes), "accepted_evaluation_episode_count": metrics["accepted_trace_episode_count"], "trace_manifest_sha256": trace_module.sha256_json([step["trace_sha256"] for step in all_steps])}
    catalog_sha256 = hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest()
    checkpoint_sha256 = hashlib.sha256(EFFECT_CHECKPOINT_PATH.read_bytes()).hexdigest()
    report = {"protocol_id": "sift-pg46-active-probe-v1", "schema_version": "pg-pk-46-active-probe-report-v1", "status": "diagnostic_only", "catalog": {"path": str(CATALOG_PATH.relative_to(ROOT)), "sha256": catalog_sha256, "pair_count": len(pairs), "raw_probe_strings_stored": False, "raw_response_bodies_stored": False}, "model": {"effect_checkpoint": str(EFFECT_CHECKPOINT_PATH.relative_to(ROOT)), "checkpoint_sha256": checkpoint_sha256, "effect_head_input": "bounded invariant candidate-control delta", "typed_oracle_consumed_by_policy": False, "family_consumed_by_policy": False}, "policy": {"action_order": [f"{method}.{phase}" for method, phase in ACTION_ORDER], "stop_threshold": EFFECT_THRESHOLD, "posterior_update": "model effect probability plus bounded semantic binding", "online_weight_update": False, "long_term_memory_write": False}, "metrics": metrics, "action_counts": dict(action_counts), "method_counts": dict(method_counts), "positive_semantic_episode_counts": dict(positive_semantics), "safe_gate": safe_gate, "formal_capability_claim_allowed": False, "promotion": {"status": "quarantined_active_policy", "training_allowed": False, "memory_promotion_allowed": False}, "trace": str(TRACE_PATH.relative_to(ROOT)), "manifest_sha256": hashlib.sha256(json.dumps({"protocol_id": "sift-pg46-active-probe-v1", "catalog_sha256": catalog_sha256, "checkpoint_sha256": checkpoint_sha256, "trace_sha256": trace["trace_manifest_sha256"]}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-46 active probe policy", "", "effect-gated belief update；动作顺序 GET.screen → POST.screen → GET.confirm → POST.confirm。", "", "| 指标 | 值 |", "|---|---:|"]
    for key in ("episode_count", "effect_success_rate", "known_family_recall", "unknown_strict_abstain", "negative_false_accept_count", "median_queries", "mean_queries", "mean_query_reduction_rate", "get_post_covered", "accepted_trace_episode_count"):
        lines.append(f"| {key} | {metrics[key]} |")
    lines.extend(["", f"安全门禁：`{safe_gate['status']}`；formal capability claim=false；训练/记忆不晋升。", ""])
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "metrics": metrics, "action_counts": dict(action_counts), "safe_gate": safe_gate, "report": str(REPORT_PATH.relative_to(ROOT)), "trace": str(TRACE_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
