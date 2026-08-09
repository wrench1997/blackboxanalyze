"""Hard audit of the pre-oracle probe policy handoff.

PG-47 already contains a learned action-value head and a safe PG-42 replay.
PG-60 does not retrain it blindly.  It audits whether the apparent query
reduction came from genuinely state-dependent choices or merely from a fixed
GET/POST order.  A fixed order blocks the capability claim and triggers a
requirement for new counterfactual action data.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PG47_REPORT_PATH = ROOT / "research" / "pg47_counterfactual_action_value_report_v1.json"
PG47_TRACE_PATH = ROOT / "research" / "pg47_counterfactual_action_trace_v1.json"
PG47_SCRIPT_PATH = ROOT / "scripts" / "train_pg47_counterfactual_action_value.py"
PG42_PATH = ROOT / "research" / "pg42_independent_semantic_catalog_v1.json"
REPORT_PATH = ROOT / "research" / "pg60_pre_oracle_probe_policy_audit_report_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg60_pre_oracle_probe_policy_audit_report_v1.md"
SEED = 20600803


def _entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 1 or len(counts) <= 1:
        return 0.0
    raw = -sum((count / total) * math.log(count / total) for count in counts.values())
    return raw / math.log(len(counts))


def _function_has_forbidden_ast(function: ast.AST, forbidden: set[str]) -> bool:
    return any(isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in forbidden for node in ast.walk(function))


def _audit_model_input_contract() -> dict[str, Any]:
    tree = ast.parse(PG47_SCRIPT_PATH.read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    state = functions.get("_state_features")
    invariant = functions.get("_invariant")
    forbidden = {"oracle_projection", "family", "semantic_reference", "raw_payload", "raw_response"}
    return {
        "state_features_function_present": state is not None,
        "invariant_function_present": invariant is not None,
        "state_features_reads_evaluator_fields": bool(state and _function_has_forbidden_ast(state, forbidden)),
        "invariant_reads_evaluator_fields": bool(invariant and _function_has_forbidden_ast(invariant, forbidden)),
        "report_declares_typed_oracle_consumed_by_model": False,
        "report_declares_family_consumed_by_model": False,
    }


def main() -> int:
    report = json.loads(PG47_REPORT_PATH.read_text(encoding="utf-8"))
    trace = json.loads(PG47_TRACE_PATH.read_text(encoding="utf-8"))
    pg42 = json.loads(PG42_PATH.read_text(encoding="utf-8"))
    episodes = trace.get("episodes") or []
    steps = trace.get("steps") or []
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step in steps:
        by_episode[str(step.get("episode_id"))].append(step)
    patterns: Counter[str] = Counter()
    confirmation_actions: Counter[str] = Counter()
    first_actions: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    for episode in episodes:
        # The trace writer appends steps in replay order; retain that order
        # instead of lexicographically sorting IDs (which would turn
        # GET.screen/POST.screen into a misleading sequence).
        episode_steps = by_episode.get(str(episode.get("episode_id")), [])
        methods = [str((step.get("action_manifest") or {}).get("method", "UNKNOWN")) for step in episode_steps]
        patterns["→".join(methods)] += 1
        if methods:
            first_actions[methods[0]] += 1
        for step in episode_steps:
            method = str((step.get("action_manifest") or {}).get("method", "UNKNOWN"))
            method_counts[method] += 1
            if step.get("next_action") == "stop_episode":
                phase = str((step.get("action_manifest") or {}).get("phase", "confirm"))
                confirmation_actions[f"{method}.{phase}"] += 1
                break
    # PG-42's pair matrix is the denominator for real action choice.  Every
    # episode should expose both GET and POST candidates before confirmation.
    sample_groups: dict[tuple[str, str, str, int], set[str]] = defaultdict(set)
    for row in pg42.get("samples") or []:
        key = (str(row.get("implementation")), str(row.get("surface_id")), str(row.get("surface_variant")), int(row.get("sampling_seed", 0)))
        sample_groups[key].add(str(row.get("method")))
    paired_groups = sum(int(methods == {"GET", "POST"}) for methods in sample_groups.values())
    contract = _audit_model_input_contract()
    metrics = {
        "episode_count": len(episodes),
        "step_count": len(steps),
        "action_sequence_pattern_count": len(patterns),
        "action_sequence_patterns": dict(patterns),
        "first_action_counts": dict(first_actions),
        "confirmation_action_counts": dict(confirmation_actions),
        "method_counts": dict(method_counts),
        "confirmation_action_entropy": round(_entropy(confirmation_actions), 6),
        "paired_get_post_group_count": paired_groups,
        "get_post_coverage": set(method_counts) == {"GET", "POST"},
        "fresh_reset_count": sum(int(bool((step.get("fresh_reset") or {}).get("completed"))) for step in steps),
        "evaluation_oracle_fields_present": sum(int("oracle_projection" in step) for step in steps),
        "raw_probe_stored_count": sum(int(bool(step.get("raw_probe_stored", False))) for step in steps),
        "raw_response_stored_count": sum(int(bool(step.get("raw_response_stored", False))) for step in steps),
    }
    reasons: list[str] = []
    if metrics["confirmation_action_entropy"] < 0.5:
        reasons.append("confirmation_action_is_fixed_order_confounded")
    if len(confirmation_actions) < 2:
        reasons.append("no_state_dependent_confirmation_action_diversity")
    if not metrics["get_post_coverage"]:
        reasons.append("missing_get_post_coverage")
    if metrics["fresh_reset_count"] != metrics["step_count"]:
        reasons.append("fresh_reset_incomplete")
    if contract["state_features_reads_evaluator_fields"] or contract["invariant_reads_evaluator_fields"]:
        reasons.append("pre_oracle_model_input_contains_evaluator_field")
    if metrics["raw_probe_stored_count"] or metrics["raw_response_stored_count"]:
        reasons.append("raw_material_retained")
    gate = {
        "schema_version": "pg60-pre-oracle-probe-policy-hard-gate-v1",
        "status": "blocked" if reasons else "passed",
        "claim_allowed": not reasons,
        "reasons": reasons,
        "training_allowed": False,
        "memory_promotion_allowed": False,
    }
    report_out = {
        "protocol_id": "pg-pk-60-pre-oracle-probe-policy-audit-v1",
        "schema_version": "pg60-pre-oracle-probe-policy-audit-report-v1",
        "status": "diagnostic_only",
        "inputs": {
            "pg47_report": str(PG47_REPORT_PATH.relative_to(ROOT)),
            "pg47_trace": str(PG47_TRACE_PATH.relative_to(ROOT)),
            "pg42_catalog": str(PG42_PATH.relative_to(ROOT)),
            "model_input_contract": contract,
        },
        "metrics": metrics,
        "hard_gate": gate,
        "interpretation": "PG-47's safety and query reduction metrics are not a proof of active policy learning when confirmation action entropy is zero; collect counterfactual targets where the best GET/POST action varies under the same abstract state.",
        "required_next_dataset": {
            "same_abstract_state_with_different_best_method": True,
            "randomized_action_order": True,
            "matched_get_post_negative_controls": True,
            "typed_oracle_after_action_only": True,
            "fresh_reset_per_action": True,
            "evidence_hash_per_action": True,
            "no_raw_probe_or_response_retention": True,
        },
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "formal_capability_claim_allowed": False,
            "status": "blocked_fixed_action_order",
        },
    }
    report_out["report_sha256"] = hashlib.sha256(json.dumps(report_out, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    REPORT_PATH.write_text(json.dumps(report_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("\n".join([
        "# PG-60 pre-oracle probe policy hard audit",
        "",
        f"episodes/steps：`{len(episodes)}/{len(steps)}`；confirmation actions：`{dict(confirmation_actions)}`；归一化动作熵：`{metrics['confirmation_action_entropy']:.3f}`。",
        f"安全硬门：`{gate['status']}`；原因：`{', '.join(reasons) or 'none'}`。",
        "PG-47 的 GET→POST→GET 序列不能作为真正主动学习证明；下一轮必须补充最佳 GET/POST 会随抽象状态变化的反事实样本。",
        "",
    ]) + "\n", encoding="utf-8")
    print(json.dumps({"protocol_id": report_out["protocol_id"], "metrics": metrics, "hard_gate": gate, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
