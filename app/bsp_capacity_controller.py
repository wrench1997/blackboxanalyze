"""Fail-closed BSP capacity pressure planning for trace-learning experiments.

The controller is a policy layer, not a neural model and not a weight editor.
It maps typed bottleneck evidence to a bounded action plan.  A real BSP
Page/Node/Expert mutation must replay the same Rule IR and pass the ablation
gate before it can be applied by the Zig/CUDA model core.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "bsp-capacity-pressure-v1"
_UNIT_KINDS = frozenset({"bsp_page", "bsp_node", "expert_slot", "fragment_head"})
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_KEYS = frozenset({
    "family",
    "hypothesis",
    "oracle",
    "oracle_id",
    "positive",
    "positive_authority",
    "raw_body",
    "raw_payload",
    "evaluator_label",
})


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(str(key) in _FORBIDDEN_KEYS or _contains_forbidden_key(child) for key, child in value.items())
    if isinstance(value, list):
        return any(_contains_forbidden_key(child) for child in value)
    return False


def _bounded_float(value: Any, *, low: float = 0.0, high: float = 1.0) -> float:
    result = float(value)
    if not math.isfinite(result) or result < low or result > high:
        raise ValueError("capacity pressure metric is outside its bounded range")
    return result


def _bounded_nonnegative(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("capacity pressure metric must be finite and non-negative")
    return result


def _unit_ids(value: Any) -> list[str]:
    if not isinstance(value, list) or len(value) > 64:
        raise ValueError("capacity unit list is outside its bounded range")
    result = [str(item) for item in value]
    if any(not _HASH_RE.fullmatch(item) for item in result) or len(set(result)) != len(result):
        raise ValueError("capacity unit ids must be unique SHA-256 commitments")
    return result


def validate_pressure_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a typed pressure observation."""

    if _contains_forbidden_key(observation):
        raise ValueError("capacity pressure observation contains an evaluator or raw field")
    target_id = str(observation.get("target_id", ""))
    if not target_id or len(target_id) > 128:
        raise ValueError("capacity pressure observation requires a bounded target id")
    unit_kind = str(observation.get("unit_kind", ""))
    if unit_kind not in _UNIT_KINDS:
        raise ValueError("capacity pressure observation has an unknown unit kind")
    capacity_units = int(observation.get("capacity_units", -1))
    if capacity_units < 0 or capacity_units > 1_000_000:
        raise ValueError("capacity unit count is outside its bounded range")
    normalized = {
        "schema_version": SCHEMA_VERSION,
        "target_id": target_id,
        "unit_kind": unit_kind,
        "capacity_units": capacity_units,
        "typed_bottleneck": bool(observation.get("typed_bottleneck", False)),
        "fresh_holdout_gap": bool(observation.get("fresh_holdout_gap", False)),
        "cross_dataset_evidence": bool(observation.get("cross_dataset_evidence", False)),
        "cross_seed_evidence": bool(observation.get("cross_seed_evidence", False)),
        "cross_implementation_evidence": bool(observation.get("cross_implementation_evidence", False)),
        "known_recall": _bounded_float(observation.get("known_recall", 0.0)),
        "false_accept_count": int(observation.get("false_accept_count", 0)),
        "unknown_abstain_rate": _bounded_float(observation.get("unknown_abstain_rate", 0.0)),
        "all_abstain": bool(observation.get("all_abstain", False)),
        "latency_ms": _bounded_nonnegative(observation.get("latency_ms", 0.0)),
        "latency_budget_ms": _bounded_nonnegative(observation.get("latency_budget_ms", 0.0)),
        "memory_ratio": _bounded_float(observation.get("memory_ratio", 1.0), low=0.0, high=8.0),
        "redundancy_evidence": bool(observation.get("redundancy_evidence", False)),
        "low_contribution_units": _unit_ids(observation.get("low_contribution_units", [])),
    }
    if normalized["false_accept_count"] < 0:
        raise ValueError("false accept count cannot be negative")
    normalized["observation_sha256"] = _sha256(normalized)
    return normalized


def plan_capacity_action(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Choose one bounded capacity action; never mutates a model."""

    normalized = validate_pressure_observation(observation)
    common = {
        "schema_version": SCHEMA_VERSION,
        "target_id": normalized["target_id"],
        "unit_kind": normalized["unit_kind"],
        "capacity_before": normalized["capacity_units"],
        "executable": False,
        "requires_rule_ir_replay": True,
        "requires_fresh_holdout": True,
        "rollback_required": True,
        "promotion_eligible": False,
        "observation_sha256": normalized["observation_sha256"],
    }
    all_cross_evidence = all(
        normalized[field] for field in (
            "cross_dataset_evidence",
            "cross_seed_evidence",
            "cross_implementation_evidence",
        )
    )
    if normalized["false_accept_count"] > 0:
        action = "hold_and_repair_evidence"
        reason = "false_accept_precedes_capacity_change"
        capacity_after = normalized["capacity_units"]
    elif not all_cross_evidence:
        action = "hold_and_collect_cross_evidence"
        reason = "cross_dataset_seed_implementation_evidence_incomplete"
        capacity_after = normalized["capacity_units"]
    elif (
        normalized["typed_bottleneck"]
        and normalized["fresh_holdout_gap"]
        and normalized["latency_ms"] > normalized["latency_budget_ms"]
    ):
        action = "hold_and_measure_tradeoff"
        reason = "typed_gap_and_speed_pressure_require_tradeoff_measurement"
        capacity_after = normalized["capacity_units"]
    elif normalized["typed_bottleneck"] and normalized["fresh_holdout_gap"]:
        action = "wake_target_unit"
        reason = "typed_bottleneck_with_fresh_holdout_gap"
        capacity_after = normalized["capacity_units"] + 1
    elif normalized["latency_ms"] > normalized["latency_budget_ms"]:
        if normalized["redundancy_evidence"] and normalized["low_contribution_units"]:
            action = "merge_then_ablate_low_contribution_units"
            reason = "speed_pressure_with_functional_redundancy_evidence"
            capacity_after = max(0, normalized["capacity_units"] - len(normalized["low_contribution_units"]))
        else:
            action = "measure_speed_without_ablation"
            reason = "speed_pressure_without_redundancy_proof"
            capacity_after = normalized["capacity_units"]
    else:
        action = "hold_capacity"
        reason = "no_typed_capacity_pressure"
        capacity_after = normalized["capacity_units"]
    return {
        **common,
        "action": action,
        "reason": reason,
        "capacity_after_proposed": capacity_after,
        "low_contribution_units": list(normalized["low_contribution_units"]),
    }


def evaluate_ablation_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    action: str,
) -> dict[str, Any]:
    """Evaluate a speed-constrained ablation and return pass or rollback."""

    if action != "merge_then_ablate_low_contribution_units":
        return {"schema_version": SCHEMA_VERSION, "status": "blocked", "reason": "ablation_action_not_requested", "rollback": True}
    if _contains_forbidden_key(baseline) or _contains_forbidden_key(candidate):
        return {"schema_version": SCHEMA_VERSION, "status": "blocked", "reason": "ablation_metrics_contain_forbidden_field", "rollback": True}
    required = ("known_recall", "false_accept_count", "unknown_abstain_rate", "latency_ms", "memory_ratio", "fresh_holdout_tested", "capacity_units")
    if any(key not in baseline or key not in candidate for key in required):
        return {"schema_version": SCHEMA_VERSION, "status": "blocked", "reason": "ablation_metrics_incomplete", "rollback": True}
    reasons: list[str] = []
    if not bool(baseline["fresh_holdout_tested"]) or not bool(candidate["fresh_holdout_tested"]):
        reasons.append("fresh_holdout_missing")
    if float(candidate["known_recall"]) < float(baseline["known_recall"]):
        reasons.append("known_recall_regressed")
    if int(candidate["false_accept_count"]) > int(baseline["false_accept_count"]):
        reasons.append("false_accept_increased")
    if float(candidate["unknown_abstain_rate"]) < float(baseline["unknown_abstain_rate"]):
        reasons.append("unknown_abstain_regressed")
    speed_or_memory_improved = (
        float(candidate["latency_ms"]) < float(baseline["latency_ms"])
        or float(candidate["memory_ratio"]) < float(baseline["memory_ratio"])
    )
    if not speed_or_memory_improved:
        reasons.append("no_speed_or_memory_gain")
    if int(candidate["capacity_units"]) >= int(baseline["capacity_units"]):
        reasons.append("capacity_not_reduced")
    status = "passed_ablation_gate" if not reasons else "rollback"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "rollback": status != "passed_ablation_gate",
        "reasons": reasons,
        "known_recall_delta": round(float(candidate["known_recall"]) - float(baseline["known_recall"]), 6),
        "false_accept_delta": int(candidate["false_accept_count"]) - int(baseline["false_accept_count"]),
        "unknown_abstain_delta": round(float(candidate["unknown_abstain_rate"]) - float(baseline["unknown_abstain_rate"]), 6),
        "latency_delta_ms": round(float(candidate["latency_ms"]) - float(baseline["latency_ms"]), 6),
        "memory_ratio_delta": round(float(candidate["memory_ratio"]) - float(baseline["memory_ratio"]), 6),
        "capacity_delta": int(candidate["capacity_units"]) - int(baseline["capacity_units"]),
    }


__all__ = [
    "SCHEMA_VERSION",
    "evaluate_ablation_gate",
    "plan_capacity_action",
    "validate_pressure_observation",
]
