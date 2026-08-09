"""Evaluator-only multi-axis distance for the maze trace UI.

There is no universal scalar distance to a security conclusion.  This module
keeps the meaningful axes separate: unmet acceptance gates, posterior
uncertainty, false-accept risk, and action cost.  A weighted display value is
provided only for ordering/rendering; it is never included in model inputs or
used as an oracle label.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from .generic_belief_state import GENERIC_STATES, entropy


SCHEMA_VERSION = "maze-multi-axis-distance-v1"
DEFAULT_GATES = (
    "authorized_scope",
    "fresh_reset",
    "safe_probe",
    "matched_negative_control",
    "cross_channel_replay",
    "typed_effect",
    "evidence_hash",
    "rule_ir_binding",
)


def _clamp(value: Any) -> float:
    return min(max(float(value), 0.0), 1.0)


def _normalized_entropy(posterior: Mapping[str, float]) -> float:
    return _clamp(entropy({state: float(posterior.get(state, 0.0) or 0.0) for state in GENERIC_STATES}) / math.log(len(GENERIC_STATES)))


def compute_maze_distance(
    *,
    gates: Mapping[str, bool],
    posterior: Mapping[str, float],
    false_accept_risk: float,
    action_cost: int,
    action_budget: int,
    gate_order: tuple[str, ...] = DEFAULT_GATES,
) -> dict[str, Any]:
    """Return a vector distance; this function does not decide confirmation.

    ``gates`` are evaluator-side facts.  Missing gates are treated as unmet,
    which makes the result fail closed.  ``display_distance`` is deliberately
    labelled non-authoritative so consumers cannot mistake it for a typed
    vulnerability score.
    """

    if action_cost < 0 or action_budget <= 0:
        raise ValueError("action cost must be non-negative and budget positive")
    ordered = tuple(gate_order)
    if not ordered:
        raise ValueError("at least one acceptance gate is required")
    remaining = [name for name in ordered if not bool(gates.get(name, False))]
    gate_axis = len(remaining) / len(ordered)
    uncertainty_axis = _normalized_entropy(posterior)
    risk_axis = _clamp(false_accept_risk)
    cost_axis = _clamp(action_cost / action_budget)
    # UI/scheduler ordering only.  It is not a training feature or oracle.
    display_distance = 0.45 * gate_axis + 0.25 * uncertainty_axis + 0.20 * risk_axis + 0.10 * cost_axis
    return {
        "schema_version": SCHEMA_VERSION,
        "distance_kind": "multi_axis_evaluator_only",
        "remaining_gate_count": len(remaining),
        "gate_count": len(ordered),
        "remaining_gates": remaining,
        "axes": {
            "gate": round(gate_axis, 6),
            "uncertainty": round(uncertainty_axis, 6),
            "false_accept_risk": round(risk_axis, 6),
            "action_cost": round(cost_axis, 6),
        },
        "display_distance": round(display_distance, 6),
        "display_only": True,
        "model_input_allowed": False,
        "confirmation_allowed": False,
    }


__all__ = ["DEFAULT_GATES", "SCHEMA_VERSION", "compute_maze_distance"]
