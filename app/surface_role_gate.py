"""Promotion gate for the coarse surface-role discriminator.

The head is a useful prior, not a vulnerability oracle.  This gate prevents a
calibration split with poor positive recall from becoming an acceptance rule.
"""

from __future__ import annotations

import math
from typing import Any, Iterable


SURFACE_ROLE_GATE_SCHEMA = "sift-surface-role-gate-v1"
DEFAULT_SURFACE_ROLE_GATE = {
    "min_positive_recall": 0.90,
    "max_non_attribute_accept_rate": 0.05,
    "min_validation_precision": 0.99,
}


def assess_surface_role_gate(
    predictions: Iterable[dict[str, Any]],
    expected_roles: Iterable[str],
    *,
    validation_precision: float | None = None,
    training_acceptance: bool | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    requirements = dict(DEFAULT_SURFACE_ROLE_GATE)
    requirements.update(dict(policy or {}))
    rows = [dict(row) for row in predictions]
    roles = [str(role) for role in expected_roles]
    if len(rows) != len(roles):
        raise ValueError("surface role predictions and expected roles must have equal length")
    positive = [index for index, role in enumerate(roles) if role == "reflected_attribute"]
    negative = [index for index, role in enumerate(roles) if role != "reflected_attribute"]
    accepted = [
        bool(row.get("candidate_role") == "reflected_attribute" and not bool(row.get("abstained", True)))
        for row in rows
    ]
    positive_recall = sum(accepted[index] for index in positive) / len(positive) if positive else 0.0
    non_attribute_rate = sum(accepted[index] for index in negative) / len(negative) if negative else 0.0
    reasons: list[str] = []
    if training_acceptance is not True:
        reasons.append("training_stability_below_gate")
    if validation_precision is None or not math.isfinite(float(validation_precision)) or float(validation_precision) < float(requirements["min_validation_precision"]):
        reasons.append("validation_precision_below_gate")
    if positive_recall < float(requirements["min_positive_recall"]):
        reasons.append("positive_recall_below_gate")
    if non_attribute_rate > float(requirements["max_non_attribute_accept_rate"]):
        reasons.append("non_attribute_accept_rate_above_gate")
    return {
        "schema_version": SURFACE_ROLE_GATE_SCHEMA,
        "status": "enabled" if not reasons else "diagnostic_only",
        "enabled": not reasons,
        "reasons": reasons,
        "requirements": requirements,
        "metrics": {
            "sample_count": len(rows),
            "positive_count": len(positive),
            "positive_accept_count": sum(accepted[index] for index in positive),
            "positive_recall": positive_recall,
            "non_attribute_count": len(negative),
            "non_attribute_accept_count": sum(accepted[index] for index in negative),
            "non_attribute_accept_rate": non_attribute_rate,
            "validation_precision": validation_precision,
            "training_acceptance": training_acceptance,
        },
    }


__all__ = ["DEFAULT_SURFACE_ROLE_GATE", "SURFACE_ROLE_GATE_SCHEMA", "assess_surface_role_gate"]
