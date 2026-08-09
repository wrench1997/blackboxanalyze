"""Deterministic, label-free active probe ranking for shadow observations."""

from __future__ import annotations

import math
from typing import Any, Iterable


def normalized_entropy(probabilities: dict[str, float] | Iterable[float]) -> float:
    values = [max(0.0, float(value)) for value in (probabilities.values() if isinstance(probabilities, dict) else probabilities)]
    total = sum(values)
    if total <= 0 or len(values) <= 1:
        return 0.0
    values = [value / total for value in values]
    entropy = -sum(value * math.log(value) for value in values if value > 0)
    return entropy / math.log(len(values))


def active_probe_score(row: dict[str, Any]) -> float:
    """Prefer ambiguous semantic responses, with response score as a tie-break signal."""

    decoder = dict(row.get("rule_ir_decoder") or {})
    probabilities = decoder.get("probabilities") or {}
    confidence = float(decoder.get("confidence", 0.0) or 0.0)
    entropy = normalized_entropy(probabilities)
    response_score = float(row.get("model_score", 0.0) or 0.0)
    # Entropy dominates: the next probe should separate competing abstract
    # families.  The small score term keeps a clearly informative response
    # preferred when two candidates are equally uncertain.
    return round(0.75 * entropy + 0.20 * (1.0 - confidence) + 0.05 * response_score, 6)


def choose_active_probe(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("active probe candidate list must not be empty")
    scored = [(active_probe_score(row), -index, row) for index, row in enumerate(candidates)]
    return max(scored, key=lambda item: (item[0], item[1]))[2]
