"""Small, dependency-light confidence and evidence calibration helpers.

The decoder's class probability answers "which abstract family looks most
like this trace?".  A separate bounded oracle answers "does that family have
the evidence needed to emit a Rule IR?".  These helpers keep the two signals
separate, calibrate the class probability on a held-out set, and expose a
fail-closed evidence gate.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def _clip(value: float, epsilon: float = 1e-6) -> float:
    return min(1.0 - epsilon, max(epsilon, float(value)))


def _logit(value: float) -> float:
    probability = _clip(value)
    return math.log(probability / (1.0 - probability))


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def multiclass_nll(probabilities: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have equal length")
    if not labels:
        return 0.0
    losses: list[float] = []
    for row, label in zip(probabilities, labels):
        if not row or not 0 <= int(label) < len(row):
            raise ValueError("multiclass label is outside probability row")
        total = sum(max(0.0, float(value)) for value in row)
        if total <= 0:
            raise ValueError("probability row must have positive mass")
        losses.append(-math.log(_clip(float(row[int(label)]) / total)))
    return sum(losses) / len(losses)


def temperature_scale(probabilities: Sequence[Sequence[float]], temperature: float) -> list[list[float]]:
    """Apply temperature scaling to already-normalized class probabilities."""

    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled: list[list[float]] = []
    for row in probabilities:
        if not row:
            scaled.append([])
            continue
        logits = [math.log(_clip(float(value))) / temperature for value in row]
        pivot = max(logits)
        exponentials = [math.exp(value - pivot) for value in logits]
        total = sum(exponentials)
        scaled.append([value / total for value in exponentials])
    return scaled


def fit_temperature(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    candidates: Iterable[float] | None = None,
) -> dict[str, float]:
    """Fit a deterministic scalar temperature by held-out multiclass NLL."""

    grid = tuple(float(value) for value in (candidates or (
        0.25, 0.35, 0.50, 0.65, 0.80, 1.00, 1.25, 1.50, 2.00, 3.00, 4.00,
    )))
    if not grid or any(value <= 0 for value in grid):
        raise ValueError("temperature candidate grid must contain positive values")
    before = multiclass_nll(probabilities, labels)
    scored = [(multiclass_nll(temperature_scale(probabilities, value), labels), value) for value in grid]
    best_nll, best_temperature = min(scored, key=lambda item: (item[0], abs(item[1] - 1.0)))
    return {
        "temperature": float(best_temperature),
        "nll_before": float(before),
        "nll_after": float(best_nll),
    }


def expected_calibration_error(
    confidences: Sequence[float],
    correctness: Sequence[bool | int | float],
    *,
    bins: int = 10,
) -> float:
    if len(confidences) != len(correctness):
        raise ValueError("confidence and correctness lengths must match")
    if bins <= 0:
        raise ValueError("calibration bins must be positive")
    if not confidences:
        return 0.0
    counts = [0] * bins
    confidence_sum = [0.0] * bins
    accuracy_sum = [0.0] * bins
    for confidence, correct in zip(confidences, correctness):
        probability = min(1.0, max(0.0, float(confidence)))
        index = min(bins - 1, int(probability * bins))
        counts[index] += 1
        confidence_sum[index] += probability
        accuracy_sum[index] += float(bool(correct))
    total = len(confidences)
    return sum(
        (counts[index] / total)
        * abs(confidence_sum[index] / counts[index] - accuracy_sum[index] / counts[index])
        for index in range(bins) if counts[index]
    )


def binary_brier_score(confidences: Sequence[float], labels: Sequence[bool | int | float]) -> float:
    if len(confidences) != len(labels):
        raise ValueError("confidence and labels lengths must match")
    if not confidences:
        return 0.0
    return sum((float(confidence) - float(bool(label))) ** 2 for confidence, label in zip(confidences, labels)) / len(confidences)


def family_oracle_support(family: str, projection: dict[str, object]) -> float:
    """Return bounded support for a candidate family from its own oracle only."""

    if family == "xss":
        # Reflection alone is not an executable sink.  Keep text/JSON/header
        # echoes below the acceptance threshold; only a bounded attribute or
        # script-source sink receives strong family-specific support.
        if bool(projection.get("marker_in_script_source")):
            return 1.0
        if bool(projection.get("marker_in_attribute")):
            return 0.96
        if bool(projection.get("marker_in_html_text")):
            return 0.16
        if bool(projection.get("marker_in_json_value")) or bool(projection.get("marker_in_header")):
            return 0.08
        if bool(projection.get("marker_reflected")):
            return 0.12
        return 0.02
    if family == "injection":
        error = bool(projection.get("sql_error_shape"))
        shape_delta = float(projection.get("body_length_delta_abs", 0) or 0) >= 256
        if bool(projection.get("controlled_differential")) and bool(projection.get("interpreter_boundary")):
            modality = str(projection.get("modality", ""))
            return {
                "syntax_error": 0.92,
                "bounded_timing": 0.90,
                "blind_response": 0.86,
                "local_side_channel": 0.82,
                "ast_shape": 0.78,
            }.get(modality, 0.72)
        return 1.0 if error and shape_delta else 0.88 if error else 0.68 if shape_delta else 0.02
    if family == "url_redirect":
        return 0.96 if bool(projection.get("external_redirect")) else 0.02
    # No bounded evidence is currently authorized for these families in the
    # Pikachu HTTP track; fail closed until a family-specific oracle exists.
    return 0.02


def evidence_fused_confidence(model_probability: float, oracle_support: float, *, temperature: float = 1.0) -> float:
    """Fuse model confidence and bounded evidence without allowing either to hide the other."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return _sigmoid((_logit(model_probability) + _logit(oracle_support)) / float(temperature))


def accept_with_evidence(
    *,
    calibrated_confidence: float,
    oracle_support: float,
    confidence_threshold: float = 0.70,
    evidence_threshold: float = 0.50,
) -> bool:
    return bool(
        calibrated_confidence >= float(confidence_threshold)
        and oracle_support >= float(evidence_threshold)
    )


__all__ = [
    "accept_with_evidence",
    "binary_brier_score",
    "evidence_fused_confidence",
    "expected_calibration_error",
    "family_oracle_support",
    "fit_temperature",
    "multiclass_nll",
    "temperature_scale",
]
