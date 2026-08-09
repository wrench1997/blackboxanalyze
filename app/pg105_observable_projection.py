"""PG-105 bounded causal projection for opaque local effects.

Some authorized fixtures can produce a typed-positive outcome while their
response shape is identical to the matched control.  This module adds only a
runtime-safe relation between anonymous input geometry and the already
bounded response-change bit.  It never stores field names, values, bodies,
oracle labels, routes, or family names.

The relation is a suspicion/abstention aid, not a vulnerability oracle:
``input_changed_response_unchanged`` must not become ``effect_present``.
Independent typed replay remains the only positive authority.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Mapping, Sequence

from .active_probe_signature import PROBE_IDS, model_input_has_forbidden_field, sha256_json


SCHEMA_VERSION = "bounded-causal-projection-v1"
_GEOMETRY_FIELDS = (
    "field_count",
    "boolean_count",
    "true_boolean_count",
    "numeric_count",
    "nonzero_numeric_count",
    "string_count",
    "empty_string_count",
    "length_bucket_sum",
    "numeric_bucket_sum",
)
_NUMERIC_BUCKETS = ("zero", "small", "medium", "large", "non_finite")
_RELATION_SAME_SAME = 0
_RELATION_INPUT_CHANGED_RESPONSE_UNCHANGED = 1
_RELATION_INPUT_UNCHANGED_RESPONSE_CHANGED = 2
_RELATION_BOTH_CHANGED = 3


def _clip(value: Any, *, limit: int = 128) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError("causal projection geometry must contain integers") from None
    if abs(number) > limit:
        raise ValueError("causal projection geometry is outside the bounded range")
    return number


def _length_bucket(value: Any) -> int:
    length = len(str(value))
    if length == 0:
        return 0
    if length <= 16:
        return 1
    if length <= 64:
        return 2
    if length <= 255:
        return 3
    return 4


def _numeric_bucket(value: Any) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 4
    if number != number or number in {float("inf"), float("-inf")}:
        return 4
    magnitude = abs(number)
    if magnitude == 0:
        return 0
    if magnitude < 10:
        return 1
    if magnitude < 100:
        return 2
    return 3


def _numeric_string_bucket(value: str) -> int | None:
    text = str(value).strip()
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
        return None
    return _numeric_bucket(text)


def bounded_request_geometry(values: Mapping[str, Any]) -> dict[str, Any]:
    """Project only anonymous, bounded value-shape statistics."""

    if not isinstance(values, Mapping):
        raise ValueError("request geometry requires a mapping")
    counts = {field: 0 for field in _GEOMETRY_FIELDS}
    bucket_counts = [0] * len(_NUMERIC_BUCKETS)
    counts["field_count"] = min(64, len(values))
    for value in list(values.values())[:64]:
        if isinstance(value, bool):
            counts["boolean_count"] += 1
            counts["true_boolean_count"] += int(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            counts["numeric_count"] += 1
            counts["nonzero_numeric_count"] += int(abs(float(value)) > 1e-9)
            bucket = _numeric_bucket(value)
            bucket_counts[bucket] += 1
            counts["numeric_bucket_sum"] += bucket
        else:
            counts["string_count"] += 1
            counts["empty_string_count"] += int(str(value) == "")
            numeric_bucket = _numeric_string_bucket(str(value))
            if numeric_bucket is not None:
                # Form/query transports often expose numbers as strings.  The
                # bucket is anonymous and bounded; the original text is never
                # retained.
                counts["numeric_count"] += 1
                counts["nonzero_numeric_count"] += int(numeric_bucket != 0)
                bucket_counts[numeric_bucket] += 1
                counts["numeric_bucket_sum"] += numeric_bucket
        counts["length_bucket_sum"] += _length_bucket(value)
    return {
        "schema_version": SCHEMA_VERSION,
        "geometry": {field: min(128, int(value)) for field, value in counts.items()},
        "numeric_bucket_counts": [min(64, int(value)) for value in bucket_counts],
    }


def make_causal_projection(
    control_values: Mapping[str, Any],
    candidate_values: Mapping[str, Any],
    *,
    response_changed: bool,
) -> dict[str, Any]:
    """Create a label-free input/response relation for one matched pair."""

    control = bounded_request_geometry(control_values)
    candidate = bounded_request_geometry(candidate_values)
    control_geometry = control["geometry"]
    candidate_geometry = candidate["geometry"]
    shape_delta = {
        field: max(-128, min(128, int(candidate_geometry[field]) - int(control_geometry[field])))
        for field in _GEOMETRY_FIELDS
    }
    input_changed = any(value != 0 for value in shape_delta.values()) or candidate["numeric_bucket_counts"] != control["numeric_bucket_counts"]
    response_changed = bool(response_changed)
    if input_changed and response_changed:
        relation = _RELATION_BOTH_CHANGED
    elif input_changed:
        relation = _RELATION_INPUT_CHANGED_RESPONSE_UNCHANGED
    elif response_changed:
        relation = _RELATION_INPUT_UNCHANGED_RESPONSE_CHANGED
    else:
        relation = _RELATION_SAME_SAME
    projection = {
        "schema_version": SCHEMA_VERSION,
        "input_changed": bool(input_changed),
        "response_changed": response_changed,
        "input_changed_response_unchanged": bool(input_changed and not response_changed),
        "relation_code": relation,
        "input_shape_delta": shape_delta,
        "numeric_bucket_delta": [
            max(-64, min(64, int(candidate["numeric_bucket_counts"][index]) - int(control["numeric_bucket_counts"][index])))
            for index in range(len(_NUMERIC_BUCKETS))
        ],
    }
    if model_input_has_forbidden_field(projection):
        raise ValueError("causal projection leaked an evaluator or raw field")
    return projection


def attach_causal_extension(
    signature: Mapping[str, Any],
    projections: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Attach a bounded extension to a complete active-probe signature."""

    if len(projections) != len(PROBE_IDS):
        raise ValueError("causal extension requires one projection per probe")
    if model_input_has_forbidden_field(signature):
        raise ValueError("signature leaked an evaluator or raw field")
    normalized = [copy.deepcopy(dict(item)) for item in projections]
    for item in normalized:
        if str(item.get("schema_version", "")) != SCHEMA_VERSION:
            raise ValueError("causal extension contains an unknown projection schema")
        if model_input_has_forbidden_field(item):
            raise ValueError("causal extension leaked an evaluator or raw field")
        if int(item.get("relation_code", -1)) not in range(4):
            raise ValueError("causal extension relation code is outside the bounded vocabulary")
    value = copy.deepcopy(dict(signature))
    value.pop("signature_sha256", None)
    response_pattern = [bool(item) for item in list(value.get("delta_pattern") or [])]
    if len(response_pattern) != len(PROBE_IDS):
        raise ValueError("signature requires the canonical response delta pattern")
    suspicious_pattern = [bool(item.get("input_changed_response_unchanged", False)) for item in normalized]
    value["causal_extension"] = {
        "schema_version": SCHEMA_VERSION,
        "input_changed_pattern": [bool(item.get("input_changed", False)) for item in normalized],
        "input_changed_response_unchanged_pattern": suspicious_pattern,
        "relation_code_pattern": [int(item["relation_code"]) for item in normalized],
        "input_shape_delta_pattern": [
            [int(item["input_shape_delta"].get(field, 0)) for field in _GEOMETRY_FIELDS]
            for item in normalized
        ],
        "numeric_bucket_delta_pattern": [list(item.get("numeric_bucket_delta") or [0] * len(_NUMERIC_BUCKETS)) for item in normalized],
    }
    # Attention includes a generic anomaly, but effect_present remains tied to
    # the response delta pattern in the Rule IR composition layer.
    value["attention_pattern"] = [
        bool(response_pattern[index] or suspicious_pattern[index])
        for index in range(len(PROBE_IDS))
    ]
    if model_input_has_forbidden_field(value):
        raise ValueError("causal extension leaked an evaluator or raw field")
    value["signature_sha256"] = sha256_json(value)
    return value


__all__ = [
    "SCHEMA_VERSION",
    "attach_causal_extension",
    "bounded_request_geometry",
    "make_causal_projection",
]
