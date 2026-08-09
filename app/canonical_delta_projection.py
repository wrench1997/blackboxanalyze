"""Representation-invariant, evaluator-free response-delta projection.

PG-98 intentionally throws away surface-specific field names.  A bounded
numeric/shape change becomes ``DELTA_EFFECT_INCREASE`` or
``DELTA_EFFECT_DECREASE``; status/content changes remain a separate generic
response channel.  No family, probe text, marker, oracle projection, or raw
body enters this representation.
"""

from __future__ import annotations

from typing import Any, Mapping


SCHEMA_VERSION = "canonical-delta-projection-v1"
_SHAPE_FIELDS = ("array_count", "bool_count", "key_count", "number_count", "scalar_count", "string_count")
_SURFACE_FIELDS = ("array_field_count", "boolean_field_count", "nonzero_numeric_count", "numeric_field_count", "true_boolean_count")
_GEOMETRY_FIELDS = (
    "array_count",
    "array_item_count",
    "boolean_count",
    "leaf_count",
    "max_depth",
    "nonzero_numeric_count",
    "numeric_count",
    "object_count",
    "string_count",
    "string_length_bucket_sum",
    "true_boolean_count",
)


def _relation(before: Any, after: Any) -> str | None:
    if before == after:
        return None
    if isinstance(before, (int, float)) and isinstance(after, (int, float)) and not isinstance(before, bool) and not isinstance(after, bool):
        return "INCREASE" if after > before else "DECREASE"
    return "CHANGE"


def canonical_delta_tokens(control: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[str, ...]:
    """Project a matched pair into a small generic difference vocabulary."""

    tokens: set[str] = set()
    for field in ("status_class", "content_type_class", "body_length_bucket", "frame_policy"):
        relation = _relation(control.get(field), candidate.get(field))
        if relation:
            tokens.add(f"DELTA_RESPONSE_{relation}")
    for group_name, fields in (("shape", _SHAPE_FIELDS), ("effect_surface", _SURFACE_FIELDS), ("effect_geometry", _GEOMETRY_FIELDS)):
        before = control.get(group_name) or {}
        after = candidate.get(group_name) or {}
        for field in fields:
            relation = _relation(before.get(field), after.get(field))
            if relation:
                tokens.add(f"DELTA_EFFECT_{relation}")
    for field in ("location_origin_changed", "state_changed", "transport_error"):
        relation = _relation(control.get(field), candidate.get(field))
        if relation:
            tokens.add(f"DELTA_BOUNDARY_{relation}")
    return tuple(sorted(tokens))


__all__ = ["SCHEMA_VERSION", "canonical_delta_tokens"]
