"""Representation-invariant token funnel for bounded effect observations.

The funnel removes implementation-specific absolute cardinalities (for
example one fixture exposing one boolean field and another exposing two) while
preserving whether a bounded field is empty/present and whether a matched
effect delta is zero/positive/negative/unknown.  It never consumes family,
oracle, route labels, raw probes or response bodies.
"""

from __future__ import annotations

from typing import Iterable


SCHEMA_VERSION = "invariant-token-funnel-v1"


def _presence(value: str) -> str:
    normalized = str(value).casefold()
    if normalized in {"0", "zero", "empty"}:
        return "ZERO"
    if normalized in {"u", "unknown", "missing"}:
        return "UNKNOWN"
    return "PRESENT"


def canonicalize_token(token: str) -> str:
    """Map one internal PG-86 token to the invariant bounded vocabulary."""

    value = str(token)
    # Absolute response length is transport noise across independent
    # implementations; retain only empty/non-empty/unknown.
    if "_LENGTH_" in value and "_DIFF_" not in value:
        prefix, suffix = value.rsplit("_LENGTH_", 1)
        return f"{prefix}_LENGTH_{_presence(suffix)}"

    # Matched deltas retain direction but not the field name.  This prevents a
    # novel implementation from becoming an OOD token merely because its
    # effect adds a string rather than a boolean field.
    if "_DIFF_" in value:
        parts = value.split("_")
        for category in ("SURFACE", "GEOMETRY"):
            if category in parts:
                index = parts.index(category)
                if index + 1 < len(parts):
                    sign = parts[-1]
                    return "_".join(parts[: index + 1] + ["ANY", sign])
        return value

    # Absolute bounded surface/geometry counts retain only presence.  Other
    # response tokens (status/content/modality/rule) are unchanged.
    for category in ("SURFACE", "GEOMETRY"):
        marker = f"_{category}_"
        if marker in value and "_EFFECT_MISSING" not in value:
            prefix, suffix = value.rsplit(marker, 1)
            if "_" in suffix:
                field, raw = suffix.rsplit("_", 1)
                return f"{prefix}{marker}{field}_{_presence(raw)}"
    return value


def canonicalize_tokens(tokens: Iterable[str]) -> list[str]:
    return [canonicalize_token(token) for token in tokens]


__all__ = ["SCHEMA_VERSION", "canonicalize_token", "canonicalize_tokens"]
