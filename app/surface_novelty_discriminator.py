"""Bounded surface-novelty/OOD discriminator for PG-99.

This is an intentionally conservative component.  It learns a support set of
generic response-surface fingerprints and abstains on fingerprints not seen
in the design source.  It does not infer a vulnerability family and cannot
turn an in-domain fingerprint into a positive finding.  PG-99 also audits
whether known and unknown typed positives share exactly the same fingerprint;
that overlap is an impossibility witness for any classifier restricted to the
same visible projection.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "bounded-surface-novelty-v1"
_FORBIDDEN_FIELDS = {
    "family",
    "hypothesis",
    "oracle",
    "oracle_projection",
    "decision",
    "belief_before",
    "belief_after",
    "next_action",
    "target_instance_id",
    "route_template_id",
    "probe_ref",
    "probe_sha256",
    "body_sha256",
    "semantic_body_sha256",
    "projection_sha256",
    "marker",
}
_SHAPE_FIELDS = ("array_count", "bool_count", "key_count", "number_count", "scalar_count", "string_count")


def _bounded_token(value: Any) -> str:
    text = str(value)
    if not re.fullmatch(r"[A-Z0-9_:-]{1,160}", text):
        raise ValueError("surface novelty received an unbounded token")
    return text


def _summary(projection: Mapping[str, Any]) -> dict[str, Any]:
    shape = projection.get("shape") or {}
    return {
        "status_class": str(projection.get("status_class", "")),
        "content_type_class": str(projection.get("content_type_class", "")),
        "body_length_bucket": str(projection.get("body_length_bucket", "")),
        "frame_policy": str(projection.get("frame_policy", "")),
        "header_count": len(projection.get("header_names") or []),
        "location_origin_changed": bool(projection.get("location_origin_changed", False)),
        "state_changed": bool(projection.get("state_changed", False)),
        "transport_error": bool(projection.get("transport_error", False)),
        "shape": {field: shape.get(field) for field in _SHAPE_FIELDS},
    }


def make_surface_observation(control_projection: Mapping[str, Any], candidate_projection: Mapping[str, Any], *, method: str, encoding_class: str, phase: str, safe_probe: bool) -> dict[str, Any]:
    """Build a label-free bounded observation for one matched pair."""

    if method not in {"GET", "POST"}:
        raise ValueError("surface novelty only accepts GET/POST")
    if not re.fullmatch(r"(?:identity|url_percent|html_entity|json_escape)(?:->(?:identity|url_percent|html_entity|json_escape))*", encoding_class):
        raise ValueError("encoding class is outside the bounded vocabulary")
    if phase not in {"screen", "confirm", "error", "timeout"}:
        raise ValueError("phase is outside the bounded vocabulary")
    return {
        "schema_version": "bounded-surface-observation-v1",
        "method": method,
        "encoding_class": encoding_class,
        "phase": phase,
        "safe_probe": bool(safe_probe),
        "baseline": _summary(control_projection),
        "candidate": _summary(candidate_projection),
    }


def observation_fingerprint(observation: Mapping[str, Any]) -> str:
    forbidden = _FORBIDDEN_FIELDS.intersection(observation)
    if forbidden:
        raise ValueError(f"surface novelty observation leaked evaluator fields: {sorted(forbidden)}")
    payload = json.dumps(observation, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class SurfaceNoveltyDiscriminator:
    """Exact bounded-support OOD gate; it never emits a family label."""

    def __init__(self) -> None:
        self._support: set[str] = set()

    def fit(self, observations: Sequence[Mapping[str, Any]]) -> "SurfaceNoveltyDiscriminator":
        if not observations:
            raise ValueError("surface novelty requires a non-empty design set")
        self._support = {observation_fingerprint(observation) for observation in observations}
        return self

    @property
    def support_size(self) -> int:
        return len(self._support)

    def predict(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        if not self._support:
            raise RuntimeError("surface novelty discriminator is not fitted")
        fingerprint = observation_fingerprint(observation)
        if fingerprint in self._support:
            return {"decision": "in_domain", "abstain": False, "fingerprint": fingerprint}
        return {"decision": "novel_surface", "abstain": True, "fingerprint": fingerprint}


__all__ = ["SCHEMA_VERSION", "SurfaceNoveltyDiscriminator", "make_surface_observation", "observation_fingerprint"]
