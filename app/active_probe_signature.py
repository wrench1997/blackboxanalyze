"""Family-agnostic active-probe signatures for PG-101.

The signature is built from a bounded bank of safe probes.  The model sees
which canonical probe was sent and whether the generic response geometry
changed; it never sees a family name, route, oracle verdict, raw body, or
field name.  A small support decoder is intentionally used as a baseline
before a neural set/sequence decoder is trained.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "bounded-active-probe-signature-v1"
PROBE_IDS = tuple(f"p{index}" for index in range(9))
_METHODS = frozenset({"GET", "POST"})
_PHASES = frozenset({"screen", "confirm", "error", "timeout"})
_ENCODINGS = frozenset({"identity", "url_percent", "html_entity", "json_escape"})
_GEOMETRY_FIELDS = (
    "object_count",
    "array_count",
    "array_item_count",
    "boolean_count",
    "true_boolean_count",
    "numeric_count",
    "nonzero_numeric_count",
    "string_count",
    "string_length_bucket_sum",
    "leaf_count",
    "max_depth",
)
_FORBIDDEN = frozenset({
    "family",
    "hypothesis",
    "oracle",
    "oracle_id",
    "positive",
    "positive_authority",
    "confirmed_effect",
    "decision",
    "route_template_id",
    "target_instance_id",
    "source_id",
    "raw_body",
    "raw_payload",
    "body_sha256",
    "semantic_body_sha256",
    "projection_sha256",
    "evidence_sha256",
})


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _bounded_int(value: Any, *, limit: int = 128) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError("active probe geometry must contain integers") from None
    if abs(number) > limit:
        raise ValueError("active probe geometry integer is outside the bounded range")
    return number


def _geometry_delta(control: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for field in _GEOMETRY_FIELDS:
        result[field] = _bounded_int(candidate.get(field, 0)) - _bounded_int(control.get(field, 0))
    return result


def _bucket(value: Any) -> int:
    return {
        "0": 0,
        "1-255": 1,
        "256-4095": 2,
        "4096-65535": 3,
        "65536+": 4,
    }.get(str(value), -1)


def make_probe_observation(
    *,
    probe_id: str,
    method: str,
    phase: str,
    encoding: str,
    control_geometry: Mapping[str, Any],
    candidate_geometry: Mapping[str, Any],
    control_projection: Mapping[str, Any],
    candidate_projection: Mapping[str, Any],
    safe_probe: bool,
) -> dict[str, Any]:
    """Create one label-free observation for a matched control/candidate pair."""

    probe_id = str(probe_id)
    method = str(method).upper()
    phase = str(phase)
    encoding = str(encoding)
    if probe_id not in PROBE_IDS:
        raise ValueError("active probe id is outside the fixed bank")
    if method not in _METHODS or phase not in _PHASES or encoding not in _ENCODINGS:
        raise ValueError("active probe transport metadata is outside the bounded vocabulary")
    delta = _geometry_delta(control_geometry, candidate_geometry)
    shape_control = control_projection.get("shape") or {}
    shape_candidate = candidate_projection.get("shape") or {}
    shape_delta = tuple(
        _bounded_int(shape_candidate.get(field, 0), limit=256) - _bounded_int(shape_control.get(field, 0), limit=256)
        for field in ("key_count", "scalar_count", "array_count", "bool_count", "number_count", "string_count")
    )
    status_control = int(control_projection.get("status_code", 0) or 0)
    status_candidate = int(candidate_projection.get("status_code", 0) or 0)
    body_length_delta = _bucket(candidate_projection.get("body_length_bucket")) - _bucket(control_projection.get("body_length_bucket"))
    transport_delta = {
        "status_delta": max(-9, min(9, status_candidate // 100 - status_control // 100)),
        "status_changed": bool(status_candidate != status_control),
        "content_type_changed": str(control_projection.get("content_type_class", "")) != str(candidate_projection.get("content_type_class", "")),
        "body_length_bucket_delta": max(-4, min(4, body_length_delta)),
        "header_count_delta": max(-16, min(16, len(candidate_projection.get("header_names") or []) - len(control_projection.get("header_names") or []))),
        "shape_delta": shape_delta,
        "location_origin_changed": bool(candidate_projection.get("location_origin_changed", False)) != bool(control_projection.get("location_origin_changed", False)),
        "state_changed": bool(candidate_projection.get("state_changed", False)) != bool(control_projection.get("state_changed", False)),
        "transport_error": bool(candidate_projection.get("transport_error", False)),
    }
    geometry_changed = any(value != 0 for value in delta.values())
    response_changed = bool(
        geometry_changed
        or any(value != 0 for value in shape_delta)
        or transport_delta["status_changed"]
        or transport_delta["content_type_changed"]
        or transport_delta["body_length_bucket_delta"]
        or transport_delta["header_count_delta"]
        or transport_delta["location_origin_changed"]
        or transport_delta["state_changed"]
        or transport_delta["transport_error"]
    )
    observation = {
        "schema_version": SCHEMA_VERSION,
        "probe_id": probe_id,
        "method": method,
        "phase": phase,
        "encoding": encoding,
        "safe_probe": bool(safe_probe),
        "delta_nonzero": response_changed,
        "geometry_delta": delta,
        "transport_delta": transport_delta,
    }
    if model_input_has_forbidden_field(observation):
        raise ValueError("active probe observation leaked an evaluator or raw field")
    observation["observation_sha256"] = sha256_json(observation)
    return observation


def model_input_has_forbidden_field(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(str(key) in _FORBIDDEN or model_input_has_forbidden_field(child) for key, child in value.items())
    if isinstance(value, list):
        return any(model_input_has_forbidden_field(child) for child in value)
    return False


def aggregate_signature(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate one complete probe bank into a compact order-aware signature."""

    by_probe: dict[str, Mapping[str, Any]] = {}
    for observation in observations:
        probe_id = str(observation.get("probe_id", ""))
        if probe_id in by_probe:
            raise ValueError("active signature contains a duplicate probe")
        if probe_id not in PROBE_IDS:
            raise ValueError("active signature contains an unknown probe")
        if model_input_has_forbidden_field(observation):
            raise ValueError("active signature contains an evaluator field")
        by_probe[probe_id] = observation
    if set(by_probe) != set(PROBE_IDS):
        raise ValueError("active signature requires the complete fixed probe bank")
    first = by_probe[PROBE_IDS[0]]
    if any(str(item.get("method")) != str(first.get("method")) or str(item.get("phase")) != str(first.get("phase")) for item in by_probe.values()):
        raise ValueError("active signature mixes transport or phase contexts")
    signature = {
        "schema_version": SCHEMA_VERSION,
        "method": str(first.get("method")),
        "phase": str(first.get("phase")),
        "encoding": str(first.get("encoding")),
        "probe_order": list(PROBE_IDS),
        "delta_pattern": [bool(by_probe[probe_id].get("delta_nonzero")) for probe_id in PROBE_IDS],
        "geometry_sign_pattern": [
            [1 if int(value) > 0 else -1 if int(value) < 0 else 0 for value in (by_probe[probe_id].get("geometry_delta") or {}).values()]
            for probe_id in PROBE_IDS
        ],
    }
    signature["signature_sha256"] = sha256_json(signature)
    return signature


def signature_fingerprint(signature: Mapping[str, Any]) -> str:
    # The baseline intentionally uses only the canonical probe-response
    # pattern.  Fine geometry remains available for a later neural decoder,
    # but exact implementation-specific counts must not become a shortcut.
    value = {key: signature[key] for key in ("schema_version", "method", "phase", "encoding", "probe_order", "delta_pattern")}
    return sha256_json(value)


class ActiveProbeSignatureDecoder:
    """Exact-support baseline for active signatures; unknown support abstains."""

    def __init__(self) -> None:
        self._families: dict[str, set[str]] = defaultdict(set)

    def fit(self, rows: Sequence[Mapping[str, Any]], *, allowed_families: Sequence[str]) -> "ActiveProbeSignatureDecoder":
        allowed = {str(family) for family in allowed_families}
        if not allowed:
            raise ValueError("active decoder requires at least one known family")
        for row in rows:
            family = str(row.get("family", ""))
            if family not in allowed:
                raise ValueError("training row family is outside the declared known set")
            signature = row.get("signature")
            if not isinstance(signature, Mapping):
                raise ValueError("active decoder training row is missing a signature")
            self._families[family].add(signature_fingerprint(signature))
        if not any(self._families.values()):
            raise ValueError("active decoder received no signatures")
        return self

    def predict(self, signature: Mapping[str, Any]) -> dict[str, Any]:
        fingerprint = signature_fingerprint(signature)
        matches = sorted(family for family, fingerprints in self._families.items() if fingerprint in fingerprints)
        if len(matches) == 1:
            return {"decision": "candidate", "abstain": False, "candidate_family": matches[0], "signature_sha256": fingerprint}
        if not matches:
            return {"decision": "abstain", "abstain": True, "reason": "unseen_active_probe_signature", "signature_sha256": fingerprint}
        return {"decision": "abstain", "abstain": True, "reason": "ambiguous_active_probe_signature", "signature_sha256": fingerprint}


__all__ = [
    "ActiveProbeSignatureDecoder",
    "PROBE_IDS",
    "SCHEMA_VERSION",
    "aggregate_signature",
    "make_probe_observation",
    "model_input_has_forbidden_field",
    "sha256_json",
    "signature_fingerprint",
]
