"""Independent typed-oracle projection for PG-53 cross-source replay.

PG-53 deliberately does not reuse the real Docker oracle implementation.  It
interprets the bounded effect contract exposed by two separately written
loopback fixtures (PG-35 and PG-36), while keeping the model-facing response
projection free of evaluator fields.  The fixtures accept only abstract probe
classes; this module never persists request values or response bodies.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import httpx


PG53_SCHEMA = "sift-pg53-cross-source-typed-oracle-v1"
ORACLE_CONTRACT_ID = "pg53-independent-typed-effect-contract-v1"
FAMILIES = (
    "xss",
    "injection",
    "authentication",
    "access_control",
    "logic",
    "url_redirect",
    "input_validation",
    "command_injection",
    "ordinary_response",
)

_SURFACE_OBSERVATION_EXCLUDED_KEYS = {
    "fixture",
    "variant",
    "implementation",
    "surface_slot",
    "candidate_signal",
    "phase",
    "ambiguous",
    "typed_effect_ready",
    "bounded_response_delta",
    "external_network",
    "script_execution",
    "database_touched",
    "database_write",
    "credentials_accessed",
    "state_mutated",
}


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8", errors="replace")).hexdigest()


def _status_class(status: int) -> str:
    return f"{status // 100}xx" if 100 <= int(status) <= 599 else "other"


def _length_bucket(length: int) -> str:
    length = max(0, int(length))
    if length == 0:
        return "0"
    if length <= 255:
        return "1-255"
    if length <= 4095:
        return "256-4095"
    return "4096-65535"


def _shape(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            "kind": "object",
            "key_count": len(value),
            "scalar_count": sum(not isinstance(child, (dict, list)) for child in value.values()),
            "array_count": sum(isinstance(child, list) for child in value.values()),
        }
    if isinstance(value, list):
        return {"kind": "array", "key_count": 0, "scalar_count": 0, "array_count": 1}
    return {"kind": type(value).__name__, "key_count": 0, "scalar_count": 1, "array_count": 0}


def response_projection(response: httpx.Response) -> dict[str, Any]:
    """Return a bounded visible projection; never return the response body."""

    body = bytes(response.content)
    try:
        parsed = response.json()
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = None
    shape = _shape(parsed)
    content_type = str(response.headers.get("content-type", "application/octet-stream")).split(";", 1)[0].casefold()
    projection = {
        "status_code": int(response.status_code),
        "status_class": _status_class(response.status_code),
        "content_type_class": "json" if content_type == "application/json" else "other",
        "body_length_bucket": _length_bucket(len(body)),
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "semantic_body_sha256": sha256_json(shape),
        "shape": shape,
        "header_names": sorted({str(key).casefold() for key in response.headers.keys()} & {"content-type", "location", "allow"}),
        "transport_error": False,
        "state_changed": False,
        "external_network": False,
    }
    projection["projection_sha256"] = sha256_json(projection)
    return projection


def surface_observation(body: dict[str, Any]) -> dict[str, Any]:
    """Build a bounded post-probe observation without oracle labels.

    This is an observation channel for the surface discriminator, not the
    typed oracle: family/positive/evaluator fields are removed, scalar effect
    counts are clipped, and only deterministic hashes of allowed key names are
    retained.  The raw body never leaves the caller's memory.
    """

    booleans: list[bool] = []
    numerics: list[float] = []
    arrays = 0
    key_buckets: list[int] = []
    for key, value in body.items():
        key_text = str(key)
        if key_text.casefold() in _SURFACE_OBSERVATION_EXCLUDED_KEYS:
            continue
        digest = hashlib.sha256(key_text.encode("utf-8", errors="replace")).digest()
        key_buckets.append(int.from_bytes(digest[:2], "big") % 64)
        if isinstance(value, bool):
            booleans.append(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            numerics.append(float(value))
        elif isinstance(value, list):
            arrays += 1
    observation = {
        "boolean_field_count": min(32, len(booleans)),
        "true_boolean_count": min(32, sum(booleans)),
        "numeric_field_count": min(32, len(numerics)),
        "nonzero_numeric_count": min(32, sum(abs(value) > 1e-9 for value in numerics)),
        "array_field_count": min(8, arrays),
        "key_hash_buckets": sorted(set(key_buckets))[:16],
        "observation_schema": "bounded_effect_shape_v1",
    }
    observation["observation_sha256"] = sha256_json(observation)
    return observation


def generic_effect_geometry(body: dict[str, Any]) -> dict[str, Any]:
    """Project response geometry without reading or retaining field names.

    This is deliberately different from ``surface_observation``: it records
    only recursive value-type counts, bounded depth and length buckets.  No
    key names, evaluator flags or modality-specific labels are used.  It is a
    candidate model observation; the funnel still decides whether it is
    stable enough to use.
    """

    counts = {
        "object_count": 0,
        "array_count": 0,
        "array_item_count": 0,
        "boolean_count": 0,
        "true_boolean_count": 0,
        "numeric_count": 0,
        "nonzero_numeric_count": 0,
        "string_count": 0,
        "string_length_bucket_sum": 0,
        "leaf_count": 0,
        "max_depth": 0,
    }

    def visit(value: Any, depth: int) -> None:
        counts["max_depth"] = max(counts["max_depth"], min(16, int(depth)))
        if isinstance(value, dict):
            counts["object_count"] = min(64, counts["object_count"] + 1)
            for child in value.values():
                visit(child, depth + 1)
            return
        if isinstance(value, list):
            counts["array_count"] = min(32, counts["array_count"] + 1)
            counts["array_item_count"] = min(64, counts["array_item_count"] + len(value))
            for child in value[:32]:
                visit(child, depth + 1)
            return
        counts["leaf_count"] = min(128, counts["leaf_count"] + 1)
        if isinstance(value, bool):
            counts["boolean_count"] = min(64, counts["boolean_count"] + 1)
            counts["true_boolean_count"] = min(64, counts["true_boolean_count"] + int(value))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            counts["numeric_count"] = min(64, counts["numeric_count"] + 1)
            counts["nonzero_numeric_count"] = min(64, counts["nonzero_numeric_count"] + int(abs(float(value)) > 1e-9))
        elif isinstance(value, str):
            counts["string_count"] = min(64, counts["string_count"] + 1)
            length = len(value)
            counts["string_length_bucket_sum"] = min(128, counts["string_length_bucket_sum"] + (0 if length == 0 else 1 if length <= 16 else 2 if length <= 64 else 3))

    visit(body, 0)
    geometry = {key: int(value) for key, value in counts.items()}
    geometry["geometry_schema"] = "anonymous_value_type_geometry_v1"
    geometry["geometry_sha256"] = sha256_json(geometry)
    return geometry


def _bool(body: dict[str, Any], *keys: str) -> bool:
    return any(bool(body.get(key)) for key in keys)


def _bounded_signals(body: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Copy only typed scalar signals from a known evaluator contract."""

    result: dict[str, Any] = {}
    for key in keys:
        if key not in body:
            continue
        value = body[key]
        if isinstance(value, bool):
            result[key] = value
        elif isinstance(value, int) and not isinstance(value, bool):
            result[key] = max(-64, min(64, int(value)))
        elif isinstance(value, str):
            result[key] = value[:32]
        elif isinstance(value, list):
            result[key] = [str(item)[:16] for item in value[:8]]
    return result


def _family_effect(family: str, body: dict[str, Any], *, source_id: str) -> tuple[bool, str, tuple[str, ...]]:
    """Interpret the same effect contract across the two independent sources."""

    if family == "xss":
        positive = _bool(body, "dom_change") and int(body.get("marker_hits", 0) or 0) > 0 and not _bool(body, "network_access")
        return positive, "browser_dom_effect", ("dom_change", "marker_hits", "network_access", "script_execution")
    if family == "injection":
        positive = _bool(body, "controlled_differential", "ast_shape_diff") and _bool(body, "interpreter_boundary") and not _bool(body, "database_write")
        return positive, "typed_ast_difference", ("controlled_differential", "ast_shape_diff", "interpreter_boundary", "ast_node_delta", "branch_delta", "database_write")
    if family == "authentication":
        positive = _bool(body, "authentication_boundary") and not _bool(body, "authenticated")
        return positive, "typed_authentication_boundary", ("authentication_boundary", "authenticated", "state_class")
    if family == "access_control":
        positive = _bool(body, "authorization_boundary") and _bool(body, "cross_subject_access")
        return positive, "typed_authorization_boundary", ("authorization_boundary", "cross_subject_access", "subject_delta", "subject_vector")
    if family == "logic":
        positive = _bool(body, "business_invariant_boundary") and not _bool(body, "history_changed")
        return positive, "typed_logic_invariant", ("business_invariant_boundary", "history_changed", "history_depth_delta", "counter_delta", "history_vector")
    if family == "url_redirect":
        positive = _bool(body, "redirect_candidate") and _bool(body, "same_origin") and not _bool(body, "external_redirect")
        return positive, "typed_redirect_projection", ("redirect_candidate", "same_origin", "external_redirect", "origin_class")
    if family == "input_validation":
        positive = _bool(body, "validation_boundary") and _bool(body, "rejected")
        return positive, "typed_validation_boundary", ("validation_boundary", "rejected", "rejected_field_count", "boundary_class")
    if family == "command_injection":
        positive = _bool(body, "local_canary_observed") and not _bool(body, "command_executed") and _bool(body, "execution_blocked")
        return positive, "typed_local_canary", ("local_canary_observed", "command_executed", "execution_blocked", "canary_count")
    return False, "negative_control", ("ordinary_response", "candidate_signal")


def typed_effect_oracle(
    *,
    source_id: str,
    family: str,
    body: dict[str, Any],
    response: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    """Evaluate one bounded response after the probe, fail-closed."""

    if family not in FAMILIES:
        raise ValueError(f"unknown PG-53 family: {family}")
    if stage == "screen":
        positive = False
        modality = "ambiguous_screen"
        signals = {"screen_only": True}
    else:
        positive, modality, signal_keys = _family_effect(family, body, source_id=source_id)
        signals = _bounded_signals(body, signal_keys)
    safety = {
        "external_network": bool(body.get("external_network", False)),
        "script_execution": bool(body.get("script_execution", False)),
        "database_write": bool(body.get("database_write", False)),
        "state_mutated": bool(body.get("state_mutated", False)),
        "credentials_accessed": bool(body.get("credentials_accessed", False)),
    }
    if any(safety.values()):
        positive = False
        modality = "safety_gate_rejected"
    if family == "ordinary_response":
        positive = False
        modality = "negative_control"
    oracle = {
        "oracle_id": f"pg53-{family}-typed-oracle-v1",
        "oracle_contract": ORACLE_CONTRACT_ID,
        "source_implementation": source_id,
        "family": family,
        "stage": str(stage),
        "modality": modality,
        "positive": bool(positive),
        "positive_authority": bool(positive),
        "signals": signals,
        "response_projection_sha256": response.get("projection_sha256", ""),
        "safety": safety,
    }
    oracle["evidence_projection_sha256"] = sha256_json(oracle)
    return oracle


def build_payload_manifest(
    *,
    source_id: str,
    surface: str,
    family: str,
    method: str,
    placement: str,
    probe_kind: str,
    probe_value: str,
    route_template_id: str,
    field_name: str,
    stage: str,
) -> dict[str, Any]:
    """Persist only hashes and abstract descriptors, never the request value."""

    manifest = {
        "manifest_id": f"pg53-{source_id}-{surface}-{method.casefold()}-{stage}",
        "payload_sha256": sha256_text(probe_value),
        "field_name_sha256": sha256_text(field_name),
        "probe_ref": "abstract-positive-class" if stage == "candidate" else "baseline-control",
        "probe_kind": probe_kind,
        "route_template_id": route_template_id,
        "method": method,
        "placement": placement,
        "encoding_chain": ["identity"],
        "encoding_depth": 0,
        "stage": stage,
        "family_label_for_evaluator_only": family,
        "max_bytes": 96,
        "safety": {
            "does_not_execute": True,
            "no_external_network": True,
            "no_database_write": True,
            "no_credential_access": True,
            "loopback_only": True,
        },
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    return manifest


__all__ = [
    "FAMILIES",
    "ORACLE_CONTRACT_ID",
    "PG53_SCHEMA",
    "build_payload_manifest",
    "response_projection",
    "surface_observation",
    "generic_effect_geometry",
    "sha256_json",
    "sha256_text",
    "typed_effect_oracle",
]
