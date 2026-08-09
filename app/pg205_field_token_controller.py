"""PG-205 binding of request/response field tokens to the runtime route."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .pg204_token_binding_controller import build_runtime_token_packet
from .pg205_request_response_tokens import FIELD_TOKEN_SCHEMA, field_tokens_for_runtime


PG205_BINDING_SCHEMA = "pg205-field-token-binding-v1"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def build_field_token_packet(
    candidate: Mapping[str, Any] | None,
    *,
    route: Mapping[str, Any],
    response_projection: Mapping[str, Any],
    typed_available: bool,
    redirect_hops: int | None = None,
) -> dict[str, Any]:
    """Create a hash-bound packet that joins route fields to response shape."""

    runtime = build_runtime_token_packet(
        candidate,
        route=route,
        failure_projection=response_projection,
        typed_available=typed_available,
    )
    fields = sorted({str(item) for item in route.get("fields", []) if str(item)})
    field_tokens = field_tokens_for_runtime(
        method=str(route.get("method", "GET")),
        field_names=fields,
        projection=response_projection,
        typed_available=typed_available,
        redirect_hops=redirect_hops,
    )
    binding = {
        "schema_version": PG205_BINDING_SCHEMA,
        "field_token_schema": FIELD_TOKEN_SCHEMA,
        "runtime_binding_sha256": runtime["binding_sha256"],
        "route_method": runtime["route_method"],
        "route_path": runtime["route_path"],
        "route_fields": fields,
        "encoding_name": runtime["encoding_name"],
        "failure_name": runtime["failure_name"],
        "response_projection_sha256": str(response_projection.get("projection_sha256", "")),
        "redirect_hop_count": int(redirect_hops if redirect_hops is not None else response_projection.get("redirect_hop_count", 0) or 0),
        "typed_available": bool(typed_available),
        "field_tokens": field_tokens,
    }
    binding["field_token_sha256"] = _digest(field_tokens)
    binding["binding_sha256"] = _digest(binding)
    return binding


def validate_field_token_packet(
    packet: Mapping[str, Any] | None,
    *,
    candidate: Mapping[str, Any] | None,
    route: Mapping[str, Any],
    response_projection: Mapping[str, Any],
    typed_available: bool,
    redirect_hops: int | None = None,
) -> dict[str, Any]:
    """Reject missing or stale request/response tokens before model/network use."""

    if packet is None:
        return {"valid": False, "reason": "missing_field_token_packet", "network_allowed": False}
    if not packet.get("field_tokens"):
        return {"valid": False, "reason": "missing_field_tokens", "network_allowed": False}
    try:
        expected = build_field_token_packet(
            candidate,
            route=route,
            response_projection=response_projection,
            typed_available=typed_available,
            redirect_hops=redirect_hops,
        )
    except (TypeError, ValueError) as error:
        return {"valid": False, "reason": str(error), "network_allowed": False}
    if str(packet.get("binding_sha256", "")) != expected["binding_sha256"]:
        return {"valid": False, "reason": "field_binding_hash_mismatch", "network_allowed": False}
    if list(packet.get("field_tokens") or []) != list(expected["field_tokens"]):
        return {"valid": False, "reason": "field_token_mismatch", "network_allowed": False}
    if str(packet.get("field_token_sha256", "")) != expected["field_token_sha256"]:
        return {"valid": False, "reason": "field_token_hash_mismatch", "network_allowed": False}
    return {"valid": True, "reason": "field_tokens_bound", "network_allowed": True, "binding_sha256": expected["binding_sha256"]}


__all__ = ["PG205_BINDING_SCHEMA", "build_field_token_packet", "validate_field_token_packet"]
