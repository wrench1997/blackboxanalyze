"""PG-204 runtime token binding and fail-closed controller."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .detection_payload import validate_detection_payload
from .pg201_multitask_decoder import ENCODING_NAMES, FAILURE_NAMES
from .pg203_token_aware_decoder import TOKEN_FEATURE_DIM, token_features_for_row


PG204_SCHEMA = "pg204-runtime-token-binding-v1"
_ENCODING_KIND = {
    "http_canary": 0,
    "inert_dom_markup": 1,
    "encoded_dom_markup": 2,
    "sql_channel_class": 3,
}
_FAILURE_KIND = {
    "no_effect": 0,
    "status_changed": 1,
    "redirect_chain": 2,
    "post_validation": 3,
}
_FORBIDDEN_FIELDS = frozenset({"password", "passwd", "secret", "token", "csrf", "cookie", "authorization", "uploadfile"})


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _failure_kind(*, method: str, projection: Mapping[str, Any], typed_available: bool) -> str:
    status_class = str(projection.get("status_class", "other"))
    redirect_hops = projection.get("redirect_hop_count")
    try:
        has_redirect_hop = int(redirect_hops or 0) > 0
    except (TypeError, ValueError):
        has_redirect_hop = False
    if bool(projection.get("location_origin_changed")) or status_class == "3xx" or has_redirect_hop or bool(projection.get("redirect_chain")):
        return "redirect_chain"
    if status_class in {"4xx", "5xx", "transport_error"} or bool(projection.get("status_changed")):
        return "status_changed"
    if str(method).upper() == "POST" and not typed_available:
        return "post_validation"
    return "no_effect"


def build_runtime_token_packet(
    candidate: Mapping[str, Any] | None,
    *,
    route: Mapping[str, Any],
    failure_projection: Mapping[str, Any],
    typed_available: bool,
) -> dict[str, Any]:
    """Bind structural tokens to the exact candidate and observed route."""

    if candidate is None:
        raise ValueError("PG-204 candidate token is missing")
    payload = validate_detection_payload(dict(candidate.get("payload") or {}))
    method = str(route.get("method", "")).upper()
    path = str(route.get("path", ""))
    fields = sorted({str(item) for item in route.get("fields", []) if str(item)})
    if method not in {"GET", "POST"} or not path.startswith("/") or not fields:
        raise ValueError("PG-204 route binding is incomplete")
    if any(field.casefold() in _FORBIDDEN_FIELDS for field in fields):
        raise ValueError("PG-204 route binding contains a secret field")
    if payload["method"] != method or payload["path"] != path:
        raise ValueError("PG-204 candidate method/path binding mismatch")
    if method == "POST":
        form_fields = sorted({str(item) for item in dict(payload.get("form") or {})})
        if form_fields != fields:
            raise ValueError("PG-204 POST form field binding mismatch")
    encoding_index = _ENCODING_KIND.get(str(payload["probe_kind"]))
    if encoding_index is None:
        raise ValueError("PG-204 encoding token is missing")
    failure_kind = _failure_kind(method=method, projection=failure_projection, typed_available=typed_available)
    failure_index = _FAILURE_KIND[failure_kind]
    row = {
        "method": method,
        "redirect_hops": 0,
        "status_class": str(failure_projection.get("status_class", "2xx")),
        "candidate_signal": int(bool((failure_projection.get("marker") or {}).get("reflected", False))),
        "typed_available": int(bool(typed_available)),
        "negative_control": 1,
        "budget_remaining": 1,
        "failure_kind": failure_kind,
        "label": 2,
        "encoding_label": encoding_index,
        "failure_label": failure_index,
    }
    token_features = token_features_for_row(row)
    if len(token_features) != TOKEN_FEATURE_DIM:
        raise ValueError("PG-204 token packet dimension mismatch")
    binding = {
        "schema_version": PG204_SCHEMA,
        "route_method": method,
        "route_path": path,
        "route_fields": fields,
        "payload_sha256": str(payload["payload_sha256"]),
        "encoding_name": ENCODING_NAMES[encoding_index],
        "failure_name": FAILURE_NAMES[failure_index],
        "typed_available": bool(typed_available),
        "token_features": token_features,
    }
    binding["binding_sha256"] = _digest(binding)
    return binding


def validate_runtime_token_packet(
    packet: Mapping[str, Any] | None,
    *,
    candidate: Mapping[str, Any] | None,
    route: Mapping[str, Any],
    failure_projection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a token packet before the model or network sender is called."""

    if packet is None:
        return {"valid": False, "reason": "missing_token_packet", "network_allowed": False}
    try:
        expected = build_runtime_token_packet(
            candidate,
            route=route,
            failure_projection=dict(failure_projection or {"status_class": "2xx"}),
            typed_available=bool(packet.get("typed_available", False)),
        )
    except (TypeError, ValueError) as error:
        return {"valid": False, "reason": str(error), "network_allowed": False}
    packet_hash = str(packet.get("binding_sha256", ""))
    if packet_hash != expected["binding_sha256"]:
        return {"valid": False, "reason": "binding_hash_mismatch", "network_allowed": False}
    if list(packet.get("token_features") or []) != list(expected["token_features"]):
        return {"valid": False, "reason": "token_features_mismatch", "network_allowed": False}
    return {"valid": True, "reason": "bound", "network_allowed": True, "binding_sha256": packet_hash}


__all__ = ["PG204_SCHEMA", "build_runtime_token_packet", "validate_runtime_token_packet"]
