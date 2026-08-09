"""Pure in-memory structural adapter for the PG-246 VulnerableApp lanes.

The caller may provide already-authorised, de-identified markup, response
headers and a request projection.  Markup is parsed only while this function
runs; the result contains abstract seven-axis observations and a field status
manifest, never markup, URLs, parameter values, response bodies or headers.
No transport, container, browser, or evaluator action is implemented here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from .pg331_loopback_adapter import _PageParser, _content_type, _field_capture_manifest, _status_class


SCHEMA_VERSION = "pg331-vulnerableapp-structural-adapter-v1"
AXES = ("document_structure", "navigation", "request_transport", "response_transport", "javascript_surface", "failure_feedback", "belief_and_replay")
RAW_KEYS = frozenset({"payload", "raw_payload", "request_body", "response_body", "raw_response", "body_text", "url", "origin", "path", "location_url", "cookie", "authorization", "credential", "oracle_answer", "evaluator_answer"})
METHODS = frozenset({"GET", "POST"})


def _reject_raw(value: Any, key: str = "", *, root_html: bool = False) -> None:
    name = str(key).casefold()
    if name in RAW_KEYS or name.startswith("raw_"):
        raise ValueError("PG-331 VulnerableApp adapter rejects raw/literal fields")
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            _reject_raw(child, str(child_key))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _reject_raw(child, key)
    elif isinstance(value, (bytes, bytearray)):
        raise ValueError("PG-331 VulnerableApp adapter accepts markup only as ephemeral text")
    _ = root_html


def _unknown_axes() -> dict[str, Any]:
    return {axis: None for axis in AXES}


def _parameters(value: Any) -> list[dict[str, str]]:
    if value in (None, []):
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("request_projection.parameters must be an abstract list")
    result: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) - {"role", "value_type", "presence"}:
            raise ValueError("request_projection parameter must contain only abstract keys")
        role = str(item.get("role", "unknown"))
        value_type = str(item.get("value_type", "text"))
        presence = str(item.get("presence", "present"))
        if not role.replace("_", "").isalnum() or not value_type.replace("_", "").isalnum() or presence not in {"present", "absent", "unknown"}:
            raise ValueError("request_projection parameter is not abstract")
        result.append({"role": role.casefold(), "value_type": value_type.casefold(), "presence": presence})
    return result


def _response_projection(value: Mapping[str, Any], headers: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "status", "body_length", "body_shape", "connection_outcome", "failure_class",
        "failure_stage", "error_shape", "charset_class", "cache_shape",
        "csrf_presence_class", "environment_failure_class", "timeout_ms",
    }
    if set(value) - allowed:
        raise ValueError("response_projection contains unsupported fields")
    status = value.get("status")
    if not isinstance(status, int) or isinstance(status, bool) or not 100 <= status < 600:
        raise ValueError("response_projection.status must be an HTTP status")
    length = value.get("body_length", 0)
    if not isinstance(length, int) or isinstance(length, bool) or length < 0 or length > 2 * 1024 * 1024:
        raise ValueError("response_projection.body_length is invalid")
    redirect = 300 <= status < 400
    content_type = _content_type(headers)
    charset = str(value.get("charset_class", "absent" if headers else "unknown"))
    cache_shape = str(value.get("cache_shape", "absent" if headers else "unknown"))
    csrf = str(value.get("csrf_presence_class", "absent" if status else "unknown"))
    return {"status_class": _status_class(status), "status_shape": "numeric", "content_type_class": content_type, "connection_outcome": str(value.get("connection_outcome", "complete")), "body_length": length, "body_shape": str(value.get("body_shape", "empty")), "charset_class": charset, "header_presence_class": "basic" if headers else "absent", "cache_shape": cache_shape, "redirect_hop_count": 1 if redirect else 0, "redirect_location_class": "present" if redirect and any(str(key).casefold() == "location" for key in headers) else "absent" if redirect else "none", "redirect_chain_shape": "single_hop" if redirect else "empty", "failure_class": str(value.get("failure_class", "none")), "failure_stage": str(value.get("failure_stage", "none")), "error_shape": str(value.get("error_shape", "empty")), "path": "", "query_key_count": 0, "query_key_shapes": [], "request_content_length": 0, "csrf_presence_class": csrf, "environment_failure_class": str(value.get("environment_failure_class", "none")), "timeout_ms": int(value.get("timeout_ms", 0) or 0)}


def capture_vulnerableapp_projection(*, html: str | None, headers: Mapping[str, Any] | None, request_projection: Mapping[str, Any] | None, response_projection: Mapping[str, Any] | None, post_supported: bool = False, failure_projection: Mapping[str, Any] | None = None, belief_projection: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Convert de-identified local projections into strict abstract axes.

    Missing markup/request/response is represented as ``None`` for its axes,
    which the ontology tokenizer turns into explicit ``not_observed`` fields.
    A PG-246 POST remains typed-unavailable and ASK-safe regardless of an
    otherwise well-formed response.
    """
    if html is not None and not isinstance(html, str):
        raise ValueError("html must be text or None")
    if headers is not None and not isinstance(headers, Mapping):
        raise ValueError("headers must be an object or None")
    for value, name in ((headers, "headers"), (request_projection, "request_projection"), (response_projection, "response_projection")):
        if value is not None:
            _reject_raw(value, name)
    if html is not None and len(html) > 2 * 1024 * 1024:
        raise ValueError("html exceeds bounded in-memory limit")

    observation = _unknown_axes()
    request: dict[str, Any] | None = None
    if request_projection is not None:
        if set(request_projection) - {"method", "parameters", "csrf_presence_class", "cookie_presence_class", "content_length"}:
            raise ValueError("request_projection contains unsupported fields")
        method = str(request_projection.get("method", "")).upper()
        if method not in METHODS:
            raise ValueError("request_projection.method must be GET or POST")
        parameters = _parameters(request_projection.get("parameters"))
        request = {"method": method, "parameters": parameters}

    response = _response_projection(dict(response_projection), dict(headers or {})) if response_projection is not None else None
    if html is not None:
        parser = _PageParser(urlsplit("http://127.0.0.1/"))
        parser.feed(html)
        parser.close()
        parsed = parser.observation_projection(request_method=request["method"] if request else "GET", request_data=None, query_data=None, response=response or {"status_class": "unknown", "status_shape": "unknown", "content_type_class": "unknown", "connection_outcome": "unknown", "body_length": 0, "body_shape": "unknown", "redirect_hop_count": 0, "redirect_location_class": "unknown", "redirect_chain_shape": "unknown", "path": "", "query_key_count": 0, "query_key_shapes": []})
        # Seeing an <html> element without a lang attribute is an observed
        # absence, not an unobserved field.  Keep unknown only when no
        # document root was captured at all.
        if parser.tag_counts.get("html") and parsed.get("document_structure", {}).get("html_lang") == "unknown":
            parsed["document_structure"]["html_lang"] = "absent"
        for axis in ("document_structure", "navigation", "javascript_surface"):
            observation[axis] = parsed[axis]
    if request is not None:
        method = request["method"]
        csrf = str(request_projection.get("csrf_presence_class", "present" if any(str(item.get("role")) == "anti_csrf" for item in request["parameters"]) else "absent"))
        observation["request_transport"] = {"method": method, "placement": "form" if method == "POST" else "query", "content_type_class": "form_urlencoded" if method == "POST" else "none", "encoding_chain": "form_urlencoded" if method == "POST" else "url_percent", "charset_class": "utf8", "body_shape": "form" if method == "POST" and request["parameters"] else "empty", "query_count": len(request["parameters"]) if method == "GET" else 0, "form_count": len(request["parameters"]) if method == "POST" else 0, "json_field_count": 0, "multipart_part_count": 0, "header_presence_class": "basic", "cookie_presence_class": str(request_projection.get("cookie_presence_class", "absent")), "csrf_presence_class": csrf, "content_length": int(request_projection.get("content_length", 0) or 0), "parameters": [{**item, "name_shape": "abstract", "order": index + 1} for index, item in enumerate(request["parameters"])]}
    if response is not None:
        observation["response_transport"] = {key: value for key, value in response.items() if key not in {"path", "query_key_count", "query_key_shapes", "request_content_length", "csrf_presence_class", "failure_class", "failure_stage", "error_shape", "environment_failure_class", "timeout_ms"}}
        observation["failure_feedback"] = {"failure_class": response["failure_class"], "failure_stage": response["failure_stage"], "error_shape": response["error_shape"], "parse_error_class": "none", "encoding_error_class": "none", "redirect_error_class": "none", "blocked_reason_class": "none", "previous_action": "none", "next_action": "observe", "repair_delta_axis": "none", "repair_outcome": "not_applicable", "new_observation": "present", "retry_count": 0, "timeout_bucket": "none", "environment_failure_class": str(response.get("environment_failure_class", "none"))}
    if failure_projection is not None:
        _reject_raw(failure_projection, "failure_projection")
        allowed_failure = {"failure_class", "failure_stage", "error_shape", "parse_error_class", "encoding_error_class", "redirect_error_class", "blocked_reason_class", "previous_action", "next_action", "repair_delta_axis", "repair_outcome", "new_observation", "retry_count", "timeout_bucket", "environment_failure_class"}
        if set(failure_projection) - allowed_failure:
            raise ValueError("failure_projection contains unsupported fields")
        if observation["failure_feedback"] is None:
            observation["failure_feedback"] = {}
        observation["failure_feedback"].update({str(k): v for k, v in failure_projection.items()})
    # Belief/evidence fields are intentionally absent until a distinct typed
    # evaluator sidecar binds candidate/reference/negative/replay evidence.
    observation["belief_and_replay"] = None if belief_projection is None else dict(belief_projection)
    if belief_projection is not None:
        _reject_raw(belief_projection, "belief_projection")
        allowed_belief = {"observation_presence", "observation_delta_axis", "belief_prior_bucket", "belief_posterior_bucket", "belief_delta_axis", "history_action", "history_length", "history_length_bucket", "typed_available", "evidence_present", "negative_control", "fresh_reset", "replay_ready", "reference_present", "candidate_present", "step_budget", "probe_count", "probe_count_bucket", "evidence_hash_present", "failure_class", "failure_stage", "error_shape", "parse_error_class", "encoding_error_class", "redirect_error_class", "timeout_bucket", "blocked_reason_class", "previous_action", "next_action", "repair_delta_axis", "repair_outcome", "method", "placement", "content_type_class", "query_count", "form_count", "json_field_count", "multipart_part_count", "parameter_role", "parameter_name_shape", "parameter_value_type", "parameter_presence", "parameter_order", "header_presence_class", "cookie_presence_class", "csrf_presence_class", "content_length_bucket", "encoding_chain", "charset_class", "body_shape", "status_class", "status_shape", "body_length_bucket", "cache_shape", "redirect_hop_count", "redirect_location_class", "redirect_chain_shape", "connection_outcome"}
        if set(belief_projection) - allowed_belief:
            raise ValueError("belief_projection contains unsupported fields")
    manifest = _field_capture_manifest(observation)
    method = request["method"] if request else "unknown"
    post_unavailable = method == "POST" and not post_supported
    # This adapter has no evaluator input by design.  A response shape is an
    # observation, never proof of an effect; both GET and POST must wait for a
    # separately bound candidate/reference/negative/replay typed sidecar.
    result = {"schema_version": SCHEMA_VERSION, "observation": observation, "field_capture_manifest": manifest, "typed_projection": {"typed_available": False, "next_action": "ask_typed", "safe_to_send": False, "post_supported": bool(post_supported), "post_unavailable": bool(post_unavailable)}, "raw_markup_stored": False, "raw_response_body_stored": False, "raw_payload_stored": False, "raw_url_stored": False}
    if any(fragment in str(result).casefold() for fragment in ("<html", "http://", "https://")):
        raise ValueError("adapter result leaked raw material")
    return result


__all__ = ["SCHEMA_VERSION", "capture_vulnerableapp_projection"]
