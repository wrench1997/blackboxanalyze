"""PG-379 independent implementation A: dependency-free WSGI fixture.

The fixture is intentionally small and disposable.  It exposes six abstract
GET and six abstract POST route classes, returns only bounded response-shape
projections, and never writes application state or calls an external service.
Input values are classified in memory into a length bucket and a safe
reflection/filtering category; the original probe is never logged, persisted,
or returned.  A collector/evaluator may run this module inside a fresh
network-none container, but importing it or running the static tests does not
open a socket.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import parse_qs


IMPLEMENTATION_ID = "pg379_dynamic_real_train_impl_a"
RUNTIME_ID = "python_wsgiref_stdlib"
RUNTIME_VERSION = "python3-stdlib"
SCHEMA_VERSION = "pg379-independent-implementation-a-v1"
METHODS = ("GET", "POST")
ROLES = ("candidate", "reference", "negative", "replay")

# Route paths are implementation-local literals.  A source-row adapter must
# project them to the route class/shape fields and never place these literals
# in model context.
ROUTE_METADATA: tuple[dict[str, str], ...] = (
    {
        "route_class": "get_query_html_text",
        "method": "GET",
        "path": "/pg379/a/get/query-html-text",
        "parameter_role": "query_text",
        "encoding_chain": "url_percent",
        "response_shape": "html_text",
        "redirect_shape": "none",
        "script_surface": "none",
    },
    {
        "route_class": "get_path_dom_text",
        "method": "GET",
        "path": "/pg379/a/get/path-dom-text",
        "parameter_role": "path_segment",
        "encoding_chain": "identity",
        "response_shape": "html_dom_text",
        "redirect_shape": "none",
        "script_surface": "inline_dom_text",
    },
    {
        "route_class": "get_fragment_js_navigation",
        "method": "GET",
        "path": "/pg379/a/get/fragment-js-navigation",
        "parameter_role": "fragment_identifier",
        "encoding_chain": "fragment",
        "response_shape": "html_fragment",
        "redirect_shape": "none",
        "script_surface": "spa_navigation",
    },
    {
        "route_class": "get_json_shape",
        "method": "GET",
        "path": "/pg379/a/get/json-shape",
        "parameter_role": "json_value",
        "encoding_chain": "json_string",
        "response_shape": "json_shape",
        "redirect_shape": "none",
        "script_surface": "inline_json_data",
    },
    {
        "route_class": "get_redirect_control",
        "method": "GET",
        "path": "/pg379/a/get/redirect-control",
        "parameter_role": "view_mode",
        "encoding_chain": "query_parameter",
        "response_shape": "redirect_shape",
        "redirect_shape": "302_location",
        "script_surface": "history_navigation",
    },
    {
        "route_class": "get_failure_feedback",
        "method": "GET",
        "path": "/pg379/a/get/failure-feedback",
        "parameter_role": "query_term",
        "encoding_chain": "form_urlencoded",
        "response_shape": "error_shape",
        "redirect_shape": "none",
        "script_surface": "none",
    },
    {
        "route_class": "post_form_dom_update",
        "method": "POST",
        "path": "/pg379/a/post/form-dom-update",
        "parameter_role": "form_field",
        "encoding_chain": "form_urlencoded",
        "response_shape": "html_dom_text",
        "redirect_shape": "none",
        "script_surface": "inline_dom_text",
    },
    {
        "route_class": "post_json_state_transition",
        "method": "POST",
        "path": "/pg379/a/post/json-state-transition",
        "parameter_role": "json_value",
        "encoding_chain": "json_object_then_utf8",
        "response_shape": "state_delta",
        "redirect_shape": "none",
        "script_surface": "module_fetch",
    },
    {
        "route_class": "post_redirect_control",
        "method": "POST",
        "path": "/pg379/a/post/redirect-control",
        "parameter_role": "view_mode",
        "encoding_chain": "form_urlencoded_then_url_percent",
        "response_shape": "redirect_shape",
        "redirect_shape": "303_location",
        "script_surface": "history_navigation",
    },
    {
        "route_class": "post_attribute_shape",
        "method": "POST",
        "path": "/pg379/a/post/attribute-shape",
        "parameter_role": "attribute_value",
        "encoding_chain": "form_urlencoded",
        "response_shape": "html_attribute",
        "redirect_shape": "none",
        "script_surface": "none",
    },
    {
        "route_class": "post_parser_failure",
        "method": "POST",
        "path": "/pg379/a/post/parser-failure",
        "parameter_role": "structured_value",
        "encoding_chain": "json_object_then_utf8",
        "response_shape": "error_shape",
        "redirect_shape": "none",
        "script_surface": "dialog_shape",
    },
    {
        "route_class": "post_replay_shape",
        "method": "POST",
        "path": "/pg379/a/post/replay-shape",
        "parameter_role": "record_cursor",
        "encoding_chain": "query_parameter_then_url_percent",
        "response_shape": "replay_shape",
        "redirect_shape": "none",
        "script_surface": "module_fetch",
    },
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def source_digest() -> str:
    """Hash the implementation source for an evaluator-side attestation."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def route_hash(route: Mapping[str, str], *, source_sha256: str | None = None) -> str:
    abstract = {
        "schema_version": SCHEMA_VERSION,
        "implementation_id": IMPLEMENTATION_ID,
        "source_sha256": source_sha256 or source_digest(),
        **{key: str(route[key]) for key in ("route_class", "method", "parameter_role", "encoding_chain", "response_shape", "redirect_shape", "script_surface")},
    }
    return hashlib.sha256(_canonical(abstract)).hexdigest()


def route_by_path(path: str) -> Mapping[str, str] | None:
    for route in ROUTE_METADATA:
        if route["path"] == path:
            return route
        if route["route_class"] == "get_path_dom_text" and path.startswith(route["path"] + "/"):
            return route
    return None


def route_by_class(route_class: str) -> Mapping[str, str] | None:
    for route in ROUTE_METADATA:
        if route["route_class"] == route_class:
            return route
    return None


def _length_bucket(value: str | None) -> str:
    if value is None:
        return "missing"
    length = len(value)
    if length == 0:
        return "empty"
    if length <= 8:
        return "short"
    if length <= 64:
        return "medium"
    return "long"


def _classify(value: str | None) -> dict[str, Any]:
    """Classify a value without returning or storing the value itself."""

    if value is None:
        return {
            "input_class": "missing",
            "filter_reason": "missing_value",
            "presence": "absent",
            "length_bucket": "missing",
            "reflection_status": "not_observed",
            "filter_delta": "not_observed",
        }
    # The fixture's filter boundary is intentionally conservative.  It is a
    # safe canary differential, not an exploit implementation.
    suspicious = any(marker in value.casefold() for marker in ("<", ">", "script", "onerror", "javascript:", "\"", "'"))
    safe_canary = "canary" in value.casefold()
    return {
        "input_class": "filtered" if suspicious else "safe_canary" if safe_canary else "ordinary",
        "filter_reason": "bounded_filter" if suspicious else "none",
        "presence": "present",
        "length_bucket": _length_bucket(value),
        "reflection_status": "filtered_bounded" if suspicious else "reflected_bounded",
        "filter_delta": "changed_shape" if suspicious else "same_shape",
    }


def _first_value(values: Mapping[str, list[str]], preferred: Iterable[str]) -> str | None:
    for key in preferred:
        value = values.get(key)
        if value:
            return str(value[0])
    for value in values.values():
        if value:
            return str(value[0])
    return None


def _request_value(environ: Mapping[str, Any], route: Mapping[str, str]) -> tuple[str | None, str]:
    method = route["method"]
    if method == "GET":
        if route["parameter_role"] == "path_segment":
            path = str(environ.get("PATH_INFO", ""))
            prefix = str(route["path"]) + "/"
            if path.startswith(prefix):
                return path[len(prefix):], "path"
        query = parse_qs(str(environ.get("QUERY_STRING", "")), keep_blank_values=True)
        return _first_value(query, ("q", "value", "mode", "fragment", "cursor", "term")), "query"
    length_text = str(environ.get("CONTENT_LENGTH", "0") or "0")
    try:
        length = max(0, min(int(length_text), 1024 * 1024))
    except ValueError:
        length = 0
    body_stream = environ.get("wsgi.input")
    body = body_stream.read(length) if body_stream is not None and length else b""
    if isinstance(body, bytes):
        text = body.decode("utf-8", errors="replace")
    else:
        text = str(body or "")
    content_type = str(environ.get("CONTENT_TYPE", "")).casefold()
    if "json" in content_type:
        try:
            document = json.loads(text or "{}")
        except (TypeError, ValueError):
            return text[:0], "json_invalid"
        if isinstance(document, Mapping):
            for key in ("value", "q", "mode", "cursor", "record"):
                value = document.get(key)
                if isinstance(value, (str, int, float, bool)):
                    return str(value), "json"
        return None, "json"
    values = parse_qs(text, keep_blank_values=True)
    return _first_value(values, ("q", "value", "mode", "cursor", "record")), "form"


def _response_projection(route: Mapping[str, str], value: str | None, source: str) -> dict[str, Any]:
    classified = _classify(value)
    route_class = route["route_class"]
    status = 200
    if classified["input_class"] == "filtered":
        status = 422
    if route_class in {"get_failure_feedback", "post_parser_failure"} and classified["presence"] == "absent":
        status = 400
    if route["response_shape"] == "redirect_shape":
        status = 302 if route["method"] == "GET" else 303
    return {
        "implementation_id": IMPLEMENTATION_ID,
        "runtime_id": RUNTIME_ID,
        "route_class": route_class,
        "method": route["method"],
        "parameter_role": route["parameter_role"],
        "encoding_chain": route["encoding_chain"],
        "response_shape": route["response_shape"],
        "redirect_shape": route["redirect_shape"],
        "script_surface": route["script_surface"],
        "status_class": "filtered_input" if classified["input_class"] == "filtered" else "missing_input" if classified["input_class"] == "missing" else "accepted_input",
        "http_status_class": f"{status // 100}xx",
        "status_code": status,
        "input_source": source,
        "input": classified,
        "state_transition": "ephemeral_shape_only" if route["response_shape"] == "state_delta" else "none",
        "state_write": False,
        "external_network": False,
        "typed_shape_delta": classified["filter_delta"],
        "raw_input_returned": False,
    }


def fresh_reset() -> dict[str, Any]:
    """Return a stateless reset attestation for evaluator-side orchestration.

    The app keeps no mutable store, so reset is a bounded declaration rather
    than a database operation.  A live collector must still create a fresh
    disposable process/container and attest its identity separately.
    """

    return {
        "fresh_reset": True,
        "state_clean": True,
        "instance_digest": source_digest(),
        "network_mode": "none",
        "external_network": False,
        "loopback_only": True,
        "volume_mount_count": 0,
        "container_restart_used": False,
        "attested": False,
    }


def _json_response(start_response: Callable[..., Any], status: int, document: Mapping[str, Any], *, headers: Mapping[str, str] | None = None) -> list[bytes]:
    body = _canonical(document)
    response_headers = [("Content-Type", "application/json; charset=utf-8"), ("Content-Length", str(len(body)))]
    for key, value in (headers or {}).items():
        response_headers.append((str(key), str(value)))
    start_response(f"{status} {'OK' if status < 400 else 'Bad Request'}", response_headers)
    return [body]


def application(environ: Mapping[str, Any], start_response: Callable[..., Any]) -> list[bytes]:
    """WSGI application; it is safe to call directly in a unit test."""

    path = str(environ.get("PATH_INFO", ""))
    method = str(environ.get("REQUEST_METHOD", "GET")).upper()
    if path == "/__health" and method == "GET":
        return _json_response(
            start_response,
            200,
            {
                "implementation_id": IMPLEMENTATION_ID,
                "runtime_id": RUNTIME_ID,
                "status_class": "2xx",
                "state_clean": True,
                "external_network": False,
                "network_mode": "none",
                "loopback_only": True,
                "persistent_storage": False,
            },
        )
    if path == "/__manifest" and method == "GET":
        # The manifest is evaluator-side metadata; callers should project it
        # before constructing any model context.
        return _json_response(start_response, 200, manifest())
    if path == "/__reset" and method == "POST":
        return _json_response(start_response, 200, fresh_reset())
    route = route_by_path(path)
    if route is None or route["method"] != method:
        return _json_response(
            start_response,
            404,
            {
                "implementation_id": IMPLEMENTATION_ID,
                "response_shape": "not_found_shape",
                "method_observed": method in METHODS,
                "state_write": False,
                "external_network": False,
            },
        )
    value, source = _request_value(environ, route)
    projection = _response_projection(route, value, source)
    if route["redirect_shape"] != "none":
        status = int(projection["status_code"])
        # The redirect stays inside this fixture and is an abstract shape
        # marker; the evaluator may project it without following a network.
        return _json_response(start_response, status, projection, headers={"Location": "/pg379/a/landing"})
    return _json_response(start_response, int(projection["status_code"]), projection)


def manifest(*, source_sha256: str | None = None) -> dict[str, Any]:
    source_sha256 = source_sha256 or source_digest()
    routes = []
    for route in ROUTE_METADATA:
        routes.append(
            {
                "route_class": route["route_class"],
                "method": route["method"],
                "path": route["path"],
                "parameter_role": route["parameter_role"],
                "encoding_chain": route["encoding_chain"],
                "response_shape": route["response_shape"],
                "redirect_shape": route["redirect_shape"],
                "script_surface": route["script_surface"],
                "source_sha256": source_sha256,
                "source_digest": source_sha256,
                "route_hash_sha256": route_hash(route, source_sha256=source_sha256),
                "route_hash": route_hash(route, source_sha256=source_sha256),
                "raw_probe_stored": False,
                "state_write": False,
                "external_network": False,
                "training_eligible": False,
                "promotion": {
                    "training_allowed": False,
                    "memory_promotion_allowed": False,
                    "payload_catalog_promotion_allowed": False,
                    "vulnerability_claim_allowed": False,
                },
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "fixture_only_live_unbound",
        "implementation_id": IMPLEMENTATION_ID,
        "runtime": {
            "runtime_id": RUNTIME_ID,
            "runtime_version": RUNTIME_VERSION,
            "source_sha256": source_sha256,
            "source_digest": source_sha256,
            "runtime_module_sha256": source_sha256,
            "process_boundary_attested": False,
        },
        "source_attestation": {
            "source_digest": source_sha256,
            "runtime_module_sha256": source_sha256,
            "process_boundary_attested": False,
            "bound": False,
        },
        "image_attestation": {"bound": False, "image_digest": None, "operator_authorization": None},
        "network_contract": {"network_mode": "none", "loopback_only": True, "external_network": False, "bind_or_volume_mounts": False},
        "fresh_reset_contract": {"before_each_role": True, "after_each_role": True, "state_clean": True, "teardown_after_each_episode": True},
        "roles": list(ROLES),
        "typed_role_contract": {
            "candidate_reference_negative_replay": True,
            "role_bound_evidence_sha256_required": True,
            "negative_violation_max": 0,
            "replay_sidecar_only": True,
        },
        "failure_repair_belief_contract": {
            "failure_action_change_required": True,
            "repair_action_required": True,
            "belief_prior_posterior_delta_required": True,
            "replay_state_required": True,
        },
        "sidecar_context_firewall": {
            "typed_sidecar_evaluator_only": True,
            "evidence_sha256_evaluator_only": True,
            "oracle_answer_in_context": False,
            "raw_payload_response_wire_in_context": False,
        },
        "route_count": len(routes),
        "get_count": sum(route["method"] == "GET" for route in routes),
        "post_count": sum(route["method"] == "POST" for route in routes),
        "routes": routes,
        "input_classes": ["safe_canary", "ordinary", "filtered", "missing", "parser_error"],
        "filter_categories": ["bounded_filter", "missing_value"],
        "raw_probe_evaluator_only": True,
        "rows_emitted": False,
        "training_eligible": False,
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
    }


def validate_manifest(document: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if document.get("schema_version") != SCHEMA_VERSION:
        failures.append("schema_version")
    if document.get("implementation_id") != IMPLEMENTATION_ID:
        failures.append("implementation_id")
    if list(document.get("roles") or []) != list(ROLES):
        failures.append("roles")
    routes = list(document.get("routes") or [])
    if len(routes) != 12:
        failures.append("route_count")
    if int(document.get("get_count", -1)) != 6 or int(document.get("post_count", -1)) != 6:
        failures.append("get_post_balance")
    runtime = document.get("runtime") if isinstance(document.get("runtime"), Mapping) else {}
    source_sha256 = str(runtime.get("source_sha256", ""))
    if len(source_sha256) != 64:
        failures.append("source_sha256")
    if runtime.get("source_digest") != source_sha256:
        failures.append("source_digest")
    if runtime.get("runtime_module_sha256") != source_sha256 or runtime.get("process_boundary_attested") is not False:
        failures.append("runtime_attestation")
    source_attestation = document.get("source_attestation") if isinstance(document.get("source_attestation"), Mapping) else {}
    if source_attestation.get("source_digest") != source_sha256 or source_attestation.get("runtime_module_sha256") != source_sha256 or source_attestation.get("process_boundary_attested") is not False or source_attestation.get("bound") is not False:
        failures.append("source_attestation")
    for route in routes:
        if route.get("source_sha256") != source_sha256 or route.get("source_digest") != source_sha256 or len(str(route.get("route_hash_sha256", ""))) != 64 or route.get("route_hash") != route.get("route_hash_sha256"):
            failures.append("route_attestation")
        if route.get("raw_probe_stored") is not False or route.get("state_write") is not False or route.get("external_network") is not False or route.get("training_eligible") is not False:
            failures.append("route_safety")
    if document.get("rows_emitted") is not False or document.get("training_eligible") is not False:
        failures.append("rows_or_training")
    image = document.get("image_attestation") if isinstance(document.get("image_attestation"), Mapping) else {}
    if image.get("bound") is not False or image.get("image_digest") is not None:
        failures.append("image_unbound")
    return {"status": "passed" if not failures else "blocked", "failures": sorted(set(failures)), "route_count": len(routes)}


# Conventional WSGI names for an evaluator-side relay/importer.  Aliases do
# not start a server or alter the no-side-effect contract.
wsgi_app = application
app = application


def serve() -> None:  # pragma: no cover - explicit operator action only
    from wsgiref.simple_server import make_server

    bind = os.environ.get("PG379_IMPL_A_BIND", "127.0.0.1")
    port = int(os.environ.get("PG379_IMPL_A_PORT", "8080"))
    with make_server(bind, port, application) as server:
        server.serve_forever()


if __name__ == "__main__":  # pragma: no cover - never used by static tests
    serve()


__all__ = [
    "IMPLEMENTATION_ID",
    "METHODS",
    "ROLES",
    "ROUTE_METADATA",
    "RUNTIME_ID",
    "SCHEMA_VERSION",
    "application",
    "app",
    "manifest",
    "route_by_class",
    "route_by_path",
    "route_hash",
    "source_digest",
    "fresh_reset",
    "validate_manifest",
    "wsgi_app",
]
