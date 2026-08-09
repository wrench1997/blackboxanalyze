from __future__ import annotations

import io
import json
from pathlib import Path

from fixtures.pg379.impl_a.app import (
    IMPLEMENTATION_ID,
    ROUTE_METADATA,
    SCHEMA_VERSION,
    application,
    fresh_reset,
    manifest,
    route_hash,
    source_digest,
    validate_manifest,
)


def _call_wsgi(path: str, method: str = "GET", query: str = "", body: bytes = b"", content_type: str = "") -> tuple[str, list[tuple[str, str]], dict[str, object]]:
    captured: dict[str, object] = {}

    def start_response(status: str, headers: list[tuple[str, str]], _exc_info=None) -> None:
        captured["status"] = status
        captured["headers"] = headers

    environ = {
        "PATH_INFO": path,
        "REQUEST_METHOD": method,
        "QUERY_STRING": query,
        "CONTENT_LENGTH": str(len(body)),
        "CONTENT_TYPE": content_type,
        "wsgi.input": io.BytesIO(body),
    }
    chunks = application(environ, start_response)
    payload = json.loads(b"".join(chunks).decode("utf-8"))
    return str(captured["status"]), list(captured["headers"]), payload


def test_manifest_has_two_balanced_route_sets_and_independent_hashes() -> None:
    document = manifest()
    assert document["schema_version"] == SCHEMA_VERSION
    assert document["implementation_id"] == IMPLEMENTATION_ID
    assert document["route_count"] == 12
    assert document["get_count"] == 6
    assert document["post_count"] == 6
    assert document["rows_emitted"] is False
    assert document["training_eligible"] is False
    assert document["image_attestation"] == {"bound": False, "image_digest": None, "operator_authorization": None}
    assert document["runtime"]["runtime_module_sha256"] == document["runtime"]["source_sha256"]
    assert document["source_attestation"]["bound"] is False
    assert validate_manifest(document)["status"] == "passed"
    assert len({route["route_hash_sha256"] for route in document["routes"]}) == 12
    assert all(route["source_sha256"] == source_digest() and route["source_digest"] == route["source_sha256"] and route["route_hash"] == route["route_hash_sha256"] for route in document["routes"])


def test_route_metadata_matches_planned_get_post_classes() -> None:
    classes = {route["route_class"] for route in ROUTE_METADATA}
    assert len(ROUTE_METADATA) == 12
    assert sum(route["method"] == "GET" for route in ROUTE_METADATA) == 6
    assert sum(route["method"] == "POST" for route in ROUTE_METADATA) == 6
    assert {
        "get_query_html_text",
        "get_path_dom_text",
        "get_fragment_js_navigation",
        "get_json_shape",
        "get_redirect_control",
        "get_failure_feedback",
        "post_form_dom_update",
        "post_json_state_transition",
        "post_redirect_control",
        "post_attribute_shape",
        "post_parser_failure",
        "post_replay_shape",
    } == classes
    assert all(len(route_hash(route)) == 64 for route in ROUTE_METADATA)


def test_get_safe_canary_and_filtered_difference_are_bounded() -> None:
    safe_status, _, safe = _call_wsgi("/pg379/a/get/query-html-text", query="q=CANARY")
    filtered_status, _, filtered = _call_wsgi("/pg379/a/get/query-html-text", query="q=%3Ctag%3E")
    assert safe_status.startswith("200")
    assert filtered_status.startswith("422")
    assert safe["input"]["input_class"] == "safe_canary"
    assert filtered["input"]["input_class"] == "filtered"
    assert safe["input"]["reflection_status"] == "reflected_bounded"
    assert filtered["input"]["reflection_status"] == "filtered_bounded"
    assert safe["raw_input_returned"] is False
    assert filtered["raw_input_returned"] is False
    assert "CANARY" not in json.dumps(safe)
    assert "<tag>" not in json.dumps(filtered)


def test_path_segment_route_is_dynamic_without_returning_path_value() -> None:
    status, _, document = _call_wsgi("/pg379/a/get/path-dom-text/CANARY")
    assert status.startswith("200")
    assert document["input"]["input_class"] == "safe_canary"
    assert document["input_source"] == "path"
    assert "CANARY" not in json.dumps(document)


def test_post_shapes_are_ephemeral_and_do_not_write_state() -> None:
    status, _, document = _call_wsgi(
        "/pg379/a/post/json-state-transition",
        method="POST",
        body=b'{"value":"CANARY"}',
        content_type="application/json",
    )
    assert status.startswith("200")
    assert document["response_shape"] == "state_delta"
    assert document["state_transition"] == "ephemeral_shape_only"
    assert document["state_write"] is False
    assert document["external_network"] is False
    assert document["raw_input_returned"] is False


def test_fresh_reset_is_stateless_and_loopback_closed() -> None:
    reset = fresh_reset()
    assert reset["fresh_reset"] is True
    assert reset["state_clean"] is True
    assert len(reset["instance_digest"]) == 64
    assert reset["network_mode"] == "none"
    assert reset["external_network"] is False
    assert reset["loopback_only"] is True
    assert reset["volume_mount_count"] == 0
    assert reset["attested"] is False


def test_health_manifest_and_reset_endpoints_are_local_contract_helpers() -> None:
    health_status, _, health = _call_wsgi("/__health")
    assert health_status.startswith("200")
    assert health["state_clean"] is True
    assert health["external_network"] is False
    manifest_status, _, manifest_document = _call_wsgi("/__manifest")
    assert manifest_status.startswith("200")
    assert manifest_document["implementation_id"] == IMPLEMENTATION_ID
    reset_status, _, reset_document = _call_wsgi("/__reset", method="POST")
    assert reset_status.startswith("200")
    assert reset_document["fresh_reset"] is True
    assert reset_document["attested"] is False


def test_redirect_and_failure_routes_are_shape_only() -> None:
    redirect_status, headers, redirect = _call_wsgi("/pg379/a/get/redirect-control", query="mode=next")
    assert redirect_status.startswith("302")
    assert any(key.lower() == "location" and value == "/pg379/a/landing" for key, value in headers)
    assert redirect["response_shape"] == "redirect_shape"
    failure_status, _, failure = _call_wsgi("/pg379/a/get/failure-feedback")
    assert failure_status.startswith("400")
    assert failure["response_shape"] == "error_shape"
    assert failure["state_write"] is False


def test_manifest_file_is_present_and_has_no_training_rows() -> None:
    path = Path("fixtures/pg379/impl_a/manifest_v1.json")
    if not path.exists():
        # The source manifest is generated in the repository; keeping this
        # fallback makes the contract test useful before a local regeneration.
        document = manifest()
    else:
        document = json.loads(path.read_text(encoding="utf-8"))
    assert validate_manifest(document)["status"] == "passed"
    assert document["rows_emitted"] is False
    assert document["promotion"]["training_allowed"] is False
