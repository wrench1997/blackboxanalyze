from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "pg379" / "impl_b"

EXPECTED_GET = {
    "get_query_html_text",
    "get_path_dom_text",
    "get_fragment_js_navigation",
    "get_json_shape",
    "get_redirect_control",
    "get_failure_feedback",
}
EXPECTED_POST = {
    "post_form_dom_update",
    "post_json_state_transition",
    "post_redirect_control",
    "post_attribute_shape",
    "post_parser_failure",
    "post_replay_shape",
}
EXPECTED_ENCODING = {
    "get_query_html_text": "url_percent",
    "get_path_dom_text": "identity",
    "get_fragment_js_navigation": "fragment",
    "get_json_shape": "json_string",
    "get_redirect_control": "query_parameter",
    "get_failure_feedback": "form_urlencoded",
    "post_form_dom_update": "form_urlencoded",
    "post_json_state_transition": "json_object_then_utf8",
    "post_redirect_control": "form_urlencoded_then_url_percent",
    "post_attribute_shape": "form_urlencoded",
    "post_parser_failure": "json_object_then_utf8",
    "post_replay_shape": "query_parameter_then_url_percent",
}


def _manifest() -> dict[str, object]:
    return json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))


def test_impl_b_manifest_is_independent_and_balanced() -> None:
    manifest = _manifest()
    routes = manifest["route_classes"]
    assert manifest["implementation_id"] == "pg379_dynamic_real_holdout_impl_b"
    assert manifest["runtime_boundary"] == "node_native_http_single_process"
    assert manifest["route_count"] == 12
    assert manifest["get_route_count"] == 6
    assert manifest["post_route_count"] == 6
    assert {row["route_class"] for row in routes if row["method"] == "GET"} == EXPECTED_GET
    assert {row["route_class"] for row in routes if row["method"] == "POST"} == EXPECTED_POST
    assert {row["route_class"]: row["encoding_chain"] for row in routes} == EXPECTED_ENCODING
    assert len({row["response_shape"] for row in routes}) >= 7
    assert manifest["runtime"]["bind_host"] == "127.0.0.1"
    assert manifest["runtime"]["network_mode_required"] == "none"
    assert manifest["runtime"]["persistent_storage"] is False
    assert manifest["runtime"]["external_network"] is False
    assert manifest["attestation_contract"]["source_sha256_required"] is True
    assert manifest["attestation_contract"]["base_image_digest_required"] is True
    assert manifest["attestation_contract"]["runtime_image_digest_required"] is True
    assert manifest["attestation_contract"]["source_rows_emitted"] is False


def test_impl_b_server_has_native_http_boundary_and_no_runtime_dependency() -> None:
    source = (FIXTURE / "server.js").read_text(encoding="utf-8")
    assert 'require("node:http")' in source
    assert 'require("node:net")' not in source
    assert 'require("node:child_process")' not in source
    assert "fetch(" not in source
    assert 'http://127.0.0.1' not in source
    assert 'server.listen(PORT, HOST' in source
    assert 'const HOST = "127.0.0.1"' in source
    assert "MAX_INPUT_LENGTH" in source
    assert "MAX_BODY_BYTES" in source
    assert "escapeHtml" in source
    assert 'parsed.pathname === "/__reset"' in source
    for route_class in EXPECTED_GET | EXPECTED_POST:
        assert source.count(f'route_class: "{route_class}"') == 1
    assert source.count("method: \"GET\"") == 6
    assert source.count("method: \"POST\"") == 6


def test_impl_b_evaluator_projection_and_promotion_are_closed() -> None:
    manifest = _manifest()
    projection = manifest["evaluator_projection"]
    assert projection["canary_seen"] == "boolean"
    assert projection["filter_failure"] == "status_class_and_filter_reason"
    assert projection["response_shape"] == "bounded_enum"
    assert projection["evidence"] == "evaluator_side_only"
    for key in ("training_allowed", "memory_promotion_allowed", "payload_catalog_promotion_allowed", "vulnerability_claim_allowed"):
        assert projection[key] is False
    source = (FIXTURE / "server.js").read_text(encoding="utf-8")
    assert "safe_canary" in source
    assert "filtered_input" in source
    assert "missing_input" in source
    assert "parser_error" in source
    assert "fresh_reset" in source
    assert "target_instance_digest" in source
    assert "randomUUID" in source
    assert "pattern.test instanceof RegExp" in source
    assert "matcher.test(normalized)" in source
    assert "persistent_storage: false" in source
    assert "external_network: false" in source


def test_impl_b_dockerfile_is_reproducible_without_install_or_public_bind() -> None:
    dockerfile = (FIXTURE / "Dockerfile").read_text(encoding="utf-8")
    package = json.loads((FIXTURE / "package.json").read_text(encoding="utf-8"))
    assert "node:20.11.1-alpine3.19" in dockerfile
    assert "ARG NODE_BASE_IMAGE" in dockerfile
    assert "FROM ${NODE_BASE_IMAGE}" in dockerfile
    assert "latest" not in dockerfile.lower()
    assert re.search(r"(?im)^\s*run\s+npm(?:\s|$)", dockerfile) is None
    assert re.search(r"(?im)^\s*run\s+npm\s+ci(?:\s|$)", dockerfile) is None
    assert "USER node" in dockerfile
    assert "EXPOSE 8799" in dockerfile
    assert package.get("dependencies", {}) == {}
    assert package["engines"]["node"] == ">=20.11.0 <21"


def test_impl_b_source_does_not_contain_raw_probe_material() -> None:
    source = (FIXTURE / "server.js").read_text(encoding="utf-8").lower()
    readme = (FIXTURE / "README.md").read_text(encoding="utf-8").lower()
    # The fixture describes abstract filter categories only; no raw wire or
    # attack strings are part of the implementation contract.
    for fragment in ("raw_payload=", "response_body=", "evaluator_answer", "external_callback"):
        assert fragment not in source
    assert re.search(r"https?://(?!127\.0\.0\.1)", readme) is None
