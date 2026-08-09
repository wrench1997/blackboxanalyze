from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
from threading import Thread

import pytest

from app.pg331_loopback_adapter import capture_loopback
from app.pg331_web_tokenizer import tokenize_web_observation
from scripts.capture_pg331_loopback_source_row import capture_source_row


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        body = b"""<!doctype html><html lang='zh'><head><title>Demo</title><script>fetch('/api'); document.body.innerHTML = location.search;</script></head><body data-parameter-role='query_text'><a href='/next?x=1'>next</a><form method='post' action='/submit'><input name='search' type='text'></form></body></html>"""
        self.send_response(200)
        if not self.path.startswith("/no-content-type"):
            self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length:
            self.rfile.read(length)
        self.send_response(302)
        self.send_header("Location", "/done")
        self.end_headers()

    def log_message(self, *_args: object) -> None:
        return


@pytest.fixture()
def local_origin():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_get_capture_returns_all_seven_abstract_axes(local_origin: str) -> None:
    result = capture_loopback(f"{local_origin}/?q=1", method="GET")
    observation = result["observation"]
    assert result["raw_body_stored"] is False
    assert result["raw_payload_stored"] is False
    assert set(result["field_capture_manifest"]) == {"document_structure", "navigation", "request_transport", "response_transport", "javascript_surface", "failure_feedback", "belief_and_replay"}
    assert result["transport"]["method"] == "GET"
    assert result["transport"]["status_class"] == "2xx"
    assert result["transport"]["failure_class"] == "none"
    assert set(observation) == {"document_structure", "navigation", "request_transport", "response_transport", "javascript_surface", "failure_feedback", "belief_and_replay"}
    assert observation["navigation"]["links"]
    assert "path_segment_shape" in observation["navigation"]
    assert observation["navigation"]["navigation_event"] == "initial_load"
    assert observation["request_transport"]["parameters"]
    assert any(parameter["role"] == "query_term" for parameter in observation["request_transport"]["parameters"])
    assert observation["javascript_surface"]["fetch_count"] == 1
    assert observation["javascript_surface"]["xhr_method"] == "ABSENT"
    assert observation["response_transport"]["status_shape"] == "numeric"
    assert observation["response_transport"]["content_type_class"] == "html"
    assert observation["response_transport"]["cache_shape"] == "absent"
    assert "section_order" in observation["document_structure"]
    assert any(element["text_shape"] != "empty" for element in observation["document_structure"]["elements"])
    assert "Demo" not in str(result)


def test_get_capture_preserves_all_page_and_transport_fields(local_origin: str) -> None:
    result = capture_loopback(f"{local_origin}/demo?q=1", method="GET")
    manifest = result["field_capture_manifest"]
    expected_unknown = {
        "request_transport": set(),
        "response_transport": set(),
        "belief_and_replay": {"belief_prior_bucket", "belief_posterior_bucket", "negative_control", "fresh_reset", "step_budget"},
    }
    for axis in ("document_structure", "navigation", "request_transport", "response_transport", "javascript_surface", "failure_feedback", "belief_and_replay"):
        actual_unknown = {field for field, status in manifest[axis].items() if status in {"unknown", "not_observed"}}
        assert actual_unknown == expected_unknown.get(axis, set()), axis
    # Belief/evaluator fields are intentionally unknown until a replay/evaluator
    # sidecar exists; they must remain visible rather than being guessed.
    assert any(status == "unknown" for status in manifest["belief_and_replay"].values())


def test_html_shape_is_parsed_when_content_type_header_is_missing(local_origin: str) -> None:
    result = capture_loopback(f"{local_origin}/no-content-type?q=1", method="GET")
    document = result["observation"]["document_structure"]
    assert document["doctype"] == "html"
    assert document["html_lang"] == "zh"
    assert result["observation"]["response_transport"]["content_type_class"] == "html"
    assert result["observation"]["response_transport"]["cache_shape"] == "absent"
    assert result["observation"]["response_transport"]["charset_class"] == "absent"
    assert result["observation"]["request_transport"]["csrf_presence_class"] == "absent"
    tokens = set(tokenize_web_observation(result["observation"])["context_tokens"])
    assert "document_structure_field_doctype=html" in tokens
    assert "document_structure_field_html_lang=zh" in tokens
    assert result["field_capture_manifest"]["document_structure"]["doctype"] == "observed"
    assert result["field_capture_manifest"]["document_structure"]["html_lang"] == "observed"
    assert result["raw_body_stored"] is False


def test_completed_response_without_content_type_is_observed_absence(local_origin: str) -> None:
    result = capture_loopback(f"{local_origin}/submit", method="POST", form_data={"email": "", "password": ""})
    response = result["observation"]["response_transport"]
    assert response["content_type_class"] == "absent"
    assert response["charset_class"] == "absent"
    assert result["observation"]["request_transport"]["csrf_presence_class"] == "absent"


def test_post_capture_keeps_redirect_shape_without_following_it(local_origin: str) -> None:
    result = capture_loopback(f"{local_origin}/submit", method="POST", form_data={"search": ""})
    response = result["observation"]["response_transport"]
    assert result["transport"]["method"] == "POST"
    assert response["status_class"] == "3xx"
    assert response["redirect_hop_count"] == 1
    assert response["redirect_location_class"] == "present"
    assert response["redirect_chain_shape"] == "single_hop"


def test_parameter_roles_preserve_get_post_semantics_without_raw_names(local_origin: str) -> None:
    get_result = capture_loopback(f"{local_origin}/?id=1&submit=", method="GET")
    get_parameters = get_result["observation"]["request_transport"]["parameters"]
    get_roles = {item["role"] for item in get_parameters}
    assert {"identifier", "submit_control"} <= get_roles
    assert all("name" not in item for item in get_parameters)
    post_result = capture_loopback(f"{local_origin}/submit", method="POST", form_data={"id": "", "submit": ""})
    post_parameters = post_result["observation"]["request_transport"]["parameters"]
    post_roles = {item["role"] for item in post_parameters}
    assert {"identifier", "submit_control"} <= post_roles
    assert all("name" not in item for item in post_parameters)


def test_visible_surface_role_hint_is_kept_as_abstract_observation(local_origin: str) -> None:
    result = capture_loopback(f"{local_origin}/?surface=1", method="GET")
    parameters = result["observation"]["request_transport"]["parameters"]
    # The fixture exposes the role through the visible form structure.  The
    # adapter keeps the semantic role but never the input name/value.
    assert any(item["role"] == "query_text" for item in parameters)
    hint = next(item for item in parameters if item["role"] == "query_text")
    assert hint["presence"] == "surface_observed"
    assert hint["value_type"] == "surface_hint"
    assert all("name" not in item and "value" not in item for item in parameters)


def test_adapter_rejects_non_loopback_origins() -> None:
    with pytest.raises(ValueError):
        capture_loopback("https://example.com:443/")


def test_evaluator_callback_sees_body_only_in_memory_and_returns_abstract_projection(local_origin: str) -> None:
    seen: list[bytes] = []

    def evaluator(body: bytes, headers: object, status: int | None) -> dict[str, object]:
        seen.append(body)
        return {"status_class": "2xx" if status == 200 else "unknown", "body_shape": "html" if body else "empty"}

    result = capture_loopback(f"{local_origin}/?q=1", method="GET", evaluator=evaluator)
    assert seen and b"Demo" in seen[0]
    assert result["evaluator_projection"] == {"status_class": "2xx", "body_shape": "html"}
    assert "body" not in result["evaluator_projection"]
    assert result["raw_body_stored"] is False


def test_evaluator_callback_raw_projection_is_rejected(local_origin: str) -> None:
    def evaluator(_body: bytes, _headers: object, _status: int | None) -> dict[str, object]:
        return {"response_body": "forbidden"}

    with pytest.raises(ValueError, match="raw material"):
        capture_loopback(f"{local_origin}/?q=1", method="GET", evaluator=evaluator)


def test_source_row_bridge_keeps_baseline_incomplete_and_emits_ask(local_origin: str) -> None:
    digest = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    row = capture_source_row(
        origin=f"{local_origin}/?q=1",
        method="GET",
        post_fields=[],
        record_id="loopback:demo:get",
        source_meta={
            "source_id": "local-demo",
            "implementation": "fixture",
            "collector_id": "test",
            "authorization_id": "local-test",
            "image_digest": digest("image"),
            "source_digest": digest("source"),
        },
        reset={
            "fresh_reset": True,
            "reset_id": "reset-1",
            "target_instance_digest": digest("instance"),
            "network_mode": "loopback",
            "external_network": False,
            "loopback_only": True,
            "state_clean": True,
        },
        evaluator={
            "typed_available": False,
            "negative_control": False,
            "reference_present": False,
            "candidate_present": False,
            "fresh_reset": True,
            "evidence_hash": digest("evidence"),
            "evaluator_version": "fixture-v1",
        },
        target_projection={
            "question": "none",
            "next_action": "assemble_rule_ir",
            "repair_action": "none",
            "transport_ref": "request_method",
            "field_role_ref": "surface_field_role",
            "encoding_ref": "encoding_chain",
            "probe_variant_ref": "none",
            "safe_to_send": False,
        },
    )
    assert row["training_eligible"] is False
    assert row["target_projection"]["question"] == "ask_typed"
    assert row["target_projection"]["next_action"] == "ask_typed"
    assert row["raw_payload_stored"] is False
    assert row["raw_response_body_stored"] is False
    assert local_origin not in str(row)
    assert any(item.startswith("field_") for item in row["failures"])


def test_source_row_bridge_post_is_neutral_and_preserves_redirect(local_origin: str) -> None:
    digest = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    row = capture_source_row(
        origin=f"{local_origin}/submit",
        method="POST",
        post_fields=["search"],
        record_id="loopback:demo:post",
        source_meta={
            "source_id": "local-demo",
            "implementation": "fixture",
            "collector_id": "test",
            "authorization_id": "local-test",
            "image_digest": digest("image"),
            "source_digest": digest("source"),
        },
        reset={
            "fresh_reset": True,
            "reset_id": "reset-2",
            "target_instance_digest": digest("instance-2"),
            "network_mode": "loopback",
            "external_network": False,
            "loopback_only": True,
            "state_clean": True,
        },
        evaluator={"evidence_hash": digest("evidence-2"), "evaluator_version": "fixture-v1"},
        target_projection={"next_action": "abstain", "safe_to_send": False},
    )
    response = row["context_tokens"]
    assert "response_status_class=3xx" in response
    assert row["training_eligible"] is False
    assert "search" not in str(row)
