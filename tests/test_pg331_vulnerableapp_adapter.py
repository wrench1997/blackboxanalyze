from __future__ import annotations

import pytest

from app.pg331_vulnerableapp_adapter import capture_vulnerableapp_projection


HTML = "<!doctype html><html><head><script>fetch('/api'); document.body.innerHTML=location.search</script></head><body><form method='get'><input name='q'></form></body></html>"


def test_get_html_projection_has_abstract_axes_and_no_raw_result() -> None:
    result = capture_vulnerableapp_projection(html=HTML, headers={"Content-Type": "text/html"}, request_projection={"method": "GET", "parameters": [{"role": "query_term"}]}, response_projection={"status": 200, "body_length": len(HTML), "body_shape": "html"})
    assert set(result["observation"]) == {"document_structure", "navigation", "request_transport", "response_transport", "javascript_surface", "failure_feedback", "belief_and_replay"}
    assert result["observation"]["javascript_surface"]["fetch_count"] == 1
    assert result["field_capture_manifest"]["document_structure"]["doctype"] == "observed"
    assert result["typed_projection"]["safe_to_send"] is False
    assert result["typed_projection"]["typed_available"] is False
    assert result["typed_projection"]["next_action"] == "ask_typed"
    assert HTML not in str(result) and "127.0.0.1" not in str(result)


def test_html_root_without_doctype_marks_doctype_absent() -> None:
    result = capture_vulnerableapp_projection(
        html="<html><body>bounded</body></html>",
        headers={"Content-Type": "text/html"},
        request_projection={"method": "GET"},
        response_projection={"status": 200, "body_length": 31, "body_shape": "html"},
    )
    assert result["observation"]["document_structure"]["doctype"] == "absent"


def test_html_fragment_marks_root_fields_absent() -> None:
    result = capture_vulnerableapp_projection(
        html="<div>bounded fragment</div>",
        headers={"Content-Type": "text/html"},
        request_projection={"method": "GET"},
        response_projection={"status": 200, "body_length": 29, "body_shape": "html"},
    )
    document = result["observation"]["document_structure"]
    assert document["doctype"] == "absent"
    assert document["html_lang"] == "absent"


def test_302_and_post_unsupported_stay_abstract_and_ask_safe() -> None:
    result = capture_vulnerableapp_projection(html="<html></html>", headers={"Location": "/done"}, request_projection={"method": "POST", "parameters": []}, response_projection={"status": 302, "body_length": 0}, post_supported=False)
    response = result["observation"]["response_transport"]
    assert response["status_class"] == "3xx" and response["redirect_location_class"] == "present"
    assert result["typed_projection"] == {"typed_available": False, "next_action": "ask_typed", "safe_to_send": False, "post_supported": False, "post_unavailable": True}


def test_missing_observations_are_explicit_not_observed() -> None:
    result = capture_vulnerableapp_projection(html=None, headers=None, request_projection=None, response_projection=None)
    assert all(value is None for value in result["observation"].values())
    assert all(status == "not_observed" for axis in result["field_capture_manifest"].values() for status in axis.values())
    assert result["typed_projection"]["next_action"] == "ask_typed"


def test_json_response_marks_document_root_absent_not_unobserved() -> None:
    result = capture_vulnerableapp_projection(
        html='{"response_shape":"json"}',
        headers={"Content-Type": "application/json"},
        request_projection={"method": "GET"},
        response_projection={"status": 200, "body_length": 28, "body_shape": "json"},
        post_supported=True,
    )
    document = result["observation"]["document_structure"]
    assert document["doctype"] == "absent"
    assert document["html_lang"] == "absent"


@pytest.mark.parametrize("bad", [{"response_body": "x"}, {"url": "http://example.test"}, {"payload": "x"}])
def test_raw_input_side_channels_are_rejected(bad: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="raw/literal"):
        capture_vulnerableapp_projection(html="<html></html>", headers=bad, request_projection={"method": "GET"}, response_projection={"status": 200})
