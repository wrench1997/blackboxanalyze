from app.surface_sink_oracle import observe_surface_sink


def test_sink_oracle_binds_html_attribute_without_execution():
    result = observe_surface_sink(
        '<div data-sift-marker="sink-test-marker">ok</div>',
        marker="sink-test-marker",
        content_type="text/html",
    )
    assert result["sink_kind"] == "html_attribute"
    assert result["marker_in_attribute"] is True
    assert result["script_execution"] is False
    assert result["network_access"] is False
    assert result["body_sha256"]


def test_sink_oracle_distinguishes_text_json_header_and_none():
    text = observe_surface_sink(
        "<p>sink-test-marker</p>", marker="sink-test-marker", content_type="text/html"
    )
    json_value = observe_surface_sink(
        '{"message":"sink-test-marker"}', marker="sink-test-marker", content_type="application/json"
    )
    header = observe_surface_sink(
        "<p>no body marker</p>", marker="sink-test-marker", content_type="text/html", headers={"x-sift": "sink-test-marker"}
    )
    clean = observe_surface_sink("<p>clean</p>", marker="sink-test-marker", content_type="text/html")
    assert text["sink_kind"] == "html_text"
    assert json_value["sink_kind"] == "json_value"
    assert header["sink_kind"] == "response_header"
    assert clean["sink_kind"] == "none"


def test_sink_oracle_rejects_non_inert_marker():
    try:
        observe_surface_sink("x", marker="<script>", content_type="text/html")
    except ValueError:
        return
    raise AssertionError("unsafe marker was not rejected")
