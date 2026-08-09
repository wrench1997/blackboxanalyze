import httpx
import pytest

from app.pg185_pikachu_dom_adapter import (
    build_dom_action_manifest,
    build_query,
    inert_dom_probe,
    project_dom_response,
)


def _response(body: str) -> httpx.Response:
    request = httpx.Request("GET", "http://127.0.0.1:3104/vul/xss/xss_reflected_get.php")
    return httpx.Response(200, headers={"content-type": "text/html"}, content=body.encode(), request=request)


def test_pg185_inert_probe_has_no_executable_surface() -> None:
    value = inert_dom_probe("pg185-test-a1")
    assert value == '<span data-sift-marker="pg185-test-a1">pg185-test-a1</span>'
    assert "<script" not in value.casefold()
    assert "onerror" not in value.casefold()
    assert "javascript:" not in value.casefold()
    encoded = inert_dom_probe("pg185-test-a1", encoding="html_entity")
    assert "<span" not in encoded.casefold()
    assert "pg185-test-a1" in encoded


def test_pg185_manifest_binds_observed_get_field_and_hashes() -> None:
    manifest = build_dom_action_manifest(
        path="/vul/xss/xss_reflected_get.php",
        surface="xss_reflected_get",
        field_names=["message", "submit"],
        probe_role="candidate",
        marker="pg185-test-a2",
    )
    assert manifest["probe_kind"] == "inert_dom_markup"
    assert manifest["method"] == "GET"
    assert manifest["placement"] == "query"
    assert manifest["manifest_sha256"]
    assert manifest["payload_sha256"]
    assert "marker" not in manifest
    encoded_manifest = build_dom_action_manifest(
        path="/vul/xss/xss_reflected_get.php",
        surface="xss_reflected_get",
        field_names=["message", "submit"],
        probe_role="candidate",
        marker="pg185-test-a2b",
        encoding_chain=["html_entity", "html_entity"],
    )
    assert encoded_manifest["encoding_depth"] == 2


def test_pg185_typed_dom_surface_is_not_a_vulnerability_positive() -> None:
    marker = "pg185-test-a3"
    body = f"<html><body>{inert_dom_probe(marker)}</body></html>"
    result = project_dom_response(_response(body), marker=marker, baseline_status=200)
    assert result["typed_surface_effect"] is True
    assert result["oracle_projection"]["confirmed_effect"] == "dom_structure"
    assert result["oracle_projection"]["positive"] is False
    assert result["oracle_projection"]["positive_authority"] is False
    assert result["raw_response_retained"] is False


def test_pg185_escaped_markup_does_not_look_like_dom_execution() -> None:
    marker = "pg185-test-a4"
    escaped = "&lt;span data-sift-marker=\"pg185-test-a4\"&gt;pg185-test-a4&lt;/span&gt;"
    result = project_dom_response(_response(escaped), marker=marker, baseline_status=200)
    assert result["typed_surface_effect"] is False
    assert result["oracle_projection"]["positive"] is False
    assert result["oracle_projection"]["safety"]["script_execution"] is False


def test_pg185_query_is_strictly_observed_field_grounded() -> None:
    query, marker = build_query(field_names=["message", "submit"], role="candidate", marker="pg185-test-a5")
    assert query["submit"] == "submit"
    assert "message" in query and marker == "pg185-test-a5"
    with pytest.raises(ValueError):
        build_query(field_names=["invented"], role="candidate", marker="pg185-test-a6")
