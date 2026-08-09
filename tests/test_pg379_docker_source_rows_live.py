from __future__ import annotations

import pytest

from scripts.run_pg379_docker_source_rows_live import DockerRuntime, SAFE_CANARY, run_live


def _route(*, method: str = "GET", input_source: str = "query", parameter: str = "q") -> dict[str, object]:
    return {
        "route_class": "get_query_html_text",
        "path": "/fixture/query",
        "method": method,
        "parameter": parameter,
        "parameter_role": "query_text",
        "encoding_chain": "url_percent",
        "response_shape": "html_text",
        "script_surface": "none",
        "input_source": input_source,
    }


def test_wire_uses_manifest_path_and_bounded_canary_for_python_fixture() -> None:
    route = _route()
    runtime = DockerRuntime(
        implementation_id="pg379_impl_a",
        lane="train",
        seed=37901,
        route=route,
        role="candidate",
        image="pg379-impl-a:reviewed@sha256:" + "a" * 64,
        manifest={"routes": [route]},
        name="pg379-test-a",
    )

    path, body = runtime._wire("GET", route)

    assert path.startswith("/fixture/query?q=")
    assert SAFE_CANARY in path
    assert body == b""


def test_wire_replaces_dynamic_path_value_for_node_fixture() -> None:
    route = _route(input_source="path", parameter="value")
    route["path"] = "/fixture/path/<value>"
    runtime = DockerRuntime(
        implementation_id="pg379_impl_b",
        lane="holdout",
        seed=37901,
        route=route,
        role="reference",
        image="pg379-impl-b:reviewed@sha256:" + "b" * 64,
        manifest={"routes": [route]},
        name="pg379-test-b",
    )

    path, body = runtime._wire("GET", route)

    assert path == "/fixture/path/PG379B_CANARY_safe"
    assert body == b""


def test_live_requires_explicit_operator_review_before_image_inspection() -> None:
    with pytest.raises(ValueError, match="operator-reviewed"):
        run_live(
            output=None,  # type: ignore[arg-type]
            sidecar_output=None,  # type: ignore[arg-type]
            rows_output=None,  # type: ignore[arg-type]
            authorization_id="test",
            image_a="missing-a",
            image_b="missing-b",
            operator_reviewed=False,
        )
