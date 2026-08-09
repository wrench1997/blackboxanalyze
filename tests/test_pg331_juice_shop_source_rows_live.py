from __future__ import annotations

import json

import pytest

import scripts.run_pg331_juice_shop_source_rows_live as live
from scripts.run_pg331_juice_shop_source_rows_live import (
    IMAGE,
    ROUTES,
    _oracle_projection,
    _route_request,
    _strict_reset,
)


def test_live_collector_routes_are_bounded_and_neutral() -> None:
    assert len(ROUTES) == 3
    assert {str(route["method"]) for route in ROUTES} == {"GET", "POST"}
    assert all(str(route["path"]).startswith("/") for route in ROUTES)
    for route in ROUTES:
        path, form = _route_request(route)
        assert path.startswith("/")
        assert "PG331-NEUTRAL" in path or form is not None
        if str(route["method"]) == "POST":
            assert form == {"email": "", "password": ""}


def test_reset_alias_is_reduced_to_strict_source_row_shape() -> None:
    reset = _strict_reset(
        {
            "reset_id": "pg324-reset-1",
            "fresh_target": True,
            "container_id_sha256": "a" * 64,
            "network_mode": "none",
            "external_network": False,
            "relay_loopback_only": True,
            "domain_data_write_allowed": False,
            "database_health_gate": "juice_shop_http_health_ok",
            "bind_or_volume_mount_count": 0,
        }
    )
    assert reset == {
        "reset_id": "pg324-reset-1",
        "fresh_reset": True,
        "target_instance_digest": "a" * 64,
        "network_mode": "none",
        "external_network": False,
        "loopback_only": True,
        "state_clean": True,
        "database_health_gate": "juice_shop_http_health_ok",
    }


def test_oracle_projection_is_allowlisted_and_drops_route_oracle_literals() -> None:
    projection = _oracle_projection(
        {
            "challenge_state_delta": True,
            "challenge_solved": True,
            "sink_present": True,
            "route_id": "forbidden-route-literal",
            "payload": "forbidden-payload",
        }
    )
    assert projection == {"challenge_state_delta": True, "challenge_solved": True, "sink_present": True}
    assert "route_id" not in projection
    assert "payload" not in projection


def test_live_collector_is_pinned_to_juice_shop_image() -> None:
    assert IMAGE.startswith("bkimminich/juice-shop@sha256:")
    assert len(IMAGE.rsplit("@sha256:", 1)[-1]) == 64


class _FakeRequest:
    def __init__(self, url: str) -> None:
        self.url = url


class _FakeRoute:
    def __init__(self, url: str) -> None:
        self.request = _FakeRequest(url)
        self.actions: list[str] = []

    def continue_(self) -> None:
        self.actions.append("continue")

    def abort(self) -> None:
        self.actions.append("abort")


class _FakePage:
    def __init__(self, markup: str, request_urls: list[str], *, fail_content: bool = False) -> None:
        self.markup = markup
        self.request_urls = request_urls
        self.fail_content = fail_content
        self.handler = None
        self.routes: list[_FakeRoute] = []
        self.goto_args: tuple[tuple[object, ...], dict[str, object]] | None = None
        self.wait_ms = 0
        self.closed = False

    def route(self, pattern: str, handler: object) -> None:
        assert pattern == "**/*"
        self.handler = handler

    def goto(self, *args: object, **kwargs: object) -> None:
        self.goto_args = (args, kwargs)
        assert callable(self.handler)
        for url in self.request_urls:
            route = _FakeRoute(url)
            self.routes.append(route)
            self.handler(route)

    def wait_for_timeout(self, milliseconds: int) -> None:
        self.wait_ms = milliseconds

    def content(self) -> str:
        if self.fail_content:
            raise RuntimeError("content unavailable")
        return self.markup

    def close(self) -> None:
        self.closed = True


class _FakeContext:
    def __init__(self, page: _FakePage) -> None:
        self.page = page
        self.closed = False

    def new_page(self) -> _FakePage:
        return self.page

    def close(self) -> None:
        self.closed = True


class _FakeBrowser:
    def __init__(self, page: _FakePage) -> None:
        self.page = page
        self.context: _FakeContext | None = None
        self.context_kwargs: dict[str, object] | None = None

    def new_context(self, **kwargs: object) -> _FakeContext:
        self.context_kwargs = kwargs
        self.context = _FakeContext(self.page)
        return self.context


def test_browser_page_capture_projects_rendered_spa_and_aborts_external_requests() -> None:
    markup = """<!doctype html><html lang='en'><head><title>Track</title></head>
    <body><main><a href='/rest/products'>Products</a><script>fetch('/api')</script></main></body></html>"""
    page = _FakePage(
        markup,
        ["http://127.0.0.1:4312/#/", "https://cdn.example.invalid/app.js"],
    )
    browser = _FakeBrowser(page)

    result = live._capture_browser_page_projection(browser, "http://127.0.0.1:4312")

    assert result["ok"] is True
    assert result["raw_html_stored"] is False
    assert result["blocked_external_count"] == 1
    assert page.goto_args is not None
    assert str(page.goto_args[0][0]).endswith("/#/")
    assert [route.actions for route in page.routes] == [["continue"], ["abort"]]
    assert browser.context_kwargs == {"java_script_enabled": True, "service_workers": "block"}
    observation = result["observation"]
    assert observation["document_structure"]["doctype"] == "html"
    assert observation["document_structure"]["html_lang"] == "en"
    assert observation["navigation"]["link_count"] == 1
    assert observation["javascript_surface"]["fetch_count"] == 1
    assert markup not in json.dumps(result, ensure_ascii=False)
    assert page.closed is True
    assert browser.context is not None and browser.context.closed is True


def test_browser_page_capture_failure_is_explicit_environment_ask() -> None:
    page = _FakePage("", [], fail_content=True)
    result = live._capture_browser_page_projection(_FakeBrowser(page), "http://127.0.0.1:4313")

    assert result["ok"] is False
    assert result["environment_failure_class"] == "browser_capture_error"
    assert result["raw_html_stored"] is False
    assert all(result["observation"][axis] is None for axis in live.PAGE_CAPTURE_AXES)
    merged = live._merge_observation(
        result["observation"],
        {"failure_feedback": {"failure_class": "none"}},
        role="candidate",
        typed_available=True,
        page_failure=result["environment_failure_class"],
    )
    assert merged["failure_feedback"]["failure_class"] == "environment_failure"
    assert merged["failure_feedback"]["environment_failure_class"] == "browser_capture_error"
    target = live._role_target(ROUTES[0], "candidate", environment_failure="browser_capture_error")
    assert target["question"] == "ask_failure"
    assert target["next_action"] == "ask"
    assert target["safe_to_send"] is False


def test_collect_role_wires_browser_projection_and_marks_capture_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakePG324:
        def __init__(self) -> None:
            self.stopped: list[str] = []

        def _start(self, seed: int, index: int):
            return (
                "unused",
                4314,
                "unused",
                {
                    "reset_id": "pg324-reset-fake",
                    "fresh_target": True,
                    "container_id_sha256": "b" * 64,
                    "network_mode": "none",
                    "external_network": False,
                    "relay_loopback_only": True,
                    "domain_data_write_allowed": False,
                    "database_health_gate": "juice_shop_http_health_ok",
                },
            )

        def _stop(self, name: str) -> None:
            self.stopped.append(name)

        def _safe_browser_oracle(self, browser: object, body: str, route: object, marker: str):
            return {"executed": True, "typed_effect_confirmed": True, "evidence_sha256": "c" * 64}

    request_observation = {
        "request_transport": {"method": "GET", "placement": "query", "content_type_class": "none"},
        "response_transport": {"status_class": "2xx", "status_shape": "numeric", "content_type_class": "json", "connection_outcome": "complete"},
        "failure_feedback": {"failure_class": "none", "failure_stage": "none", "error_shape": "empty", "next_action": "ask", "previous_action": "none"},
    }
    monkeypatch.setattr(live, "capture_loopback", lambda *args, **kwargs: {"observation": request_observation, "target_contacted": True})
    monkeypatch.setattr(
        live,
        "_capture_browser_page_projection",
        lambda browser, origin: {"ok": False, "observation": {axis: None for axis in live.PAGE_CAPTURE_AXES}, "environment_failure_class": "browser_timeout", "blocked_external_count": 2, "raw_html_stored": False},
    )
    fake_pg324 = _FakePG324()

    item = live._collect_role(fake_pg324, object(), seed=33111, route=ROUTES[0], index=0, role="candidate")

    assert item["target"]["question"] == "ask_failure"
    assert item["target"]["next_action"] == "ask"
    assert item["target"]["safe_to_send"] is False
    assert item["observation"]["failure_feedback"]["failure_class"] == "environment_failure"
    assert item["observation"]["failure_feedback"]["environment_failure_class"] == "browser_timeout"
    assert item["target_contacted"] is False
    assert item["details"]["browser_page_capture"] == {"ok": False, "blocked_external_count": 2, "environment_failure_class": "browser_timeout", "raw_html_stored": False}
    assert fake_pg324.stopped == ["sift-pg324-juice-33111-0"]
