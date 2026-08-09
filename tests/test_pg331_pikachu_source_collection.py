from __future__ import annotations

import importlib.util
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_pg331_pikachu_source_collection.py"
SPEC = importlib.util.spec_from_file_location("pg331_source_collection", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

RELAY_SPEC = importlib.util.spec_from_file_location("pg331_pikachu_relay", ROOT / "app" / "pg331_pikachu_docker_relay.py")
assert RELAY_SPEC is not None and RELAY_SPEC.loader is not None
RELAY = importlib.util.module_from_spec(RELAY_SPEC)
RELAY_SPEC.loader.exec_module(RELAY)


def test_route_plan_keeps_get_and_post_neutral_and_allowlisted() -> None:
    assert {str(route["method"]).upper() for route in MODULE.ROUTES} == {"GET", "POST"}
    get_route = next(route for route in MODULE.ROUTES if route["method"] == "GET" and route["fields"])
    url = MODULE._route_url("http://127.0.0.1:1234", get_route)
    assert url.startswith("http://127.0.0.1:1234/")
    assert "name=" in url and "submit=" in url
    assert "payload" not in url.casefold()
    assert any(route["method"] == "POST" and route["fields"] for route in MODULE.ROUTES)


def test_incomplete_baseline_target_is_safe_typed_ask() -> None:
    target = MODULE._target_projection()
    assert target["question"] == "ask_typed_oracle"
    assert target["next_action"] == "ask_typed"
    assert target["safe_to_send"] is False
    assert target["probe_variant_ref"] == "none"


def test_source_metadata_is_sidecar_digest_only() -> None:
    route = next(route for route in MODULE.ROUTES if route["method"] == "POST")
    meta = MODULE._source_meta(route)
    assert meta["image_digest"] == MODULE.IMAGE.split("@sha256:", 1)[1]
    assert len(meta["source_digest"]) == 64
    assert "/vul/" not in meta["source_digest"]


class _FakeBridge:
    def request(self, method: str, path: str, *, body: bytes = b"", headers=None):
        return {"status": 200, "headers": {"content-type": "text/html"}, "body": b"<html><body>ok</body></html>"}


def test_relay_is_loopback_only_and_forwards_bounded_origin_relative_requests() -> None:
    relay = RELAY.LoopbackRelay(RELAY.PhpDockerBridge.__new__(RELAY.PhpDockerBridge))
    # Replace the bridge after construction so this test never starts Docker.
    relay.httpd.bridge = _FakeBridge()
    try:
        response = httpx.get(f"http://127.0.0.1:{relay.port}/safe?field=", timeout=2.0)
        assert response.status_code == 200
        assert response.text == "<html><body>ok</body></html>"
        blocked = httpx.get(f"http://127.0.0.1:{relay.port}/https://example.invalid/", timeout=2.0)
        assert blocked.status_code == 400
    finally:
        relay.close()
