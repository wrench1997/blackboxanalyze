"""Bounded loopback redirect fixture used by the PG-278 data study.

The fixture is deliberately not an open redirect or a network probe.  It only
emits an internal relative Location header and the collector never follows it.
It exists to capture a request/response transition that is often absent from
page-only corpora: status, Location shape, method, and a typed local oracle.
"""

from __future__ import annotations

import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx

from .maze_engine import sha256_json


SCHEMA = "sift-pg278-redirect-fixture-v1"
ORACLE = "pg278_internal_redirect_contract_v1"
PORTS = (8850, 8851, 8852)
VARIANTS = ("amber", "cobalt", "jade")
MODES = ("baseline", "hop", "preserve_hop")
SOURCE_PATH = Path(__file__).resolve()


def source_sha256() -> str:
    return hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()


def _response(mode: str, variant: str) -> tuple[int, dict[str, Any], dict[str, str]]:
    if mode == "hop":
        return 302, {"ok": True, "transition": "internal", "variant": variant}, {"Location": "/landing?flow=internal"}
    if mode == "preserve_hop":
        return 307, {"ok": True, "transition": "internal-preserve", "variant": variant}, {"Location": "/landing?flow=preserve"}
    return 200, {"ok": True, "transition": "none", "variant": variant}, {}


class _Handler(BaseHTTPRequestHandler):
    server_version = "sift-pg278-redirect/1"

    def _serve(self, mode: str) -> None:
        variant = str(getattr(self.server, "fixture_variant", "amber"))
        if mode not in MODES or variant not in VARIANTS:
            status, body, headers = 404, {"ok": False, "kind": "not_found"}, {}
        else:
            status, body, headers = _response(mode, variant)
        data = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        mode = str(parse_qs(parsed.query, keep_blank_values=True).get("mode", ["baseline"])[0])
        self._serve(mode if parsed.path == "/route" else "__invalid__")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        size = min(int(self.headers.get("Content-Length", "0") or 0), 4096)
        values = parse_qs(self.rfile.read(size).decode("utf-8", errors="replace"), keep_blank_values=True)
        mode = str(values.get("mode", ["baseline"])[0])
        self._serve(mode if parsed.path == "/route" else "__invalid__")

    def log_message(self, format: str, *args: Any) -> None:
        return


class RedirectServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], variant: str) -> None:
        if int(address[1]) not in PORTS or variant not in VARIANTS:
            raise ValueError("redirect fixture target is not allow-listed")
        super().__init__(address, _Handler)
        self.fixture_variant = variant


def make_server(*, port: int, variant: str) -> RedirectServer:
    return RedirectServer(("127.0.0.1", int(port)), variant)


def _summary(response: httpx.Response) -> dict[str, Any]:
    location = str(response.headers.get("location", ""))
    return {
        "status_class": f"{response.status_code // 100}xx",
        "status_code": int(response.status_code),
        "content_type_class": "json" if "json" in str(response.headers.get("content-type", "")).casefold() else "other",
        "location_class": "internal_relative" if location.startswith("/") else "absent",
        "location_present": bool(location),
        "body_sha256": hashlib.sha256(response.content).hexdigest(),
        "body_length": len(response.content),
    }


def collect(*, target: str, port: int, variant: str, method: str, mode: str, sample_id: str) -> dict[str, Any]:
    method = str(method).upper()
    expected = f"http://127.0.0.1:{int(port)}"
    if target.rstrip("/") != expected or int(port) not in PORTS or variant not in VARIANTS or method not in {"GET", "POST"} or mode not in MODES:
        raise ValueError("redirect collection request is outside the bounded fixture")
    with httpx.Client(base_url=expected, timeout=5.0, follow_redirects=False, cookies={}) as client:
        baseline = client.get("/route", params={"mode": "baseline"})
        response = client.get("/route", params={"mode": mode}) if method == "GET" else client.post("/route", data={"mode": mode})
    baseline_projection, response_projection = _summary(baseline), _summary(response)
    typed_positive = bool(mode in {"hop", "preserve_hop"} and response_projection["location_class"] == "internal_relative" and response_projection["status_class"] == "3xx")
    envelope = {
        "schema_version": SCHEMA,
        "target": expected,
        "method": method,
        "mode": mode,
        "variant": variant,
        "baseline": baseline_projection,
        "response": response_projection,
        "oracle": {"name": ORACLE, "typed_positive": typed_positive, "external_navigation": False, "followed": False},
        "fresh_target": True,
        "external_network": False,
        "database_touched": False,
    }
    return {
        "schema_version": SCHEMA,
        "sample_id": sample_id,
        "target": expected,
        "method": method,
        "mode": mode,
        "variant": variant,
        "response_projection": response_projection,
        "baseline_projection": baseline_projection,
        "oracle_projection": envelope["oracle"],
        "evidence_hash": sha256_json(envelope),
        "fresh_target": True,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "external_network": False,
        "database_touched": False,
    }


__all__ = ["MODES", "ORACLE", "PORTS", "VARIANTS", "collect", "make_server", "source_sha256"]
