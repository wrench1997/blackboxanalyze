"""Loopback-only front/back runtime for the PG-367 WAF staircase.

It serves an abstract HTML surface and a bounded GET/POST endpoint.  The
request's concrete canary is never returned or persisted; only abstract
headers select a reviewed probe shape for the evaluator projection.
"""

from __future__ import annotations

import html
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .pg367_waf_staircase import POLICIES, evaluate_waf_probe


SCHEMA_VERSION = "pg367-waf-loopback-runtime-v1"


def _policy_map() -> dict[str, Any]:
    return {policy.policy_id: policy for policy in POLICIES}


def _page(policy_id: str) -> bytes:
    safe_id = html.escape(policy_id, quote=True)
    return (
        "<!doctype html><html lang='en'><head><title>WAF staircase fixture</title>"
        "<meta name='pg367-runtime' content='loopback-only'></head>"
        f"<body data-waf-policy='{safe_id}' data-state-write='false'>"
        "<main><form method='post' action='/pg367/probe' data-submit-policy='evaluator-only'>"
        "<input name='probe' value='runtime-canary' readonly><button type='submit'>probe</button>"
        "</form></main></body></html>"
    ).encode("utf-8")


class _Handler(BaseHTTPRequestHandler):
    server_version = "PG367Loopback/1"

    def _write(self, status: int, body: bytes, content_type: str = "text/html; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _projection(self, method: str, path: str, body: bytes = b"") -> dict[str, Any]:
        parts = urlsplit(path)
        policy_id = parts.path.rsplit("/", 1)[-1] if parts.path.startswith("/pg367/waf/") else ""
        policy = _policy_map().get(policy_id)
        if policy is None:
            raise ValueError("route_not_allowlisted")
        role = self.headers.get("X-PG367-Role", "candidate")
        syntax = self.headers.get("X-PG367-Syntax", "marker")
        encoding = self.headers.get("X-PG367-Encoding", "identity")
        field_role = self.headers.get("X-PG367-Field", "query_term")
        # The body/query is intentionally ignored as a concrete value.  Only
        # reviewed abstract headers enter the evaluator projection.
        _ = parse_qs(parts.query), body
        projection = evaluate_waf_probe(policy, {"role": role, "method": method, "field_role": field_role, "syntax_category": syntax, "encoding_chain": encoding})
        return projection

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/pg367/health":
            self._write(200, b"ok", "text/plain; charset=utf-8")
            return
        if self.path.startswith("/pg367/page/"):
            policy_id = self.path.rsplit("/", 1)[-1]
            if policy_id not in _policy_map():
                self._write(404, b"not found")
                return
            self._write(200, _page(policy_id))
            return
        try:
            projection = self._projection("GET", self.path)
        except ValueError:
            self._write(404, b"not found")
            return
        body = json.dumps({"schema_version": SCHEMA_VERSION, "projection": projection}, separators=(",", ":")).encode("utf-8")
        self._write(200, body, "application/json; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        length = min(int(self.headers.get("Content-Length", "0") or 0), 4096)
        body = self.rfile.read(length)
        try:
            projection = self._projection("POST", self.path, body)
        except ValueError:
            self._write(404, b"not found")
            return
        response = json.dumps({"schema_version": SCHEMA_VERSION, "projection": projection}, separators=(",", ":")).encode("utf-8")
        self._write(200, response, "application/json; charset=utf-8")

    def log_message(self, *_args: Any) -> None:
        return


def start_runtime() -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, name="pg367-loopback", daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_port}"


__all__ = ["SCHEMA_VERSION", "start_runtime"]
