"""Independent stateless workflow fixture for PG-76 unknown-family triplets."""

from __future__ import annotations

import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit


FAMILY = "workflow_invariant"
PORT = 8818
VARIANTS = ("copper", "teal", "indigo")
ROUTES = ("/commit", "/quota")


def _values(handler: BaseHTTPRequestHandler) -> dict[str, str]:
    if handler.command == "POST":
        length = int(handler.headers.get("content-length", "0") or 0)
        body = handler.rfile.read(min(length, 4096)).decode("utf-8", errors="replace")
        parsed = parse_qs(body, keep_blank_values=True)
    else:
        parsed = parse_qs(urlsplit(handler.path).query, keep_blank_values=True)
    return {str(key): str(value[-1]) for key, value in parsed.items()}


def evaluate(route: str, values: dict[str, str], variant: str) -> tuple[bool, str]:
    if route == "/commit":
        positive = values.get("verb") == "commit" and values.get("prior") == "verified" and values.get("fresh") == "new"
        return positive, "workflow_commit_transition"
    positive = values.get("member") == "1" and values.get("amount") == "100"
    return positive, "workflow_quota_boundary"


def make_server(port: int = PORT, variant: str = "copper") -> ThreadingHTTPServer:
    selected = str(variant)

    class Handler(BaseHTTPRequestHandler):
        server_version = "PG76Fixture/1.0"

        def do_GET(self) -> None:  # noqa: N802
            self._respond()

        def do_POST(self) -> None:  # noqa: N802
            self._respond()

        def _respond(self) -> None:
            route = urlsplit(self.path).path
            values = _values(self)
            positive, signal = evaluate(route, values, selected)
            status = 200 if positive else 403
            payload = {"variant": selected, "status": "transition" if positive else "blocked", "signal": signal}
            body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("x-pg76-variant", selected)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return ThreadingHTTPServer(("127.0.0.1", int(port)), Handler)


def source_sha256() -> str:
    import pathlib

    return hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()


__all__ = ["FAMILY", "PORT", "ROUTES", "VARIANTS", "evaluate", "make_server", "source_sha256"]
