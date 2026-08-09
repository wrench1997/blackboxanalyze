"""Independent local fixture for PG-106 causal-projection decoy tests.

The implementation deliberately has two routes: a boundary case whose
typed-positive response is opaque, and a harmless decoy where an input shape
changes while the response stays constant.  No markup, SQL, commands,
redirects, or external state are used.
"""

from __future__ import annotations

import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit


PG106_SCHEMA = "sift-pg106-independent-decoy-fixture-v1"
PG106_VARIANTS = ("amber", "violet")
PG106_PORTS = (8815, 8816)
PG106_ROUTES = frozenset({"/threshold", "/decoy"})


def source_sha256() -> str:
    from pathlib import Path

    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def evaluate_workflow(route: str, values: dict[str, str], variant: str) -> tuple[int, dict[str, Any], dict[str, Any]]:
    if route == "/threshold":
        member = values.get("subject") == "member"
        boundary_hit = values.get("value") == "100"
        violation = bool(member and boundary_hit)
        response = {"ok": False, "state": "boundary"} if variant == "amber" else {"accepted": False, "phase": "boundary"}
        return 200, response, {
            "oracle_id": "pg106-boundary-consistency-v1",
            "oracle_signal": "bounded_invariant_mismatch",
            "positive": violation,
            "positive_authority": True,
            "typed": {"subject_member": member, "boundary_hit": boundary_hit, "invariant_violation": violation},
        }
    if route == "/decoy":
        response = {"ok": True, "state": "steady"} if variant == "amber" else {"accepted": True, "phase": "steady"}
        return 200, response, {
            "oracle_id": "pg106-decoy-negative-v1",
            "oracle_signal": "no_typed_violation",
            "positive": False,
            "positive_authority": True,
            "typed": {"ordinary_response": True, "invariant_violation": False},
        }
    return 404, {"error": "not_found"}, {
        "oracle_id": "pg106-ordinary-negative-v1",
        "oracle_signal": "no_typed_violation",
        "positive": False,
        "positive_authority": True,
        "typed": {"ordinary_response": True, "invariant_violation": False},
    }


class _Handler(BaseHTTPRequestHandler):
    server_version = "sift-pg106-independent/1"

    def _serve(self, raw_path: str, values: dict[str, str]) -> None:
        route = urlsplit(raw_path).path
        variant = str(getattr(self.server, "fixture_variant", "amber"))
        if route not in PG106_ROUTES or variant not in PG106_VARIANTS:
            status, response = 404, {"error": "not_found"}
        else:
            status, response, _ = evaluate_workflow(route, values, variant)
        body = json.dumps(response, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        values = {str(key): str(items[0]) for key, items in parse_qs(parsed.query, keep_blank_values=True).items() if items}
        self._serve(self.path, values)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length", "0") or 0)
        encoded = self.rfile.read(min(length, 4096)).decode("utf-8", errors="replace")
        values = {str(key): str(items[0]) for key, items in parse_qs(encoded, keep_blank_values=True).items() if items}
        self._serve(self.path, values)

    def log_message(self, format: str, *args: Any) -> None:
        return


class DecoyFixtureServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], variant: str) -> None:
        if variant not in PG106_VARIANTS:
            raise ValueError("unknown PG-106 variant")
        super().__init__(address, _Handler)
        self.fixture_variant = variant


def make_workflow_server(port: int, variant: str) -> DecoyFixtureServer:
    if int(port) not in PG106_PORTS:
        raise ValueError("PG-106 port is not allow-listed")
    return DecoyFixtureServer(("127.0.0.1", int(port)), variant)


__all__ = [
    "PG106_PORTS",
    "PG106_ROUTES",
    "PG106_SCHEMA",
    "PG106_VARIANTS",
    "evaluate_workflow",
    "make_workflow_server",
    "source_sha256",
]
