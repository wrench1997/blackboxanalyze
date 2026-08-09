"""Independent, read-only HTTP fixture for PG-34 family holdout.

This module intentionally does not import FastAPI or the repository maze
routes.  It is a tiny standalone HTTP implementation used only on loopback.
Inputs are bounded abstract probe classes; no script, SQL, command, URL or
credential is accepted or executed.  The response contains bounded typed
signals and never stores the request.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit


INDEPENDENT_FIXTURE_SCHEMA = "sift-pg34-independent-http-fixture-v1"

SURFACE_SPECS: dict[str, dict[str, Any]] = {
    "surface-01": {
        "family": "xss",
        "field": "probe_class",
        "positive": "markup_candidate",
        "effect": "dom_structure",
    },
    "surface-02": {
        "family": "injection",
        "field": "fragment_class",
        "positive": "operator_like",
        "effect": "interpreter_boundary",
    },
    "surface-03": {
        "family": "authentication",
        "field": "probe_class",
        "positive": "auth_boundary_candidate",
        "effect": "authentication_boundary",
    },
    "surface-04": {
        "family": "access_control",
        "field": "probe_class",
        "positive": "id_reference",
        "effect": "authorization_boundary",
    },
    "surface-05": {
        "family": "logic",
        "field": "probe_class",
        "positive": "invariant_boundary",
        "effect": "business_invariant",
    },
    "surface-06": {
        "family": "url_redirect",
        "field": "probe_class",
        "positive": "relative_redirect",
        "effect": "redirect_origin",
    },
    "surface-07": {
        "family": "input_validation",
        "field": "probe_class",
        "positive": "boundary_value",
        "effect": "validation_boundary",
    },
    "surface-08": {
        "family": "command_injection",
        "field": "probe_class",
        "positive": "local_canary",
        "effect": "command_canary",
    },
    "surface-09": {
        "family": "ordinary_response",
        "field": "probe_class",
        "positive": "never_positive",
        "effect": "none",
    },
}


def _bounded_params(handler: BaseHTTPRequestHandler) -> dict[str, str]:
    if handler.command == "GET":
        query = parse_qs(urlsplit(handler.path).query, keep_blank_values=True)
        return {str(key)[:64]: str(values[0])[:96] for key, values in list(query.items())[:8]}
    try:
        length = min(int(handler.headers.get("content-length", "0")), 2048)
    except ValueError:
        length = 0
    raw = handler.rfile.read(max(length, 0))
    content_type = str(handler.headers.get("content-type", ""))
    if content_type.startswith("application/json"):
        try:
            value = json.loads(raw.decode("utf-8", errors="replace"))
        except (TypeError, ValueError):
            value = {}
        items = list(value.items())[:8] if isinstance(value, dict) else []
        return {
            str(key)[:64]: str(child)[:96]
            for key, child in items
            if isinstance(child, (str, int, float, bool))
        }
    parsed = parse_qs(raw.decode("utf-8", errors="replace"), keep_blank_values=True)
    return {str(key)[:64]: str(values[0])[:96] for key, values in list(parsed.items())[:8]}


def _body(surface_id: str, params: dict[str, str]) -> dict[str, Any]:
    spec = SURFACE_SPECS[surface_id]
    is_positive = params.get(spec["field"], "normal") == spec["positive"]
    common = {
        "surface_slot": surface_id,
        "candidate_signal": is_positive,
        "state_mutated": False,
        "database_touched": False,
        "credentials_accessed": False,
        "external_network": False,
        "script_execution": False,
    }
    if spec["family"] == "xss":
        common.update({"dom_change": is_positive, "marker_hits": 1 if is_positive else 0, "network_access": False})
    elif spec["family"] == "injection":
        common.update({"controlled_differential": is_positive, "interpreter_boundary": is_positive})
    elif spec["family"] == "authentication":
        common.update({"authentication_boundary": is_positive, "authenticated": False})
    elif spec["family"] == "access_control":
        common.update({"authorization_boundary": is_positive, "cross_subject_access": is_positive})
    elif spec["family"] == "logic":
        common.update({"business_invariant_boundary": is_positive, "history_changed": False})
    elif spec["family"] == "url_redirect":
        common.update({"redirect_candidate": is_positive, "same_origin": True, "external_redirect": False})
    elif spec["family"] == "input_validation":
        common.update({"validation_boundary": is_positive, "rejected": is_positive})
    elif spec["family"] == "ordinary_response":
        common.update({"ordinary_response": True, "candidate_signal": False})
    else:
        # Keep the standalone implementation on the shared typed-effect
        # contract.  This is a bounded evaluator signal only: no command is
        # accepted, executed, or persisted by this fixture.
        common.update({
            "local_canary_observed": is_positive,
            "command_executed": False,
            "execution_blocked": True,
            "canary_count": 1 if is_positive else 0,
        })
    return common


class IndependentFixtureHandler(BaseHTTPRequestHandler):
    server_version = "PG34IndependentFixture/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        # Never log request values or response content.
        return

    def _serve(self) -> None:
        path = urlsplit(self.path).path
        prefix = "/pg34/surface/"
        surface_id = path[len(prefix):] if path.startswith(prefix) else ""
        if surface_id not in SURFACE_SPECS:
            self._write(404, {"ok": False, "surface": "unknown"})
            return
        params = _bounded_params(self)
        self._write(200, _body(surface_id, params))

    def _write(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(int(status))
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        self._serve()

    def do_POST(self) -> None:  # noqa: N802
        self._serve()


def make_independent_fixture_server(port: int = 31934) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", int(port)), IndependentFixtureHandler)


__all__ = ["INDEPENDENT_FIXTURE_SCHEMA", "SURFACE_SPECS", "IndependentFixtureHandler", "make_independent_fixture_server"]
