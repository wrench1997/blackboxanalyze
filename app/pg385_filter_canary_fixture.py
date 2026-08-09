"""Local-only filter feedback fixture for the PG-385 repair demo.

This is a deliberately inert page: a canary marker followed by a delimiter is
either rejected in its raw form or accepted after an allow-listed encoding
change.  It never executes markup, touches a database, follows a callback, or
returns the submitted value.  The evaluator exposes only bounded response
shape and typed-effect fields to the repair policy.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote_plus, urlsplit


SCHEMA_VERSION = "pg385-filter-canary-fixture-v1"
ROUTE_PATH = "/pg385/filter"
FIELD_NAME = "q"
_RAW_FIELD_RE = re.compile(r"(?:^|&)q=([^&]*)", re.IGNORECASE)
_FORBIDDEN_RAW = (":", "[")


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _projection(
    *,
    state: str,
    filter_class: str,
    failure_shape: str,
    effect_class: str,
    typed_effect_confirmed: bool,
    status_class: str,
    encoding_acceptance: str,
) -> dict[str, Any]:
    projection = {
        "status_class": status_class,
        "response_shape": "bounded_json_projection",
        "filter_state": state,
        "filter_class": filter_class,
        "failure_shape": failure_shape,
        "effect_class": effect_class,
        "typed_effect_confirmed": bool(typed_effect_confirmed),
        "encoding_acceptance": encoding_acceptance,
        "external_network": False,
        "raw_response_stored": False,
    }
    projection["evidence_sha256"] = _sha(projection)
    return projection


def evaluate_raw_request(*, method: str, raw_query: str = "", raw_body: str = "") -> dict[str, Any]:
    """Return only a bounded evaluator projection for one local request."""

    method = str(method).upper()
    source = raw_query if method == "GET" else raw_body
    match = _RAW_FIELD_RE.search(source)
    if not match:
        return _projection(
            state="parser_error",
            filter_class="missing_parameter",
            failure_shape="field_not_observed",
            effect_class="none",
            typed_effect_confirmed=False,
            status_class="4xx",
            encoding_acceptance="not_observed",
        )
    raw_component = match.group(1)
    # The toy filter rejects a raw delimiter *and* a single encoded layer.
    # A second, explicitly selected encoding layer is the only accepted
    # repair.  This models a canonicalization-order bug without executing any
    # markup or database syntax.
    if any(marker in raw_component for marker in _FORBIDDEN_RAW) or (
        "%3A" in raw_component.upper() and "%25" not in raw_component.upper()
    ):
        return _projection(
            state="filtered",
            filter_class="encoding_filter",
            failure_shape="raw_delimiter_blocked",
            effect_class="none",
            typed_effect_confirmed=False,
            status_class="4xx",
            encoding_acceptance="encoded_variant_required",
        )

    # The fixture represents a simple canonicalization boundary.  One or
    # more percent-decoding layers are accepted, but the raw delimiter is not.
    normalized = raw_component
    for _ in range(3):
        next_value = unquote_plus(normalized)
        if next_value == normalized:
            break
        normalized = next_value
    if ":" not in normalized:
        return _projection(
            state="no_effect",
            filter_class="none",
            failure_shape="marker_not_reached",
            effect_class="none",
            typed_effect_confirmed=False,
            status_class="2xx",
            encoding_acceptance="no_delimiter",
        )
    if "_NEG_" in normalized:
        return _projection(
            state="no_effect",
            filter_class="matched_negative",
            failure_shape="negative_control",
            effect_class="none",
            typed_effect_confirmed=False,
            status_class="2xx",
            encoding_acceptance="encoded_variant",
        )
    return _projection(
        state="typed_effect",
        filter_class="none",
        failure_shape="none",
        effect_class="bounded_marker_reflection",
        typed_effect_confirmed=True,
        status_class="2xx",
        encoding_acceptance="encoded_variant",
    )


class _Handler(BaseHTTPRequestHandler):
    server_version = "PG385LocalFixture/1"

    def _write_projection(self, projection: dict[str, Any]) -> None:
        body = json.dumps(projection, ensure_ascii=False, sort_keys=True).encode("utf-8")
        status = 200 if projection["status_class"] == "2xx" else 400
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        if parsed.path != ROUTE_PATH:
            self.send_error(404)
            return
        self._write_projection(evaluate_raw_request(method="GET", raw_query=parsed.query))

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        if parsed.path != ROUTE_PATH:
            self.send_error(404)
            return
        try:
            length = min(int(self.headers.get("Content-Length", "0")), 8192)
        except ValueError:
            length = 0
        raw_body = self.rfile.read(max(length, 0)).decode("utf-8", errors="replace")
        self._write_projection(evaluate_raw_request(method="POST", raw_body=raw_body))

    def log_message(self, *_args: Any) -> None:
        return


class FilterCanaryServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int]):
        super().__init__(address, _Handler)
        self.reset_count = 0

    def fresh_reset(self) -> dict[str, Any]:
        self.reset_count += 1
        return {
            "fresh_reset": True,
            "reset_id": self.reset_count,
            "instance_digest": _sha({"reset_id": self.reset_count, "schema": SCHEMA_VERSION}),
            "state_clean": True,
            "external_network": False,
        }


def start_filter_canary_server(*, host: str = "127.0.0.1", port: int = 0) -> tuple[FilterCanaryServer, threading.Thread]:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("PG-385 fixture is loopback-only")
    server = FilterCanaryServer((host, int(port)))
    thread = threading.Thread(target=server.serve_forever, name="pg385-filter-fixture", daemon=True)
    thread.start()
    return server, thread


__all__ = [
    "FIELD_NAME",
    "FilterCanaryServer",
    "ROUTE_PATH",
    "SCHEMA_VERSION",
    "evaluate_raw_request",
    "start_filter_canary_server",
]
