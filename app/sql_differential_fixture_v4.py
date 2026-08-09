"""Independent v4 SQL-channel fixture for PG-195 source holdout.

This is a tiny loopback HTTP service whose only "SQL" operation is a
shape-only differential.  It never parses or executes SQL and never touches a
database.  Its handler and wire contract are independent from the v3 fixture,
so a typed result is useful for source-holdout testing without being confused
with a Pikachu backend result.
"""

from __future__ import annotations

import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx

from .detection_payload import build_detection_payload
from .cross_lab_safe_catalog import sha256_json


V4_SCHEMA = "sift-sql-differential-fixture-v4"
V4_ORACLE = "synthetic_sql_shape_differential_v4"
V4_PORTS = (8820, 8821, 8822)
V4_VARIANTS = frozenset({"delta", "epsilon", "zeta"})
V4_MODES = frozenset({"baseline", "literal", "syntax", "branch", "row", "timing", "local"})
_MODE_FRAGMENT = {
    "baseline": "plain",
    "literal": "quoted_value",
    "syntax": "syntax_error",
    "branch": "blind_boolean",
    "row": "row_shape",
    "timing": "time_delay",
    "local": "local_side_channel",
}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _shape_for(mode: str) -> dict[str, Any]:
    # Abstract nodes only: this is intentionally not a SQL parser or query
    # builder, and the values are never sent to a database.
    base = {"kind": "lookup", "source": "records", "bind": "name", "projection": ["id"]}
    if mode == "syntax":
        return {"kind": "parse_boundary", "location": "filter", "source": "records"}
    if mode == "branch":
        return {**base, "branch": {"true": "rows", "false": "empty"}}
    if mode == "row":
        return {**base, "projection": ["id", "label"]}
    if mode == "timing":
        return {**base, "latency_bucket": "bounded_100ms"}
    if mode == "local":
        return {**base, "side_channel": "loopback_marker", "external": False}
    if mode == "literal":
        return {**base, "bind": "literal_value"}
    return base


def run_v4_oracle(mode: str, *, variant: str) -> dict[str, Any]:
    mode = str(mode)
    if mode not in V4_MODES:
        raise ValueError("unknown SQL v4 mode")
    baseline = _shape_for("baseline")
    candidate = _shape_for(mode)
    differential = _digest(baseline) != _digest(candidate)
    boundary = mode in {"syntax", "branch", "row", "timing", "local"}
    return {
        "oracle": V4_ORACLE,
        "implementation": "independent_shape_only_v4",
        "variant": variant,
        "mode": mode,
        "modality": "syntax_error" if mode == "syntax" else "bounded_timing" if mode == "timing" else "blind_response" if mode in {"branch", "row"} else "local_side_channel" if mode == "local" else "ast_shape",
        "candidate_signal": bool(differential),
        "controlled_differential": bool(differential),
        "interpreter_boundary": bool(boundary),
        "baseline_shape_sha256": _digest(baseline),
        "candidate_shape_sha256": _digest(candidate),
        "execution": "not_run",
        "database_touched": False,
        "network_access": False,
        "real_sleep_performed": False,
        "simulated_latency_ms": 100 if mode == "timing" else 0,
        "timing_differential": mode == "timing",
        "syntax_error_observed": mode == "syntax",
        "blind_boolean_differential": mode in {"branch", "row"},
        "local_callback_only": mode == "local",
        "external_network": False,
    }


def _response(mode: str, variant: str) -> tuple[int, dict[str, Any]]:
    if mode == "syntax":
        return 400, {"ok": False, "class": "bounded_parse", "variant": variant}
    if mode == "branch":
        return 200, {"ok": True, "branch": "false", "rows": [], "variant": variant}
    if mode == "row":
        return 200, {"ok": True, "rows": [{"id": 1}, {"id": 2}], "variant": variant}
    if mode == "timing":
        return 200, {"ok": True, "latency_bucket": "bounded", "simulated_ms": 100, "variant": variant}
    if mode == "local":
        return 200, {"ok": True, "callback": "loopback", "external": False, "variant": variant}
    return 200, {"ok": True, "rows": [{"id": 1}], "variant": variant}


class _V4Handler(BaseHTTPRequestHandler):
    server_version = "sift-sql-fixture-v4/1"

    def _serve(self, mode: str) -> None:
        variant = str(getattr(self.server, "fixture_variant", "delta"))
        if mode not in V4_MODES or variant not in V4_VARIANTS:
            status, body = 404, {"error": "not_found"}
        else:
            status, body = _response(mode, variant)
        data = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query, keep_blank_values=True)
        if parsed.path not in {"/lookup", "/health"}:
            self._serve("__invalid__")
        elif parsed.path == "/health":
            self._serve("baseline")
        else:
            self._serve(str(query.get("mode", ["baseline"])[0]))

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        length = min(int(self.headers.get("Content-Length", "0") or 0), 4096)
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        form = parse_qs(raw, keep_blank_values=True)
        if parsed.path != "/lookup":
            self._serve("__invalid__")
        else:
            self._serve(str(form.get("mode", ["baseline"])[0]))

    def log_message(self, format: str, *args: Any) -> None:
        return


class SqlV4Server(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], variant: str) -> None:
        if int(address[1]) not in V4_PORTS or variant not in V4_VARIANTS:
            raise ValueError("SQL v4 target or variant is not allow-listed")
        super().__init__(address, _V4Handler)
        self.fixture_variant = variant


def make_sql_v4_fixture_server(*, port: int = 8820, variant: str = "delta") -> SqlV4Server:
    return SqlV4Server(("127.0.0.1", int(port)), variant)


def sql_v4_source_sha256() -> str:
    import pathlib

    return hashlib.sha256(pathlib.Path(__file__).resolve().read_bytes()).hexdigest()


def _summary(response: httpx.Response) -> dict[str, Any]:
    body = response.content
    shape: dict[str, Any] = {"type": "other", "key_count": 0, "scalar_count": 0}
    try:
        value = response.json()
        if isinstance(value, dict):
            shape = {"type": "object", "key_count": len(value), "scalar_count": sum(not isinstance(item, (dict, list)) for item in value.values())}
    except (ValueError, json.JSONDecodeError):
        pass
    return {"status_code": int(response.status_code), "body_length": len(body), "body_sha256": hashlib.sha256(body).hexdigest(), "json_shape": shape}


def collect_sql_v4(*, target: str, port: int, variant: str, method: str, mode: str, sample_id: str) -> dict[str, Any]:
    """Send one bounded abstract mode and return a projection-only record."""

    method = str(method).upper()
    if target.rstrip("/") != f"http://127.0.0.1:{port}" or port not in V4_PORTS:
        raise ValueError("SQL v4 target must be allow-listed loopback")
    if method not in {"GET", "POST"} or mode not in V4_MODES:
        raise ValueError("SQL v4 method/mode is not allow-listed")
    target = target.rstrip("/")
    payload = build_detection_payload(target=target, method=method, path="/lookup", marker=f"pg195-{sample_id}", probe=_MODE_FRAGMENT[mode], probe_kind="sql_channel_class", form={"mode": mode} if method == "POST" else None)
    with httpx.Client(base_url=target, timeout=5.0, follow_redirects=False) as client:
        baseline = client.get("/lookup", params={"mode": "baseline"})
        if method == "GET":
            response = client.get("/lookup", params={"mode": mode})
        else:
            response = client.post("/lookup", data={"mode": mode})
    baseline_summary, response_summary = _summary(baseline), _summary(response)
    oracle = run_v4_oracle(mode, variant=variant)
    oracle["evidence_hash"] = sha256_json(oracle)
    envelope = {"schema": V4_SCHEMA, "target": target, "method": method, "mode": mode, "sample_id": sample_id, "baseline": baseline_summary, "response": response_summary, "oracle": oracle, "fresh_target": True, "database_touched": False, "external_network": False, "raw_response_stored": False}
    return {
        "schema_version": V4_SCHEMA,
        "sample_id": str(sample_id),
        "target": target,
        "method": method,
        "mode": mode,
        "variant": variant,
        "payload_sha256": payload["payload_sha256"],
        "response_projection": response_summary,
        "oracle_projection": oracle,
        "evidence_hash": sha256_json(envelope),
        "fresh_target": True,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "database_touched": False,
        "external_network": False,
    }


__all__ = ["V4_MODES", "V4_PORTS", "V4_VARIANTS", "SqlV4Server", "collect_sql_v4", "make_sql_v4_fixture_server", "run_v4_oracle", "sql_v4_source_sha256"]
