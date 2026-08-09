"""Independent SQL-shape v6 fixture for PG-200 source-heldout evaluation.

This fixture models parser/plan differences without executing SQL, sleeping,
touching a database, or returning an unrestricted error body.  Its wire
responses are bounded JSON projections only.
"""

from __future__ import annotations

import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx

from .cross_lab_safe_catalog import sha256_json
from .detection_payload import build_detection_payload


V6_SCHEMA = "sift-sql-differential-fixture-v6"
V6_ORACLE = "synthetic_sql_shape_differential_v6"
V6_PORTS = (8840, 8841, 8842)
V6_VARIANTS = frozenset({"obsidian", "pearl", "saffron"})
V6_MODES = frozenset({"baseline", "branch", "syntax", "row", "timeout_shape", "error_redirect"})
_MODE_FRAGMENT = {
    "baseline": "plain",
    "branch": "blind_boolean",
    "syntax": "syntax_error",
    "row": "row_shape",
    "timeout_shape": "time_delay",
    "error_redirect": "local_side_channel",
}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _shape(mode: str, variant: str) -> dict[str, Any]:
    base = {
        "plan": "lookup",
        "predicate": "bound",
        "projection": ["key"],
        "variant_class": variant,
    }
    if mode == "branch":
        return {**base, "predicate": {"kind": "conditional", "true": "nonempty", "false": "empty"}}
    if mode == "syntax":
        return {"plan": "parse_failure", "stage": "predicate", "variant_class": variant}
    if mode == "row":
        return {**base, "projection": ["key", "label", "class"]}
    if mode == "timeout_shape":
        return {**base, "latency_bucket": "bounded_high", "budgeted": True}
    if mode == "error_redirect":
        return {**base, "error_channel": "local_redirect", "status_class": "3xx"}
    return base


def run_v6_oracle(mode: str, *, variant: str) -> dict[str, Any]:
    mode = str(mode)
    if mode not in V6_MODES or variant not in V6_VARIANTS:
        raise ValueError("unknown SQL v6 mode or variant")
    baseline, candidate = _shape("baseline", variant), _shape(mode, variant)
    differential = _digest(baseline) != _digest(candidate)
    boundary = mode in {"branch", "syntax", "row", "timeout_shape", "error_redirect"}
    return {
        "oracle": V6_ORACLE,
        "implementation": "relational_plan_projection_v6",
        "variant": variant,
        "mode": mode,
        "modality": "syntax_error" if mode == "syntax" else "bounded_timing_shape" if mode == "timeout_shape" else "redirect_shape" if mode == "error_redirect" else "plan_shape",
        "candidate_signal": bool(differential),
        "controlled_differential": bool(differential),
        "interpreter_boundary": bool(boundary),
        "baseline_shape_sha256": _digest(baseline),
        "candidate_shape_sha256": _digest(candidate),
        "execution": "not_run",
        "database_touched": False,
        "network_access": False,
        "real_sleep_performed": False,
        "external_network": False,
        "evidence_hash": _digest({"mode": mode, "variant": variant, "baseline": _digest(baseline), "candidate": _digest(candidate), "boundary": boundary}),
    }


def _response(mode: str, variant: str) -> tuple[int, dict[str, Any], dict[str, str]]:
    if mode == "syntax":
        return 422, {"ok": False, "kind": "bounded_parse", "variant": variant}, {}
    if mode == "branch":
        return 200, {"ok": True, "branch": "true", "rows": [], "variant": variant}, {}
    if mode == "row":
        return 200, {"ok": True, "rows": [{"key": 1}], "shape": "three_field", "variant": variant}, {}
    if mode == "timeout_shape":
        return 200, {"ok": True, "latency_bucket": "bounded_high", "real_sleep": False, "variant": variant}, {}
    if mode == "error_redirect":
        return 302, {"ok": False, "kind": "local_redirect", "variant": variant}, {"Location": "/query?mode=baseline"}
    return 200, {"ok": True, "rows": [{"key": 1}], "variant": variant}, {}


class _V6Handler(BaseHTTPRequestHandler):
    server_version = "sift-sql-fixture-v6/1"

    def _serve(self, mode: str) -> None:
        variant = str(getattr(self.server, "fixture_variant", "obsidian"))
        status, body, headers = _response(mode, variant) if mode in V6_MODES and variant in V6_VARIANTS else (404, {"error": "not_found"}, {})
        data = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        mode = str(parse_qs(parsed.query, keep_blank_values=True).get("mode", ["baseline"])[0])
        self._serve(mode if parsed.path == "/query" else "__invalid__")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        length = min(int(self.headers.get("Content-Length", "0") or 0), 4096)
        form = parse_qs(self.rfile.read(length).decode("utf-8", errors="replace"), keep_blank_values=True)
        self._serve(str(form.get("mode", ["baseline"])[0]) if parsed.path == "/query" else "__invalid__")

    def log_message(self, format: str, *args: Any) -> None:
        return


class SqlV6Server(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], variant: str) -> None:
        if int(address[1]) not in V6_PORTS or variant not in V6_VARIANTS:
            raise ValueError("SQL v6 target or variant is not allow-listed")
        super().__init__(address, _V6Handler)
        self.fixture_variant = variant


def make_sql_v6_fixture_server(*, port: int = 8840, variant: str = "obsidian") -> SqlV6Server:
    return SqlV6Server(("127.0.0.1", int(port)), variant)


def sql_v6_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def _summary(response: httpx.Response) -> dict[str, Any]:
    body = response.content
    shape = {"type": "other", "key_count": 0}
    try:
        value = response.json()
        if isinstance(value, dict):
            shape = {"type": "object", "key_count": len(value)}
    except (ValueError, json.JSONDecodeError):
        pass
    return {
        "status_code": int(response.status_code),
        "body_length": len(body),
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "json_shape": shape,
        "location_present": bool(response.headers.get("location")),
    }


def collect_sql_v6(*, target: str, port: int, variant: str, method: str, mode: str, sample_id: str) -> dict[str, Any]:
    method = str(method).upper()
    if target.rstrip("/") != f"http://127.0.0.1:{port}" or port not in V6_PORTS or method not in {"GET", "POST"} or mode not in V6_MODES:
        raise ValueError("SQL v6 target/method/mode is not allow-listed")
    payload = build_detection_payload(
        target=target,
        method=method,
        path="/query",
        marker=f"pg200-{sample_id}",
        probe=_MODE_FRAGMENT[mode],
        probe_kind="sql_channel_class",
        form={"mode": mode} if method == "POST" else None,
    )
    with httpx.Client(base_url=target, timeout=5.0, follow_redirects=False) as client:
        baseline = client.get("/query", params={"mode": "baseline"})
        response = client.get("/query", params={"mode": mode}) if method == "GET" else client.post("/query", data={"mode": mode})
    oracle = run_v6_oracle(mode, variant=variant)
    envelope = {
        "schema": V6_SCHEMA,
        "target": target,
        "method": method,
        "mode": mode,
        "variant": variant,
        "baseline": _summary(baseline),
        "response": _summary(response),
        "oracle": oracle,
        "fresh_target": True,
        "database_touched": False,
        "external_network": False,
    }
    return {
        "schema_version": V6_SCHEMA,
        "sample_id": sample_id,
        "target": target,
        "method": method,
        "mode": mode,
        "variant": variant,
        "payload_sha256": payload["payload_sha256"],
        "response_projection": _summary(response),
        "oracle_projection": oracle,
        "evidence_hash": sha256_json(envelope),
        "fresh_target": True,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "database_touched": False,
        "external_network": False,
    }


__all__ = [
    "V6_MODES", "V6_PORTS", "V6_VARIANTS", "collect_sql_v6", "make_sql_v6_fixture_server",
    "run_v6_oracle", "sql_v6_source_sha256",
]
