"""Independent v5 SQL shape fixture for PG-197 cross-implementation checks."""

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


V5_SCHEMA = "sift-sql-differential-fixture-v5"
V5_ORACLE = "synthetic_sql_shape_differential_v5"
V5_PORTS = (8830, 8831, 8832)
V5_VARIANTS = frozenset({"indigo", "jade", "krypton"})
V5_MODES = frozenset({"baseline", "literal", "branch", "syntax", "row"})
_MODE_FRAGMENT = {"baseline": "plain", "literal": "quoted_value", "branch": "blind_boolean", "syntax": "syntax_error", "row": "row_shape"}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _shape(mode: str, variant: str) -> dict[str, Any]:
    base = {"node": "lookup", "bind": "parameter", "projection": ["id"], "variant_class": variant}
    if mode == "literal":
        return {**base, "bind": "literal"}
    if mode == "branch":
        return {**base, "predicate": {"kind": "branch", "true": "rows", "false": "empty"}}
    if mode == "syntax":
        return {"node": "parse_error", "scope": "predicate", "variant_class": variant}
    if mode == "row":
        return {**base, "projection": ["id", "label"]}
    return base


def run_v5_oracle(mode: str, *, variant: str) -> dict[str, Any]:
    mode = str(mode)
    if mode not in V5_MODES or variant not in V5_VARIANTS:
        raise ValueError("unknown SQL v5 mode or variant")
    baseline, candidate = _shape("baseline", variant), _shape(mode, variant)
    differential = _digest(baseline) != _digest(candidate)
    boundary = mode in {"branch", "syntax", "row"}
    return {
        "oracle": V5_ORACLE,
        "implementation": "independent_shape_graph_v5",
        "variant": variant,
        "mode": mode,
        "modality": "syntax_error" if mode == "syntax" else "blind_response" if mode in {"branch", "row"} else "ast_shape",
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


def _response(mode: str, variant: str) -> tuple[int, dict[str, Any]]:
    if mode == "syntax":
        return 400, {"ok": False, "kind": "bounded_parse", "variant": variant}
    if mode == "branch":
        return 200, {"ok": True, "branch": "false", "rows": [], "variant": variant}
    if mode == "row":
        return 200, {"ok": True, "rows": [{"id": 1}], "shape": "two_field", "variant": variant}
    return 200, {"ok": True, "rows": [{"id": 1}], "variant": variant}


class _V5Handler(BaseHTTPRequestHandler):
    server_version = "sift-sql-fixture-v5/1"

    def _serve(self, mode: str) -> None:
        variant = str(getattr(self.server, "fixture_variant", "indigo"))
        status, body = _response(mode, variant) if mode in V5_MODES and variant in V5_VARIANTS else (404, {"error": "not_found"})
        data = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
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


class SqlV5Server(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], variant: str) -> None:
        if int(address[1]) not in V5_PORTS or variant not in V5_VARIANTS:
            raise ValueError("SQL v5 target or variant is not allow-listed")
        super().__init__(address, _V5Handler)
        self.fixture_variant = variant


def make_sql_v5_fixture_server(*, port: int = 8830, variant: str = "indigo") -> SqlV5Server:
    return SqlV5Server(("127.0.0.1", int(port)), variant)


def sql_v5_source_sha256() -> str:
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
    return {"status_code": int(response.status_code), "body_length": len(body), "body_sha256": hashlib.sha256(body).hexdigest(), "json_shape": shape}


def collect_sql_v5(*, target: str, port: int, variant: str, method: str, mode: str, sample_id: str) -> dict[str, Any]:
    method = str(method).upper()
    if target.rstrip("/") != f"http://127.0.0.1:{port}" or port not in V5_PORTS or method not in {"GET", "POST"} or mode not in V5_MODES:
        raise ValueError("SQL v5 target/method/mode is not allow-listed")
    payload = build_detection_payload(target=target, method=method, path="/query", marker=f"pg197-{sample_id}", probe=_MODE_FRAGMENT[mode], probe_kind="sql_channel_class", form={"mode": mode} if method == "POST" else None)
    with httpx.Client(base_url=target, timeout=5.0, follow_redirects=False) as client:
        baseline = client.get("/query", params={"mode": "baseline"})
        response = client.get("/query", params={"mode": mode}) if method == "GET" else client.post("/query", data={"mode": mode})
    oracle = run_v5_oracle(mode, variant=variant)
    envelope = {"schema": V5_SCHEMA, "target": target, "method": method, "mode": mode, "variant": variant, "baseline": _summary(baseline), "response": _summary(response), "oracle": oracle, "fresh_target": True, "database_touched": False, "external_network": False}
    return {"schema_version": V5_SCHEMA, "sample_id": sample_id, "target": target, "method": method, "mode": mode, "variant": variant, "payload_sha256": payload["payload_sha256"], "response_projection": _summary(response), "oracle_projection": oracle, "evidence_hash": sha256_json(envelope), "fresh_target": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "database_touched": False, "external_network": False}


__all__ = ["V5_MODES", "V5_PORTS", "V5_VARIANTS", "collect_sql_v5", "make_sql_v5_fixture_server", "run_v5_oracle", "sql_v5_source_sha256"]
