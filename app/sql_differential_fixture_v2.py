"""Independent safe SQL-channel fixture with a different transport surface.

The server maps renamed abstract channels to bounded JSON observations and
never parses, executes, sleeps, or sends SQL.  The collector binds those
observations to the shared AST differential oracle so transfer can be tested
without turning this research track into a real-target scanner.
"""

from __future__ import annotations

import hashlib
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, unquote, urlsplit

import httpx

from .detection_payload import build_detection_payload
from .maze_engine import sha256_json, validate_evidence
from .sql_ast_oracle import run_sql_ast_oracle


SQL_V2_BASE_URL = "http://127.0.0.1:8806"
SQL_V2_BASE_URLS = tuple(f"http://127.0.0.1:{port}" for port in (8806, 8807, 8808))
SQL_V2_PORTS = (8806, 8807, 8808)
SQL_V2_SCHEMA = "sift-sql-differential-fixture-v2"
SQL_V2_SPEC_SCHEMA = "sift-sql-differential-fixture-spec-v2"
SQL_V2_ORACLE = "synthetic_sql_ast_differential_v1"
SQL_V2_SOURCE_PATH = Path(__file__).resolve()
SQL_V2_MODES = frozenset({"baseline", "value_only", "parse_fault", "boolean_flip", "shape_expand", "bounded_wait", "local_callback", "operator_edge"})
_MARKER_RE = re.compile(r"^[A-Za-z0-9._-]{4,64}$")
_MODE_TO_FRAGMENT = {
    "baseline": "plain",
    "value_only": "quoted_value",
    "parse_fault": "syntax_error",
    "boolean_flip": "blind_boolean",
    "shape_expand": "row_shape",
    "bounded_wait": "time_delay",
    "local_callback": "local_side_channel",
    "operator_edge": "operator_like",
}


def sql_v2_source_sha256() -> str:
    return hashlib.sha256(SQL_V2_SOURCE_PATH.read_bytes()).hexdigest()


def _percent_encode_all(value: str) -> str:
    return "".join(f"%{byte:02X}" for byte in str(value).encode("utf-8"))


def _response_for_mode(mode: str, variant: str) -> tuple[int, dict[str, Any]]:
    if mode == "parse_fault":
        return 409, {"state": "parse_fault", "diagnostic": "bounded", "count": 0, "variant": variant}
    if mode == "boolean_flip":
        return 200, {"ok": True, "rows": [], "branch": "false", "variant": variant}
    if mode == "shape_expand":
        return 200, {"ok": True, "rows": [{"id": 1}, {"id": 2}], "projection": "wide", "variant": variant}
    if mode == "bounded_wait":
        return 200, {"state": "timeout_bucket", "budget": 25, "simulated": 100, "variant": variant}
    if mode == "local_callback":
        return 200, {"ok": True, "queue": "loopback-marker", "external": False, "variant": variant}
    if mode == "operator_edge":
        return 200, {"ok": True, "rows": [{"id": 1}, {"id": 2}], "projection": "operator", "variant": variant}
    return 200, {"ok": True, "rows": [{"id": 1}], "projection": "parameter", "variant": variant}


class _SqlV2Handler(BaseHTTPRequestHandler):
    server_version = "sift-sql-fixture-v2/1"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query, keep_blank_values=True)
        mode = unquote(str(query.get("channel", ["baseline"])[0]))
        variant = str(getattr(self.server, "fixture_variant", "alpha"))
        if parsed.path not in {"/lookup", "/health"} or mode not in SQL_V2_MODES or variant not in {"alpha", "beta", "gamma"}:
            status, payload = 404, {"state": "not_found"}
        elif parsed.path == "/health":
            status, payload = 200, {"state": "ready", "variant": variant}
        else:
            status, payload = _response_for_mode(mode, variant)
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("X-Sift-Channel", variant)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


class SqlV2Server(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], variant: str) -> None:
        if variant not in {"alpha", "beta", "gamma"}:
            raise ValueError("unknown SQL v2 fixture variant")
        super().__init__(address, _SqlV2Handler)
        self.fixture_variant = variant


def make_sql_v2_fixture_server(*, port: int = 8806, variant: str = "alpha") -> SqlV2Server:
    if int(port) not in SQL_V2_PORTS:
        raise ValueError("SQL v2 fixture port is not allow-listed")
    return SqlV2Server(("127.0.0.1", int(port)), variant)


def validate_sql_v2_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("SQL v2 fixture spec must be an object")
    target = str(spec.get("target", SQL_V2_BASE_URL)).rstrip("/")
    if target not in SQL_V2_BASE_URLS:
        raise ValueError("SQL v2 target must be an allow-listed loopback URL")
    if str(spec.get("method", "GET")).upper() != "GET":
        raise ValueError("SQL v2 fixture permits only GET")
    path = str(spec.get("path", "/lookup"))
    if urlsplit(path).path not in {"/lookup", "/health"}:
        raise ValueError("SQL v2 path is not allow-listed")
    raw_mode = str(spec.get("mode", "baseline"))
    mode = unquote(raw_mode)
    if mode not in SQL_V2_MODES:
        raise ValueError("SQL v2 channel is not allow-listed")
    marker = str(spec.get("marker", "sql-v2-marker"))
    if not _MARKER_RE.fullmatch(marker):
        raise ValueError("SQL v2 marker must be an inert identifier")
    source_id, lab_id = str(spec.get("source_id", "")), str(spec.get("lab_id", ""))
    if not source_id or not lab_id:
        raise ValueError("SQL v2 provenance fields are required")
    pair = dict(spec.get("pair") or {})
    if pair:
        if not str(pair.get("pair_id", "")) or str(pair.get("variant")) not in {"plain", "url_percent"}:
            raise ValueError("SQL v2 pair metadata is invalid")
        if not isinstance(pair.get("encoding_depth"), int) or not 0 <= pair["encoding_depth"] <= 2:
            raise ValueError("SQL v2 encoding depth is invalid")
    payload = build_detection_payload(target=target, method="GET", path=path, marker=marker, probe=_MODE_TO_FRAGMENT[mode], probe_kind="sql_channel_class", expected={})
    return {
        "schema_version": SQL_V2_SPEC_SCHEMA,
        "target": target,
        "method": "GET",
        "path": path,
        "mode": raw_mode,
        "decoded_mode": mode,
        "marker": marker,
        "encoding": str(spec.get("encoding", "plain")),
        "source_id": source_id,
        "lab_id": lab_id,
        "family": "injection",
        "surface": "synthetic_sql_channel_v2",
        "expected_oracle": SQL_V2_ORACLE,
        "expected_signal": str(spec.get("expected_signal", "bounded_ast_differential")),
        "payload": payload,
        **({"pair": pair} if pair else {}),
    }


def _path_for(mode: str, *, encoded: bool) -> str:
    channel = _percent_encode_all(mode) if encoded else mode
    return f"/lookup?channel={channel}"


def default_sql_v2_specs(*, dataset_id: str = "fixture-pg14-v2", target: str = SQL_V2_BASE_URL, marker: str = "sql-pg14-marker") -> list[dict[str, Any]]:
    modes = (
        ("parse_fault", "bounded_syntax_error"),
        ("boolean_flip", "blind_response_differential"),
        ("shape_expand", "row_shape_differential"),
        ("bounded_wait", "bounded_timeout"),
        ("local_callback", "local_callback_only"),
        ("operator_edge", "ast_boundary"),
    )
    specs: list[dict[str, Any]] = []
    for index, (mode, signal) in enumerate(modes, start=1):
        pair_id = f"sql-pg14-{mode}"
        plain = {
            "source_id": dataset_id,
            "lab_id": f"{mode}-plain",
            "target": target,
            "path": _path_for(mode, encoded=False),
            "mode": mode,
            "marker": marker,
            "encoding": "plain",
            "expected_signal": signal,
            "pair": {"pair_id": pair_id, "variant": "plain", "encoding_depth": 0},
        }
        encoded = dict(plain)
        encoded.update({
            "lab_id": f"{mode}-url-percent",
            "path": _path_for(mode, encoded=True),
            "mode": _percent_encode_all(mode),
            "encoding": "url_percent",
            "pair": {"pair_id": pair_id, "variant": "url_percent", "encoding_depth": 1},
        })
        specs.extend([plain, encoded])
    safe = {
        "source_id": dataset_id,
        "lab_id": "value-only-plain",
        "target": target,
        "path": _path_for("value_only", encoded=False),
        "mode": "value_only",
        "marker": marker,
        "encoding": "plain",
        "expected_signal": "parameterized_value",
        "pair": {"pair_id": "sql-pg14-safe", "variant": "plain", "encoding_depth": 0},
    }
    safe_encoded = dict(safe)
    safe_encoded.update({"lab_id": "value-only-url-percent", "path": _path_for("value_only", encoded=True), "mode": _percent_encode_all("value_only"), "encoding": "url_percent", "pair": {"pair_id": "sql-pg14-safe", "variant": "url_percent", "encoding_depth": 1}})
    specs.extend([safe, safe_encoded])
    specs.append({"source_id": dataset_id, "lab_id": "baseline-control", "target": target, "path": _path_for("baseline", encoded=False), "mode": "baseline", "marker": marker, "encoding": "plain", "expected_signal": "parameterized_baseline"})
    return specs


def _summary(response: httpx.Response) -> dict[str, Any]:
    body = response.content
    shape: dict[str, Any] = {"key_count": 0, "type": "other", "scalar_count": 0}
    try:
        value = response.json()
        if isinstance(value, dict):
            shape = {"key_count": len(value), "type": "object", "scalar_count": sum(not isinstance(item, (dict, list)) for item in value.values())}
    except (ValueError, json.JSONDecodeError):
        pass
    return {"status_code": int(response.status_code), "headers": {"content-type": str(response.headers.get("content-type", ""))}, "body_length": len(body), "body_sha256": hashlib.sha256(body).hexdigest(), "json_shape": shape}


class SqlV2Collector:
    def __init__(self, *, base_url: str = SQL_V2_BASE_URL, target_instance_id: str = "unattested", source_hash: str | None = None) -> None:
        if base_url.rstrip("/") not in SQL_V2_BASE_URLS:
            raise ValueError("SQL v2 collector target is not allow-listed")
        self.base_url = base_url.rstrip("/")
        self.target_instance_id = str(target_instance_id)[:128]
        self.source_hash = source_hash or sql_v2_source_sha256()

    async def collect(self, raw_spec: dict[str, Any]) -> dict[str, Any]:
        spec = validate_sql_v2_spec(raw_spec)
        async with httpx.AsyncClient(base_url=self.base_url, timeout=5.0, follow_redirects=False, cookies={}) as client:
            baseline = await client.get("/lookup?channel=baseline")
            response = await client.get(spec["path"], headers=spec["payload"]["headers"])
        baseline_summary, response_summary = _summary(baseline), _summary(response)
        canonical = _MODE_TO_FRAGMENT[spec["decoded_mode"]]
        oracle = run_sql_ast_oracle(canonical)
        projection = dict(oracle.evidence)
        projection.update({"body_length_delta": response_summary["body_length"] - baseline_summary["body_length"], "body_length_delta_abs": abs(response_summary["body_length"] - baseline_summary["body_length"]), "status_changed": response.status_code != baseline.status_code, "fixture_mode": spec["decoded_mode"]})
        reset = {"kind": "ephemeral_in_repo_sql_v2_fixture", "fresh": True, "fresh_target": True, "state_change_allowed": False, "evaluator_state_hidden": True, "external_network": False, "target_instance_id": self.target_instance_id, "fixture_source_sha256": self.source_hash}
        envelope = {"collector": SQL_V2_SCHEMA, "target": self.base_url, "path": spec["path"], "method": "GET", "reset": reset, "baseline": baseline_summary, "response": response_summary, "oracle_projection": projection, "local_http_loopback": True, "script_execution": False, "network_access": False, "navigation": False, "database_touched": False, "real_sleep_performed": False, "credentials_accessed": False, "encoding": spec["encoding"], "payload_sha256": spec["payload"]["payload_sha256"]}
        envelope["evidence_hash"] = sha256_json(envelope)
        checked = validate_evidence(envelope)
        positive = bool(projection.get("controlled_differential") and projection.get("interpreter_boundary"))
        record = {
            "schema_version": SQL_V2_SCHEMA,
            "sample_id": f"{spec['source_id']}-{spec['lab_id']}-{spec['payload']['payload_sha256'][:12]}",
            "source_id": spec["source_id"],
            "lab_id": spec["lab_id"],
            "family": "injection",
            "payload": spec["payload"],
            "probe_artifact": {"original": canonical, "surface_probe": spec["decoded_mode"], "encoding": spec["encoding"], "probe_sha256": hashlib.sha256(canonical.encode()).hexdigest()},
            "semantic": {"family": "injection", "surface": "synthetic_sql_channel_v2", "expected_oracle": SQL_V2_ORACLE, "expected_signal": spec["expected_signal"], "canonical_fragment_class": canonical},
            "evaluator_state_visible": False,
            "replay": {"target": self.base_url, "method": "GET", "path": spec["path"], "fresh_reset": reset, "transport": "httpx_loopback"},
            "response_projection": response_summary,
            "oracle_projection": projection,
            "evidence": checked["body"],
            "rule_ir_result": positive,
            "candidate_status": "suspicious_sql_channel" if positive else "clean_observation",
            "safety": {"local_only": True, "read_only": True, "fresh_reset": True, "fresh_target": True, "external_network": False, "script_execution": False, "database_touched": False, "real_sleep_performed": False, "raw_body_stored": False, "credentials_stored": False, "state_mutated": False},
        }
        if spec.get("pair"):
            record["pair"] = dict(spec["pair"])
        return record

    async def collect_many(self, specs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return [await self.collect(spec) for spec in specs]


__all__ = ["SQL_V2_BASE_URL", "SQL_V2_BASE_URLS", "SQL_V2_ORACLE", "SQL_V2_PORTS", "SqlV2Collector", "default_sql_v2_specs", "make_sql_v2_fixture_server", "sql_v2_source_sha256", "validate_sql_v2_spec"]
