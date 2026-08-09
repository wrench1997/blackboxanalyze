"""Safe local SQL-channel fixture backed by the abstract AST oracle.

The HTTP server never parses or executes SQL.  It maps allow-listed abstract
fragment classes to bounded JSON response shapes so the research loop can
exercise error, blind, row-shape, timeout, and local-side-channel oracles.
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
from .sql_ast_oracle import FRAGMENT_CLASSES, run_sql_ast_oracle


SQL_FIXTURE_BASE_URL = "http://127.0.0.1:8793"
SQL_FIXTURE_SCHEMA = "sift-sql-differential-fixture-v1"
SQL_FIXTURE_SPEC_SCHEMA = "sift-sql-differential-fixture-spec-v1"
SQL_FIXTURE_ORACLE = "synthetic_sql_ast_differential_v1"
SQL_FIXTURE_SOURCE_PATH = Path(__file__).resolve()
SQL_SAFE_MODES = frozenset({"plain", "quoted_value", "operator_like", "syntax_error", "blind_boolean", "row_shape", "time_delay", "local_side_channel"})
SQL_MARKER_RE = re.compile(r"^[A-Za-z0-9._-]{4,64}$")


def sql_fixture_source_sha256() -> str:
    return hashlib.sha256(SQL_FIXTURE_SOURCE_PATH.read_bytes()).hexdigest()


def _percent_encode_all(value: str) -> str:
    return "".join(f"%{byte:02X}" for byte in str(value).encode("utf-8"))


def _response_for_mode(mode: str) -> tuple[int, dict[str, Any]]:
    if mode == "syntax_error":
        return 422, {"status": "synthetic_parse_error", "error_class": "bounded_syntax_error", "rows": 0}
    if mode == "blind_boolean":
        return 200, {"status": "ok", "rows": 0, "row_shape": "false_branch"}
    if mode == "row_shape":
        return 200, {"status": "ok", "rows": 2, "row_shape": "expanded_shape"}
    if mode == "time_delay":
        return 200, {"status": "bounded_timeout", "timing_bucket": "over_budget", "timeout_budget_ms": 25, "simulated_latency_ms": 100}
    if mode == "local_side_channel":
        return 200, {"status": "ok", "local_callback": "sift-oob-marker", "external_network": False}
    if mode == "operator_like":
        return 200, {"status": "ok", "rows": 2, "query_shape": "boundary_shape"}
    return 200, {"status": "ok", "rows": 1, "query_shape": "parameterized"}


class SqlFixtureHandler(BaseHTTPRequestHandler):
    server_version = "sift-sql-fixture/1"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query, keep_blank_values=True)
        mode = unquote(str(query.get("mode", ["plain"])[0]))
        if parsed.path not in {"/query", "/plain"} or mode not in SQL_SAFE_MODES:
            status, payload = 404, {"status": "not_found"}
        else:
            status, payload = _response_for_mode(mode)
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def make_sql_fixture_server() -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", 8793), SqlFixtureHandler)


def validate_sql_fixture_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("SQL fixture spec must be an object")
    target = str(spec.get("target", SQL_FIXTURE_BASE_URL)).rstrip("/")
    if target != SQL_FIXTURE_BASE_URL:
        raise ValueError("SQL fixture target must be exactly http://127.0.0.1:8793")
    if str(spec.get("method", "GET")).upper() != "GET":
        raise ValueError("SQL fixture permits only read-only GET")
    path = str(spec.get("path", "/query"))
    if path not in {"/query", "/plain"}:
        raise ValueError("SQL fixture path is not allow-listed")
    raw_mode = str(spec.get("mode", "plain"))
    mode = unquote(raw_mode)
    if mode not in SQL_SAFE_MODES:
        raise ValueError("SQL fixture mode is not allow-listed")
    marker = str(spec.get("marker", "sql-pg09-marker"))
    if not SQL_MARKER_RE.fullmatch(marker):
        raise ValueError("SQL fixture marker must be an inert identifier")
    source_id = str(spec.get("source_id", ""))
    lab_id = str(spec.get("lab_id", ""))
    if not source_id or not lab_id:
        raise ValueError("SQL fixture source_id and lab_id are required")
    pair = dict(spec.get("pair") or {})
    if pair:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{4,96}", str(pair.get("pair_id", ""))):
            raise ValueError("SQL fixture pair_id is invalid")
        if str(pair.get("variant")) not in {"plain", "url_percent"}:
            raise ValueError("SQL fixture pair variant is invalid")
        if not isinstance(pair.get("encoding_depth"), int) or not 0 <= pair["encoding_depth"] <= 2:
            raise ValueError("SQL fixture encoding_depth is invalid")
    payload = build_detection_payload(
        target=SQL_FIXTURE_BASE_URL,
        method="GET",
        path=path,
        marker=marker,
        probe=mode,
        probe_kind="sql_channel_class",
        expected={},
    )
    return {
        "schema_version": SQL_FIXTURE_SPEC_SCHEMA,
        "target": SQL_FIXTURE_BASE_URL,
        "method": "GET",
        "path": path,
        "mode": raw_mode,
        "decoded_mode": mode,
        "marker": marker,
        "encoding": str(spec.get("encoding", "plain")),
        "source_id": source_id,
        "lab_id": lab_id,
        "family": "injection",
        "surface": "synthetic_sql_channel",
        "expected_oracle": SQL_FIXTURE_ORACLE,
        "expected_signal": str(spec.get("expected_signal", "bounded_ast_differential")),
        "payload": payload,
        **({"pair": pair} if pair else {}),
    }


def default_sql_fixture_specs(marker: str = "sql-pg09-marker") -> list[dict[str, Any]]:
    modes = (
        ("syntax_error", "bounded_syntax_error"),
        ("blind_boolean", "blind_response_differential"),
        ("row_shape", "row_shape_differential"),
        ("time_delay", "bounded_timeout"),
        ("local_side_channel", "local_callback_only"),
        ("operator_like", "ast_boundary"),
    )
    specs: list[dict[str, Any]] = []
    for index, (mode, signal) in enumerate(modes, start=1):
        pair_id = f"sql-pair-{index:02d}"
        plain = {
            "source_id": "fixture-pg09",
            "lab_id": f"{mode}-plain",
            "path": "/query",
            "mode": mode,
            "marker": marker,
            "encoding": "plain",
            "expected_signal": signal,
            "pair": {"pair_id": pair_id, "variant": "plain", "encoding_depth": 0},
        }
        encoded = dict(plain)
        encoded.update({
            "lab_id": f"{mode}-url-percent",
            "encoding": "url_percent",
        "mode": _percent_encode_all(mode),
            "pair": {"pair_id": pair_id, "variant": "url_percent", "encoding_depth": 1},
        })
        specs.extend([plain, encoded])
    safe_pair = {
        "source_id": "fixture-pg09",
        "lab_id": "quoted-value-plain",
        "path": "/query",
        "mode": "quoted_value",
        "marker": marker,
        "encoding": "plain",
        "expected_signal": "parameterized_value",
        "pair": {"pair_id": "sql-pair-safe-01", "variant": "plain", "encoding_depth": 0},
    }
    safe_encoded = dict(safe_pair)
    safe_encoded.update({
        "lab_id": "quoted-value-url-percent",
        "mode": _percent_encode_all("quoted_value"),
        "encoding": "url_percent",
        "pair": {"pair_id": "sql-pair-safe-01", "variant": "url_percent", "encoding_depth": 1},
    })
    specs.extend([safe_pair, safe_encoded])
    specs.append({
        "source_id": "fixture-pg09",
        "lab_id": "plain-control",
        "path": "/query",
        "mode": "plain",
        "marker": marker,
        "encoding": "plain",
        "expected_signal": "parameterized_baseline",
    })
    return specs


def _summary(response: httpx.Response) -> dict[str, Any]:
    body = response.content
    shape: dict[str, Any] = {}
    try:
        value = response.json()
        shape = {"key_count": len(value) if isinstance(value, dict) else 0, "type": "object" if isinstance(value, dict) else "other"}
    except (ValueError, json.JSONDecodeError):
        shape = {"key_count": 0, "type": "other"}
    return {
        "status_code": int(response.status_code),
        "headers": {"content-type": str(response.headers.get("content-type", ""))},
        "body_length": len(body),
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "json_shape": shape,
    }


class SqlDifferentialCollector:
    def __init__(self, *, target_instance_id: str = "unattested", source_hash: str | None = None) -> None:
        self.base_url = SQL_FIXTURE_BASE_URL
        self.target_instance_id = str(target_instance_id)[:128]
        self.source_hash = source_hash or sql_fixture_source_sha256()

    async def collect(self, raw_spec: dict[str, Any]) -> dict[str, Any]:
        spec = validate_sql_fixture_spec(raw_spec)
        async with httpx.AsyncClient(base_url=self.base_url, timeout=5.0, follow_redirects=False, cookies={}) as client:
            baseline = await client.get("/query", params={"mode": "plain"})
            response = await client.get(spec["path"], params={"mode": spec["mode"]}, headers=spec["payload"]["headers"])
        baseline_summary = _summary(baseline)
        response_summary = _summary(response)
        oracle = run_sql_ast_oracle(unquote(spec["mode"]))
        projection = dict(oracle.evidence)
        projection.update({
            "body_length_delta": response_summary["body_length"] - baseline_summary["body_length"],
            "body_length_delta_abs": abs(response_summary["body_length"] - baseline_summary["body_length"]),
            "status_changed": response.status_code != baseline.status_code,
        })
        reset = {
            "kind": "ephemeral_in_repo_sql_fixture",
            "fresh": True,
            "fresh_target": True,
            "state_change_allowed": False,
            "evaluator_state_hidden": True,
            "external_network": False,
            "target_instance_id": self.target_instance_id,
            "fixture_source_sha256": self.source_hash,
        }
        envelope = {
            "collector": SQL_FIXTURE_SCHEMA,
            "target": self.base_url,
            "path": spec["path"],
            "method": "GET",
            "reset": reset,
            "baseline": baseline_summary,
            "response": response_summary,
            "oracle_projection": projection,
            "local_http_loopback": True,
            "script_execution": False,
            "network_access": False,
            "navigation": False,
            "database_touched": False,
            "real_sleep_performed": False,
            "credentials_accessed": False,
            "encoding": spec["encoding"],
            "payload_sha256": spec["payload"]["payload_sha256"],
        }
        envelope["evidence_hash"] = sha256_json(envelope)
        checked = validate_evidence(envelope)
        positive = bool(projection.get("controlled_differential") and projection.get("interpreter_boundary"))
        record = {
            "schema_version": SQL_FIXTURE_SCHEMA,
            "sample_id": f"{spec['source_id']}-{spec['lab_id']}-{spec['payload']['payload_sha256'][:12]}",
            "source_id": spec["source_id"],
            "lab_id": spec["lab_id"],
            "family": "injection",
            "payload": spec["payload"],
            "probe_artifact": {"original": unquote(spec["mode"]), "encoding": spec["encoding"], "probe_sha256": hashlib.sha256(unquote(spec["mode"]).encode()).hexdigest()},
            "semantic": {"family": "injection", "surface": "synthetic_sql_channel", "expected_oracle": SQL_FIXTURE_ORACLE, "expected_signal": spec["expected_signal"]},
            "evaluator_state_visible": False,
            "replay": {"target": self.base_url, "method": "GET", "path": spec["path"], "params": {"mode": spec["mode"]}, "fresh_reset": reset, "transport": "httpx_loopback"},
            "response_projection": response_summary,
            "oracle_projection": projection,
            "evidence": checked["body"],
            "rule_ir_result": positive,
            "candidate_status": "suspicious_sql_channel" if positive else "clean_observation",
            "safety": {"local_only": True, "read_only": True, "fresh_reset": False, "fresh_target": True, "external_network": False, "script_execution": False, "database_touched": False, "real_sleep_performed": False, "raw_body_stored": False, "credentials_stored": False},
        }
        if spec.get("pair"):
            record["pair"] = dict(spec["pair"])
        return record

    async def collect_many(self, specs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for spec in specs:
            rows.append(await self.collect(spec))
        return rows


__all__ = [
    "SQL_FIXTURE_BASE_URL",
    "SQL_FIXTURE_ORACLE",
    "SqlDifferentialCollector",
    "default_sql_fixture_specs",
    "make_sql_fixture_server",
    "sql_fixture_source_sha256",
    "validate_sql_fixture_spec",
]
