"""Third independent, non-executing SQL-channel fixture for source holdout."""

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


SQL_V3_BASE_URL = "http://127.0.0.1:8809"
SQL_V3_PORTS = (8809, 8810, 8811)
SQL_V3_BASE_URLS = tuple(f"http://127.0.0.1:{port}" for port in SQL_V3_PORTS)
SQL_V3_SCHEMA = "sift-sql-differential-fixture-v3"
SQL_V3_SPEC_SCHEMA = "sift-sql-differential-fixture-spec-v3"
SQL_V3_ORACLE = "synthetic_sql_ast_differential_v1"
SQL_V3_SOURCE_PATH = Path(__file__).resolve()
SQL_V3_MODES = frozenset({"baseline", "literal_only", "parse_issue", "branch_check", "row_projection", "latency_bucket", "loopback_probe", "operator_join"})
_MARKER_RE = re.compile(r"^[A-Za-z0-9._-]{4,64}$")
_MODE_TO_FRAGMENT = {
    "baseline": "plain",
    "literal_only": "quoted_value",
    "parse_issue": "syntax_error",
    "branch_check": "blind_boolean",
    "row_projection": "row_shape",
    "latency_bucket": "time_delay",
    "loopback_probe": "local_side_channel",
    "operator_join": "operator_like",
}


def sql_v3_source_sha256() -> str:
    return hashlib.sha256(SQL_V3_SOURCE_PATH.read_bytes()).hexdigest()


def _percent_encode_all(value: str) -> str:
    return "".join(f"%{byte:02X}" for byte in str(value).encode("utf-8"))


def _response(mode: str, variant: str) -> tuple[int, str, str, dict[str, str]]:
    if mode == "parse_issue":
        return 400, json.dumps({"error": "parse", "class": "bounded", "variant": variant}), "application/problem+json", {"X-Sift-Protocol": "v3"}
    if mode == "branch_check":
        return 200, json.dumps({"ok": 1, "branch": "false", "rows": [], "variant": variant}), "application/json", {}
    if mode == "row_projection":
        return 200, json.dumps({"ok": 1, "projection": ["id", "label"], "rows": [{"id": 1}, {"id": 2}], "variant": variant}), "application/json", {}
    if mode == "latency_bucket":
        return 200, "latency_bucket=bounded;simulated_ms=100\n", "text/plain; charset=utf-8", {}
    if mode == "loopback_probe":
        return 200, json.dumps({"queue": "loopback", "external": False, "variant": variant}), "application/json", {"X-Sift-Callback": "local"}
    if mode == "operator_join":
        return 200, json.dumps({"ok": 1, "projection": "operator_join", "rows": [{"id": 1}, {"id": 2}], "variant": variant}), "application/json", {}
    return 200, json.dumps({"ok": 1, "projection": "literal", "rows": [{"id": 1}], "variant": variant}), "application/json", {}


class _SqlV3Handler(BaseHTTPRequestHandler):
    server_version = "sift-sql-fixture-v3/1"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query, keep_blank_values=True)
        mode = unquote(str(query.get("q", ["baseline"])[0]))
        variant = str(getattr(self.server, "fixture_variant", "alpha"))
        if parsed.path not in {"/search", "/health"} or mode not in SQL_V3_MODES or variant not in {"alpha", "beta", "gamma"}:
            status, body, content_type, extra = 404, "not found", "text/plain; charset=utf-8", {}
        elif parsed.path == "/health":
            status, body, content_type, extra = 200, json.dumps({"ready": True, "variant": variant}), "application/json", {}
        else:
            status, body, content_type, extra = _response(mode, variant)
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        for key, value in extra.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        return


class SqlV3Server(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], variant: str) -> None:
        if variant not in {"alpha", "beta", "gamma"}:
            raise ValueError("unknown SQL v3 variant")
        super().__init__(address, _SqlV3Handler)
        self.fixture_variant = variant


def make_sql_v3_fixture_server(*, port: int = 8809, variant: str = "alpha") -> SqlV3Server:
    if int(port) not in SQL_V3_PORTS:
        raise ValueError("SQL v3 port is not allow-listed")
    return SqlV3Server(("127.0.0.1", int(port)), variant)


def validate_sql_v3_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("SQL v3 spec must be an object")
    target = str(spec.get("target", SQL_V3_BASE_URL)).rstrip("/")
    if target not in SQL_V3_BASE_URLS:
        raise ValueError("SQL v3 target must be allow-listed loopback")
    if str(spec.get("method", "GET")).upper() != "GET":
        raise ValueError("SQL v3 permits GET only")
    path = str(spec.get("path", "/search"))
    if urlsplit(path).path not in {"/search", "/health"}:
        raise ValueError("SQL v3 path is not allow-listed")
    raw_mode = str(spec.get("mode", "baseline"))
    mode = unquote(raw_mode)
    if mode not in SQL_V3_MODES:
        raise ValueError("SQL v3 mode is not allow-listed")
    marker = str(spec.get("marker", "sql-v3-marker"))
    if not _MARKER_RE.fullmatch(marker):
        raise ValueError("SQL v3 marker must be inert")
    source_id, lab_id = str(spec.get("source_id", "")), str(spec.get("lab_id", ""))
    if not source_id or not lab_id:
        raise ValueError("SQL v3 provenance is required")
    pair = dict(spec.get("pair") or {})
    if pair and (not str(pair.get("pair_id", "")) or str(pair.get("variant")) not in {"plain", "url_percent"}):
        raise ValueError("SQL v3 pair metadata is invalid")
    payload = build_detection_payload(target=target, method="GET", path=path, marker=marker, probe=_MODE_TO_FRAGMENT[mode], probe_kind="sql_channel_class", expected={})
    return {
        "schema_version": SQL_V3_SPEC_SCHEMA,
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
        "surface": "synthetic_sql_channel_v3",
        "expected_oracle": SQL_V3_ORACLE,
        "expected_signal": str(spec.get("expected_signal", "bounded_ast_differential")),
        "payload": payload,
        **({"pair": pair} if pair else {}),
    }


def _path_for(mode: str, *, encoded: bool) -> str:
    query = _percent_encode_all(mode) if encoded else mode
    return f"/search?q={query}"


def default_sql_v3_specs(*, dataset_id: str = "fixture-pg15-v3", target: str = SQL_V3_BASE_URL, marker: str = "sql-pg15-marker") -> list[dict[str, Any]]:
    modes = (("parse_issue", "bounded_syntax_error"), ("branch_check", "blind_response_differential"), ("row_projection", "row_shape_differential"), ("latency_bucket", "bounded_timeout"), ("loopback_probe", "local_callback_only"), ("operator_join", "ast_boundary"))
    specs: list[dict[str, Any]] = []
    for index, (mode, signal) in enumerate(modes, start=1):
        pair_id = f"sql-pg15-{mode}"
        plain = {"source_id": dataset_id, "lab_id": f"{mode}-plain", "target": target, "path": _path_for(mode, encoded=False), "mode": mode, "marker": marker, "encoding": "plain", "expected_signal": signal, "pair": {"pair_id": pair_id, "variant": "plain", "encoding_depth": 0}}
        encoded = dict(plain)
        encoded.update({"lab_id": f"{mode}-url-percent", "path": _path_for(mode, encoded=True), "mode": _percent_encode_all(mode), "encoding": "url_percent", "pair": {"pair_id": pair_id, "variant": "url_percent", "encoding_depth": 1}})
        specs.extend([plain, encoded])
    safe = {"source_id": dataset_id, "lab_id": "literal-only-plain", "target": target, "path": _path_for("literal_only", encoded=False), "mode": "literal_only", "marker": marker, "encoding": "plain", "expected_signal": "parameterized_value", "pair": {"pair_id": "sql-pg15-safe", "variant": "plain", "encoding_depth": 0}}
    safe_encoded = dict(safe)
    safe_encoded.update({"lab_id": "literal-only-url-percent", "path": _path_for("literal_only", encoded=True), "mode": _percent_encode_all("literal_only"), "encoding": "url_percent", "pair": {"pair_id": "sql-pg15-safe", "variant": "url_percent", "encoding_depth": 1}})
    specs.extend([safe, safe_encoded])
    specs.append({"source_id": dataset_id, "lab_id": "baseline-control", "target": target, "path": _path_for("baseline", encoded=False), "mode": "baseline", "marker": marker, "encoding": "plain", "expected_signal": "parameterized_baseline"})
    return specs


def _summary(response: httpx.Response) -> dict[str, Any]:
    body = response.content
    shape = {"key_count": 0, "type": "other", "scalar_count": 0}
    try:
        value = response.json()
        if isinstance(value, dict):
            shape = {"key_count": len(value), "type": "object", "scalar_count": sum(not isinstance(item, (dict, list)) for item in value.values())}
    except (ValueError, json.JSONDecodeError):
        pass
    return {"status_code": int(response.status_code), "headers": {"content-type": str(response.headers.get("content-type", ""))}, "body_length": len(body), "body_sha256": hashlib.sha256(body).hexdigest(), "json_shape": shape}


class SqlV3Collector:
    def __init__(self, *, base_url: str = SQL_V3_BASE_URL, target_instance_id: str = "unattested", source_hash: str | None = None) -> None:
        if base_url.rstrip("/") not in SQL_V3_BASE_URLS:
            raise ValueError("SQL v3 collector target is not allow-listed")
        self.base_url = base_url.rstrip("/")
        self.target_instance_id = str(target_instance_id)[:128]
        self.source_hash = source_hash or sql_v3_source_sha256()

    async def collect(self, raw_spec: dict[str, Any]) -> dict[str, Any]:
        spec = validate_sql_v3_spec(raw_spec)
        async with httpx.AsyncClient(base_url=self.base_url, timeout=5.0, follow_redirects=False, cookies={}) as client:
            baseline = await client.get("/search?q=baseline")
            response = await client.get(spec["path"], headers=spec["payload"]["headers"])
        baseline_summary, response_summary = _summary(baseline), _summary(response)
        canonical = _MODE_TO_FRAGMENT[spec["decoded_mode"]]
        oracle = run_sql_ast_oracle(canonical)
        projection = dict(oracle.evidence)
        projection.update({"body_length_delta_abs": abs(response_summary["body_length"] - baseline_summary["body_length"]), "status_changed": response.status_code != baseline.status_code, "fixture_mode": spec["decoded_mode"]})
        reset = {"kind": "ephemeral_in_repo_sql_v3_fixture", "fresh": True, "fresh_target": True, "state_change_allowed": False, "evaluator_state_hidden": True, "external_network": False, "target_instance_id": self.target_instance_id, "fixture_source_sha256": self.source_hash}
        envelope = {"collector": SQL_V3_SCHEMA, "target": self.base_url, "path": spec["path"], "method": "GET", "reset": reset, "baseline": baseline_summary, "response": response_summary, "oracle_projection": projection, "local_http_loopback": True, "script_execution": False, "network_access": False, "navigation": False, "database_touched": False, "real_sleep_performed": False, "credentials_accessed": False, "encoding": spec["encoding"], "payload_sha256": spec["payload"]["payload_sha256"]}
        envelope["evidence_hash"] = sha256_json(envelope)
        checked = validate_evidence(envelope)
        positive = bool(projection.get("controlled_differential") and projection.get("interpreter_boundary"))
        record = {"schema_version": SQL_V3_SCHEMA, "sample_id": f"{spec['source_id']}-{spec['lab_id']}-{spec['payload']['payload_sha256'][:12]}", "source_id": spec["source_id"], "lab_id": spec["lab_id"], "family": "injection", "payload": spec["payload"], "probe_artifact": {"original": canonical, "surface_probe": spec["decoded_mode"], "encoding": spec["encoding"], "probe_sha256": hashlib.sha256(canonical.encode()).hexdigest()}, "semantic": {"family": "injection", "surface": "synthetic_sql_channel_v3", "expected_oracle": SQL_V3_ORACLE, "expected_signal": spec["expected_signal"], "canonical_fragment_class": canonical}, "evaluator_state_visible": False, "replay": {"target": self.base_url, "method": "GET", "path": spec["path"], "fresh_reset": reset, "transport": "httpx_loopback"}, "response_projection": response_summary, "oracle_projection": projection, "evidence": checked["body"], "rule_ir_result": positive, "candidate_status": "suspicious_sql_channel" if positive else "clean_observation", "safety": {"local_only": True, "read_only": True, "fresh_reset": True, "fresh_target": True, "external_network": False, "script_execution": False, "database_touched": False, "real_sleep_performed": False, "raw_body_stored": False, "credentials_stored": False, "state_mutated": False}}
        if spec.get("pair"):
            record["pair"] = dict(spec["pair"])
        return record

    async def collect_many(self, specs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return [await self.collect(spec) for spec in specs]


__all__ = ["SQL_V3_BASE_URL", "SQL_V3_BASE_URLS", "SQL_V3_ORACLE", "SQL_V3_PORTS", "SqlV3Collector", "default_sql_v3_specs", "make_sql_v3_fixture_server", "sql_v3_source_sha256", "validate_sql_v3_spec"]
