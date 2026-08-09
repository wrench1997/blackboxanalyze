"""Multi-surface, inert local fixture for PG-PK-07.

Every endpoint is read-only and escapes the marker before reflecting it.  The
experiment is about separating response surfaces (HTML attribute, HTML text,
JSON, and a response header), not about executing a vulnerability payload.
Only the attribute surface satisfies the positive oracle used by this track.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, unquote, urlsplit

import httpx

from .detection_payload import build_detection_payload
from .maze_engine import sha256_json, validate_evidence
from .surface_sink_oracle import observe_surface_sink


SURFACE_FIXTURE_BASE_URL = "http://127.0.0.1:8791"
SURFACE_FIXTURE_SCHEMA = "sift-cross-app-surface-fixture-v1"
SURFACE_FIXTURE_SPEC_SCHEMA = "sift-cross-app-surface-fixture-spec-v1"
SURFACE_FIXTURE_ORACLE = "fixture_surface_specific_projection_v1"
SURFACE_FIXTURE_SOURCE_PATH = Path(__file__).resolve()
SURFACE_SAFE_PATHS = frozenset({"/attribute", "/text", "/json", "/header", "/plain"})
SURFACE_MARKER_RE = re.compile(r"^[A-Za-z0-9._-]{4,64}$")


def surface_fixture_source_sha256() -> str:
    return hashlib.sha256(SURFACE_FIXTURE_SOURCE_PATH.read_bytes()).hexdigest()


def _percent_encode_all(value: str) -> str:
    return "".join(f"%{byte:02X}" for byte in str(value).encode("utf-8"))


class SurfaceFixtureHandler(BaseHTTPRequestHandler):
    server_version = "sift-surface-fixture/1"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query, keep_blank_values=True)
        raw = query.get("message", [""])[0]
        message = unquote(str(raw))
        extra_headers: dict[str, str] = {}
        content_type = "text/html; charset=utf-8"
        if parsed.path == "/attribute":
            body = (
                "<!doctype html><html><body>"
                f'<div data-sift-marker="{html.escape(message, quote=True)}">attribute echo</div>'
                "</body></html>"
            ).encode("utf-8")
            status = 200
        elif parsed.path == "/text":
            body = (
                "<!doctype html><html><body>"
                f"<p>{html.escape(message, quote=True)}</p>text echo"
                "</body></html>"
            ).encode("utf-8")
            status = 200
        elif parsed.path == "/json":
            body = json.dumps({"message": message, "ok": True}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            content_type = "application/json"
            status = 200
        elif parsed.path == "/header":
            body = b"<!doctype html><html><body><p>header echo</p></body></html>"
            extra_headers["X-Sift-Echo"] = message
            status = 200
        elif parsed.path == "/plain":
            body = b"<!doctype html><html><body><p>plain control</p></body></html>"
            status = 200
        else:
            body = b"not found"
            status = 404
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        for key, value in extra_headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def make_surface_fixture_server() -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", 8791), SurfaceFixtureHandler)


def validate_surface_fixture_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("surface fixture spec must be an object")
    target = str(spec.get("target", SURFACE_FIXTURE_BASE_URL)).rstrip("/")
    if target != SURFACE_FIXTURE_BASE_URL:
        raise ValueError("surface fixture target must be exactly http://127.0.0.1:8791")
    if str(spec.get("method", "GET")).upper() != "GET":
        raise ValueError("surface fixture permits only read-only GET")
    path = str(spec.get("path", ""))
    if path not in SURFACE_SAFE_PATHS:
        raise ValueError("surface fixture path is not allow-listed")
    params = dict(spec.get("params") or {})
    if set(params) - {"message"} or len(params) > 1:
        raise ValueError("surface fixture only permits the message query field")
    marker = str(spec.get("marker", ""))
    if not SURFACE_MARKER_RE.fullmatch(marker):
        raise ValueError("surface fixture marker must be an inert identifier")
    source_id = str(spec.get("source_id", ""))
    lab_id = str(spec.get("lab_id", ""))
    surface_role = str(spec.get("surface_role", ""))
    if not source_id or not lab_id or not re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", surface_role):
        raise ValueError("surface fixture provenance fields are invalid")
    pair = dict(spec.get("pair") or {})
    if pair:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{4,96}", str(pair.get("pair_id", ""))):
            raise ValueError("surface fixture pair_id is invalid")
        if str(pair.get("variant")) not in {"plain", "url_percent"}:
            raise ValueError("surface fixture pair variant is invalid")
        if str(pair.get("surface_role")) != surface_role:
            raise ValueError("surface fixture pair surface role must match semantic role")
        if not isinstance(pair.get("encoding_depth"), int) or not 0 <= pair["encoding_depth"] <= 2:
            raise ValueError("surface fixture encoding_depth is invalid")
    probe = str(spec.get("probe", f'<span data-sift-marker="{marker}">x</span>'))
    payload = build_detection_payload(
        target=SURFACE_FIXTURE_BASE_URL,
        method="GET",
        path=path,
        marker=marker,
        probe=probe,
        probe_kind="inert_dom_markup",
        expected={},
    )
    return {
        "schema_version": SURFACE_FIXTURE_SPEC_SCHEMA,
        "target": SURFACE_FIXTURE_BASE_URL,
        "method": "GET",
        "path": path,
        "params": {str(key): str(value) for key, value in params.items()},
        "marker": marker,
        "probe": probe,
        "encoding": str(spec.get("encoding", "plain")),
        "source_id": source_id,
        "lab_id": lab_id,
        "family": "xss",
        "surface": str(spec.get("surface", f"fixture_{surface_role}")),
        "surface_role": surface_role,
        "expected_oracle": SURFACE_FIXTURE_ORACLE,
        "expected_signal": str(spec.get("expected_signal", "surface_specific_signal")),
        "payload": payload,
        **({"pair": pair} if pair else {}),
    }


def default_surface_fixture_specs(marker_prefix: str = "fx-pg07") -> list[dict[str, Any]]:
    marker = f"{marker_prefix}-marker"
    probe = f'<span data-sift-marker="{marker}">x</span>'
    surfaces = (
        ("attribute", "/attribute", "reflected_attribute", "marker_in_attribute"),
        ("text", "/text", "reflected_text", "marker_in_html_text"),
        ("json", "/json", "json_echo", "marker_in_json_value"),
        ("header", "/header", "header_echo", "marker_in_header"),
    )
    specs: list[dict[str, Any]] = []
    for index, (name, path, role, signal) in enumerate(surfaces, start=1):
        pair_id = f"surface-pair-{index:02d}"
        plain = {
            "source_id": "fixture-pg07",
            "lab_id": f"{name}-plain",
            "path": path,
            "params": {"message": marker},
            "marker": marker,
            "probe": probe,
            "encoding": "plain",
            "surface": f"fixture_{role}",
            "surface_role": role,
            "expected_signal": signal,
            "pair": {"pair_id": pair_id, "variant": "plain", "surface_role": role, "encoding_depth": 0},
        }
        encoded = dict(plain)
        encoded.update({
            "lab_id": f"{name}-url-percent",
            "params": {"message": _percent_encode_all(marker)},
            "probe": _percent_encode_all(probe),
            "encoding": "url_percent",
            "pair": {"pair_id": pair_id, "variant": "url_percent", "surface_role": role, "encoding_depth": 1},
        })
        specs.extend([plain, encoded])
    specs.append({
        "source_id": "fixture-pg07",
        "lab_id": "plain-control",
        "path": "/plain",
        "params": {"message": marker},
        "marker": marker,
        "probe": probe,
        "encoding": "plain",
        "surface": "fixture_plain_control",
        "surface_role": "plain_control",
        "expected_signal": "no_surface_signal",
    })
    return specs


def _response_summary(response: httpx.Response) -> dict[str, Any]:
    body = response.content
    return {
        "status_code": int(response.status_code),
        "headers": {"content-type": str(response.headers.get("content-type", ""))},
        "body_length": len(body),
        "body_sha256": hashlib.sha256(body).hexdigest(),
    }


class _SurfaceShapeParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags = 0
        self.attributes = 0
        self.scripts = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags += 1
        self.attributes += len(attrs)
        self.scripts += int(tag.casefold() == "script")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)


def _surface_shape(response: httpx.Response, baseline: httpx.Response) -> dict[str, Any]:
    """Return generic response geometry, deliberately omitting oracle fields."""

    content_type = str(response.headers.get("content-type", "")).casefold()
    body = response.content.decode("utf-8", errors="replace")
    parser = _SurfaceShapeParser()
    parser.feed(body)
    json_field_count = 0
    if content_type.startswith("application/json"):
        try:
            parsed = response.json()
            json_field_count = len(parsed) if isinstance(parsed, dict) else 0
        except (ValueError, json.JSONDecodeError):
            json_field_count = 0
    status = int(response.status_code)
    status_class = f"{status // 100}xx" if 100 <= status < 600 else "other"
    return {
        "content_type_class": "json" if content_type.startswith("application/json") else "html" if content_type.startswith("text/html") else "other",
        "status_class": status_class,
        "html_tag_count": min(parser.tags, 64),
        "html_attribute_count": min(parser.attributes, 64),
        "script_count": min(parser.scripts, 32),
        "json_field_count": min(json_field_count, 32),
        "response_header_count": min(len(response.headers), 32),
        "body_length": min(len(response.content), 4096),
        "body_length_delta_abs": min(abs(len(response.content) - len(baseline.content)), 4096),
    }


def _projection(response: httpx.Response, baseline: httpx.Response, marker: str) -> dict[str, Any]:
    sink = observe_surface_sink(
        response.content,
        marker=marker,
        content_type=str(response.headers.get("content-type", "")),
        headers=dict(response.headers),
    )
    return {
        **sink,
        "sql_error_shape": False,
        "external_redirect": False,
        "redirect_present": bool(response.history),
        "status_changed": response.status_code != baseline.status_code,
        "body_length_delta": len(response.content) - len(baseline.content),
        "body_length_delta_abs": abs(len(response.content) - len(baseline.content)),
    }


class SurfaceFixtureCollector:
    def __init__(self, *, target_instance_id: str = "unattested", source_hash: str | None = None) -> None:
        self.base_url = SURFACE_FIXTURE_BASE_URL
        self.target_instance_id = str(target_instance_id)[:128]
        self.source_hash = source_hash or surface_fixture_source_sha256()

    async def collect(self, raw_spec: dict[str, Any]) -> dict[str, Any]:
        spec = validate_surface_fixture_spec(raw_spec)
        async with httpx.AsyncClient(base_url=self.base_url, timeout=5.0, follow_redirects=False, cookies={}) as client:
            baseline = await client.get(spec["path"])
            response = await client.get(spec["path"], params=spec["params"], headers=spec["payload"]["headers"])
        baseline_summary = _response_summary(baseline)
        response_summary = _response_summary(response)
        surface_shape = _surface_shape(response, baseline)
        projection = _projection(response, baseline, spec["marker"])
        reset = {
            "kind": "ephemeral_in_repo_surface_fixture",
            "fresh": True,
            "fresh_target": True,
            "state_change_allowed": False,
            "evaluator_state_hidden": True,
            "external_network": False,
            "target_instance_id": self.target_instance_id,
            "fixture_source_sha256": self.source_hash,
        }
        envelope = {
            "collector": SURFACE_FIXTURE_SCHEMA,
            "target": self.base_url,
            "path": spec["path"],
            "method": "GET",
            "reset": reset,
            "baseline": baseline_summary,
            "response": response_summary,
            "surface_shape": surface_shape,
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
        positive = bool(projection["marker_in_attribute"] and spec["surface_role"] == "reflected_attribute")
        record = {
            "schema_version": SURFACE_FIXTURE_SCHEMA,
            "sample_id": f"{spec['source_id']}-{spec['lab_id']}-{spec['payload']['payload_sha256'][:12]}",
            "source_id": spec["source_id"],
            "lab_id": spec["lab_id"],
            "family": "xss",
            "payload": spec["payload"],
            "probe_artifact": {
                "original": spec["probe"],
                "encoding": spec["encoding"],
                "probe_sha256": hashlib.sha256(spec["probe"].encode("utf-8")).hexdigest(),
            },
            "semantic": {
                "family": "xss",
                "surface": spec["surface"],
                "surface_role": spec["surface_role"],
                "expected_oracle": SURFACE_FIXTURE_ORACLE,
                "expected_signal": spec["expected_signal"],
            },
            "evaluator_state_visible": False,
            "replay": {
                "target": self.base_url,
                "method": "GET",
                "path": spec["path"],
                "params": spec["params"],
                "fresh_reset": reset,
                "transport": "httpx_loopback",
            },
            "response_projection": response_summary,
            "surface_shape": surface_shape,
            "oracle_projection": projection,
            "evidence": checked["body"],
            "rule_ir_result": positive,
            "candidate_status": "suspicious_surface_signal" if positive else "clean_observation",
            "safety": {
                "local_only": True,
                "read_only": True,
                "fresh_reset": False,
                "fresh_target": True,
                "external_network": False,
                "script_execution": False,
                "database_touched": False,
                "real_sleep_performed": False,
                "raw_body_stored": False,
                "credentials_stored": False,
            },
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
    "SURFACE_FIXTURE_BASE_URL",
    "SURFACE_FIXTURE_ORACLE",
    "SurfaceFixtureCollector",
    "default_surface_fixture_specs",
    "make_surface_fixture_server",
    "surface_fixture_source_sha256",
    "validate_surface_fixture_spec",
]
