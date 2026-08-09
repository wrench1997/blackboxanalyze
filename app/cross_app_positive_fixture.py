"""Ephemeral, in-repository positive oracle fixture for PG-PK-06.

The fixture is deliberately inert: it reflects a canary into an HTML
attribute, never executes markup, never writes state, and has no network path
other than the loopback listener started by the experiment script.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, unquote, urlsplit

import httpx

from .detection_payload import build_detection_payload
from .maze_engine import sha256_json, validate_evidence
from .surface_sink_oracle import observe_surface_sink


FIXTURE_BASE_URL = "http://127.0.0.1:8790"
FIXTURE_SCHEMA = "sift-cross-app-positive-fixture-v1"
FIXTURE_SPEC_SCHEMA = "sift-cross-app-positive-fixture-spec-v1"
FIXTURE_ORACLE = "fixture_inert_attribute_oracle_v1"
FIXTURE_SOURCE_PATH = Path(__file__).resolve()
MARKER_RE = re.compile(r"^[A-Za-z0-9._-]{4,64}$")
SAFE_PATHS = frozenset({"/reflect", "/plain"})


def fixture_source_sha256() -> str:
    return hashlib.sha256(FIXTURE_SOURCE_PATH.read_bytes()).hexdigest()


class _FixtureHTMLShapeParser(HTMLParser):
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


class FixtureHandler(BaseHTTPRequestHandler):
    server_version = "sift-fixture/1"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query, keep_blank_values=True)
        if parsed.path == "/reflect":
            raw = query.get("message", [""])[0]
            message = unquote(str(raw))
            body = (
                "<!doctype html><html><body>"
                f'<div data-sift-marker="{html.escape(message, quote=True)}">safe inert reflection</div>'
                "</body></html>"
            ).encode("utf-8")
            status = 200
        elif parsed.path == "/plain":
            body = b"<!doctype html><html><body><p>plain control</p></body></html>"
            status = 200
        else:
            body = b"not found"
            status = 404
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def make_server() -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", 8790), FixtureHandler)


def validate_fixture_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("positive fixture spec must be an object")
    target = str(spec.get("target", FIXTURE_BASE_URL)).rstrip("/")
    if target != FIXTURE_BASE_URL:
        raise ValueError("positive fixture target must be exactly http://127.0.0.1:8790")
    if str(spec.get("method", "GET")).upper() != "GET":
        raise ValueError("positive fixture permits only read-only GET")
    path = str(spec.get("path", ""))
    if path not in SAFE_PATHS:
        raise ValueError("positive fixture path is not allow-listed")
    params = dict(spec.get("params") or {})
    if set(params) - {"message"} or len(params) > 1:
        raise ValueError("positive fixture only permits the message query field")
    marker = str(spec.get("marker", ""))
    if not MARKER_RE.fullmatch(marker):
        raise ValueError("positive fixture marker must be an inert identifier")
    source_id = str(spec.get("source_id", ""))
    lab_id = str(spec.get("lab_id", ""))
    if not source_id or not lab_id:
        raise ValueError("positive fixture source_id and lab_id are required")
    pair = dict(spec.get("pair") or {})
    if pair:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{4,96}", str(pair.get("pair_id", ""))):
            raise ValueError("positive fixture pair_id is invalid")
        if str(pair.get("variant")) not in {"plain", "url_percent"}:
            raise ValueError("positive fixture pair variant is invalid")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", str(pair.get("surface_role", ""))):
            raise ValueError("positive fixture surface_role is invalid")
        if not isinstance(pair.get("encoding_depth"), int) or not 0 <= pair["encoding_depth"] <= 2:
            raise ValueError("positive fixture encoding_depth is invalid")
    probe = str(spec.get("probe", f'<span data-sift-marker="{marker}">x</span>'))
    payload = build_detection_payload(
        target=FIXTURE_BASE_URL,
        method="GET",
        path=path,
        marker=marker,
        probe=probe,
        probe_kind="inert_dom_markup",
        expected={},
    )
    return {
        "schema_version": FIXTURE_SPEC_SCHEMA,
        "target": FIXTURE_BASE_URL,
        "method": "GET",
        "path": path,
        "params": {str(key): str(value) for key, value in params.items()},
        "marker": marker,
        "probe": probe,
        "encoding": str(spec.get("encoding", "plain")),
        "source_id": source_id,
        "lab_id": lab_id,
        "family": "xss",
        "surface": str(spec.get("surface", "fixture_reflected_attribute")),
        "expected_oracle": FIXTURE_ORACLE,
        "expected_signal": str(spec.get("expected_signal", "inert_marker_in_attribute")),
        "payload": payload,
        **({"pair": pair} if pair else {}),
    }


def default_fixture_specs(marker_prefix: str = "fx-pg06") -> list[dict[str, Any]]:
    marker = f"{marker_prefix}-marker"

    # Encode every byte rather than relying on ``quote``'s intentionally safe
    # alphanumeric set.  httpx encodes the percent signs once more on the wire;
    # the fixture's single ``unquote`` then restores the same canary.  The
    # plain and encoded requests are therefore genuinely different transports
    # with the same semantic marker.
    encoded_marker = _percent_encode_all(marker)
    plain = {
        "source_id": "fixture-pg06",
        "lab_id": "reflect-attribute-plain",
        "surface": "fixture_reflected_attribute",
        "path": "/reflect",
        "params": {"message": marker},
        "marker": marker,
        "probe": f'<span data-sift-marker="{marker}">x</span>',
        "encoding": "plain",
        "pair": {"pair_id": "fixture-pair-01", "variant": "plain", "surface_role": "reflected_attribute", "encoding_depth": 0},
    }
    encoded = dict(plain)
    encoded.update({
        "lab_id": "reflect-attribute-url-percent",
        "encoding": "url_percent",
        "probe": _percent_encode_all(plain["probe"]),
        "params": {"message": encoded_marker},
        "pair": {"pair_id": "fixture-pair-01", "variant": "url_percent", "surface_role": "reflected_attribute", "encoding_depth": 1},
    })
    negative = dict(plain)
    negative.update({
        "lab_id": "plain-control",
        "path": "/plain",
        "params": {"message": marker},
        "pair": {"pair_id": "fixture-pair-02", "variant": "plain", "surface_role": "plain_control", "encoding_depth": 0},
        "expected_signal": "no_attribute_signal",
    })
    return [plain, encoded, negative]


def _percent_encode_all(value: str) -> str:
    return "".join(f"%{byte:02X}" for byte in str(value).encode("utf-8"))


def _response_summary(response: httpx.Response) -> dict[str, Any]:
    body = response.content
    return {
        "status_code": int(response.status_code),
        "headers": {"content-type": str(response.headers.get("content-type", ""))},
        "body_length": len(body),
        "body_sha256": hashlib.sha256(body).hexdigest(),
    }


def _projection(response: httpx.Response, baseline: httpx.Response, marker: str) -> dict[str, Any]:
    body = response.content.decode("utf-8", errors="replace")
    parser = _FixtureHTMLShapeParser()
    parser.feed(body)
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
        "redirect_present": False,
        "status_changed": response.status_code != baseline.status_code,
        "body_length_delta": len(response.content) - len(baseline.content),
        "body_length_delta_abs": abs(len(response.content) - len(baseline.content)),
        "html_tag_count": min(parser.tags, 512),
        "html_attribute_count": min(parser.attributes, 512),
        "script_count": min(parser.scripts, 64),
        "content_type_class": "html",
    }


class PositiveFixtureCollector:
    def __init__(self, *, target_instance_id: str = "unattested", source_hash: str | None = None) -> None:
        self.base_url = FIXTURE_BASE_URL
        self.target_instance_id = str(target_instance_id)[:128]
        self.source_hash = source_hash or fixture_source_sha256()

    async def collect(self, raw_spec: dict[str, Any]) -> dict[str, Any]:
        spec = validate_fixture_spec(raw_spec)
        async with httpx.AsyncClient(base_url=self.base_url, timeout=5.0, follow_redirects=False, cookies={}) as client:
            baseline = await client.get(spec["path"], headers=spec["payload"]["headers"])
            response = await client.get(spec["path"], params=spec["params"], headers=spec["payload"]["headers"])
        baseline_summary = _response_summary(baseline)
        response_summary = _response_summary(response)
        projection = _projection(response, baseline, spec["marker"])
        reset = {
            "kind": "ephemeral_in_repo_fixture",
            "fresh": True,
            "fresh_target": True,
            "state_change_allowed": False,
            "evaluator_state_hidden": True,
            "external_network": False,
            "target_instance_id": self.target_instance_id,
            "fixture_source_sha256": self.source_hash,
        }
        envelope = {
            "collector": FIXTURE_SCHEMA,
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
        record = {
            "schema_version": FIXTURE_SCHEMA,
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
                "expected_oracle": FIXTURE_ORACLE,
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
            "oracle_projection": projection,
            "evidence": checked["body"],
            "rule_ir_result": bool(projection["marker_in_attribute"] and spec["path"] == "/reflect"),
            "candidate_status": "suspicious_surface_signal" if projection["marker_in_attribute"] and spec["path"] == "/reflect" else "clean_observation",
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
    "FIXTURE_BASE_URL",
    "FIXTURE_ORACLE",
    "PositiveFixtureCollector",
    "default_fixture_specs",
    "fixture_source_sha256",
    "make_server",
    "validate_fixture_spec",
]
