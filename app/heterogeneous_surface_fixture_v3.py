"""PG-276 third, independently implemented loopback surface fixture.

It deliberately changes route/query names, markup, JSON shape, header name and
ports relative to the v1/v2 fixtures.  The marker is inert and escaped; the
typed oracle only confirms an attribute projection.  No scripts, database or
external network are used.
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

V3_SCHEMA = "sift-heterogeneous-surface-fixture-v3"
V3_SPEC_SCHEMA = "sift-heterogeneous-surface-spec-v3"
V3_ORACLE = "fixture_heterogeneous_surface_oracle_v3"
V3_PORTS = (8810, 8811, 8812)
V3_BASE_URL = "http://127.0.0.1:8810"
V3_BASE_URLS = tuple(f"http://127.0.0.1:{port}" for port in V3_PORTS)
V3_SOURCE_PATH = Path(__file__).resolve()
V3_SURFACES = ("attr", "text", "json", "xml", "header", "plain")
_VARIANTS = frozenset({"alpha", "beta", "gamma"})
_MARKER_RE = re.compile(r"^[A-Za-z0-9._-]{4,64}$")


def heterogeneous_surface_v3_source_sha256() -> str:
    return hashlib.sha256(V3_SOURCE_PATH.read_bytes()).hexdigest()


def _percent_encode_all(value: str) -> str:
    return "".join(f"%{byte:02X}" for byte in str(value).encode("utf-8"))


def _render(surface: str, value: str, variant: str) -> tuple[int, str, str, dict[str, str]]:
    if surface == "attr":
        if variant == "beta":
            body = f"<!doctype html><section><div data-q-slot=\"{html.escape(value, quote=True)}\">v3</div></section>"
        elif variant == "gamma":
            body = f"<!doctype html><section><output aria-label=\"{html.escape(value, quote=True)}\">v3</output></section>"
        else:
            body = f"<!doctype html><section><input data-render=\"{html.escape(value, quote=True)}\"></section>"
        return 200, body, "text/html; charset=utf-8", {}
    if surface == "text":
        return 200, f"<!doctype html><section><em>{html.escape(value, quote=True)}</em></section>", "text/html; charset=utf-8", {}
    if surface == "json":
        key = "echo" if variant != "gamma" else "value"
        return 200, json.dumps({"ok": True, "data": {key: value}}, separators=(",", ":"), sort_keys=True), "application/json; charset=utf-8", {}
    if surface == "xml":
        key = "echo" if variant != "gamma" else "value"
        return 200, f"<?xml version=\"1.0\"?><data><{key}>{html.escape(value, quote=True)}</{key}></data>", "application/xml; charset=utf-8", {}
    if surface == "header":
        return 200, "<!doctype html><section><p>header channel</p></section>", "text/html; charset=utf-8", {"X-Render-V3": value}
    return 200, "<!doctype html><section><p>plain control</p></section>", "text/html; charset=utf-8", {}


class _V3Handler(BaseHTTPRequestHandler):
    server_version = "sift-heterogeneous-surface-v3/1"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query, keep_blank_values=True)
        channel = unquote(str(query.get("channel", ["plain"])[0]))
        value = unquote(str(query.get("q", [""])[0]))
        variant = str(getattr(self.server, "fixture_variant", "alpha"))
        if parsed.path != "/view" or channel not in V3_SURFACES or variant not in _VARIANTS:
            status, body, content_type, headers = 404, "not found", "text/plain; charset=utf-8", {}
        else:
            status, body, content_type, headers = _render(channel, value, variant)
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        for key, item in headers.items():
            self.send_header(key, item)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        return


class HeterogeneousSurfaceV3Server(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], variant: str) -> None:
        if variant not in _VARIANTS:
            raise ValueError("unknown heterogeneous v3 fixture variant")
        super().__init__(address, _V3Handler)
        self.fixture_variant = variant


def make_heterogeneous_surface_v3_fixture_server(*, port: int = 8810, variant: str = "alpha") -> HeterogeneousSurfaceV3Server:
    if int(port) not in V3_PORTS:
        raise ValueError("heterogeneous v3 fixture port is not allow-listed")
    return HeterogeneousSurfaceV3Server(("127.0.0.1", int(port)), variant)


def _path_for(surface: str, marker: str, *, encoded: bool) -> str:
    channel = _percent_encode_all(surface) if encoded else surface
    value = _percent_encode_all(marker) if encoded else marker
    return f"/view?channel={channel}&q={value}"


def validate_heterogeneous_surface_v3_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("heterogeneous v3 fixture spec must be an object")
    target = str(spec.get("target", V3_BASE_URL)).rstrip("/")
    if target not in V3_BASE_URLS or str(spec.get("method", "GET")).upper() != "GET":
        raise ValueError("heterogeneous v3 fixture target/method is not allow-listed")
    path = str(spec.get("path", "/view"))
    parsed = urlsplit(path)
    values = {str(key): unquote(str(item[0])) for key, item in parse_qs(parsed.query, keep_blank_values=True).items() if item}
    if parsed.path != "/view" or set(values) != {"channel", "q"} or values["channel"] not in V3_SURFACES:
        raise ValueError("heterogeneous v3 fixture path is not allow-listed")
    marker = str(spec.get("marker", "hetero-v3-marker"))
    if not _MARKER_RE.fullmatch(marker):
        raise ValueError("heterogeneous v3 fixture marker is invalid")
    source_id, lab_id = str(spec.get("source_id", "")), str(spec.get("lab_id", ""))
    surface = str(spec.get("surface_role", values["channel"]))
    if not source_id or not lab_id or surface not in V3_SURFACES:
        raise ValueError("heterogeneous v3 fixture provenance is invalid")
    pair = dict(spec.get("pair") or {})
    if pair and (str(pair.get("variant")) not in {"plain", "url_percent"} or str(pair.get("surface_role")) != surface):
        raise ValueError("heterogeneous v3 fixture pair is invalid")
    probe = str(spec.get("probe", "v3-inert-surface-probe"))
    payload = build_detection_payload(target=target, method="GET", path=path, marker=marker, probe=probe, probe_kind="inert_dom_markup", expected={})
    return {"schema_version": V3_SPEC_SCHEMA, "target": target, "method": "GET", "path": path, "marker": marker, "probe": probe, "encoding": str(spec.get("encoding", "plain")), "source_id": source_id, "lab_id": lab_id, "family": "xss", "surface_role": surface, "expected_signal": "attribute_sink" if surface == "attr" else "reflected_counterfactual", "payload": payload, **({"pair": pair} if pair else {})}


def default_heterogeneous_surface_v3_specs(*, dataset_id: str = "fixture-pg14", target: str = V3_BASE_URL, marker: str = "hetero-v3-marker") -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for index, surface in enumerate(V3_SURFACES, start=1):
        pair_id = f"hetero-pg14-{surface}"
        plain = {"source_id": dataset_id, "lab_id": f"{surface}-plain", "target": target, "path": _path_for(surface, marker, encoded=False), "marker": marker, "probe": f"pg14-v3-probe-{index:02d}", "encoding": "plain", "surface_role": surface, "pair": {"pair_id": pair_id, "variant": "plain", "surface_role": surface, "encoding_depth": 0}}
        encoded = dict(plain)
        encoded.update({"lab_id": f"{surface}-url-percent", "path": _path_for(surface, marker, encoded=True), "encoding": "url_percent", "probe": _percent_encode_all(plain["probe"]), "pair": {"pair_id": pair_id, "variant": "url_percent", "surface_role": surface, "encoding_depth": 1}})
        specs.extend([plain, encoded])
    return specs


class _V3ShapeParser(HTMLParser):
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


def _summary(response: httpx.Response) -> dict[str, Any]:
    body = response.content
    return {"status_code": int(response.status_code), "headers": {"content-type": str(response.headers.get("content-type", ""))}, "body_length": len(body), "body_sha256": hashlib.sha256(body).hexdigest()}


def _shape(response: httpx.Response) -> dict[str, Any]:
    content_type = str(response.headers.get("content-type", "")).casefold()
    parser = _V3ShapeParser()
    parser.feed(response.content.decode("utf-8", errors="replace"))
    json_fields = 0
    if "json" in content_type:
        try:
            value = response.json()
            json_fields = len(value) if isinstance(value, dict) else 0
        except (ValueError, json.JSONDecodeError):
            pass
    kind = "json" if "json" in content_type else "xml" if "xml" in content_type else "html" if "html" in content_type else "text" if "text" in content_type else "other"
    return {"content_type_class": kind, "status_class": f"{response.status_code // 100}xx", "html_tag_count": min(parser.tags, 64), "html_attribute_count": min(parser.attributes, 64), "script_count": min(parser.scripts, 32), "json_field_count": min(json_fields, 32), "response_header_count": min(len(response.headers), 32), "body_length": min(len(response.content), 4096)}


class HeterogeneousSurfaceV3Collector:
    def __init__(self, *, base_url: str = V3_BASE_URL, target_instance_id: str = "unattested", source_hash: str | None = None) -> None:
        if base_url.rstrip("/") not in V3_BASE_URLS:
            raise ValueError("heterogeneous v3 collector target is not allow-listed")
        self.base_url = base_url.rstrip("/")
        self.target_instance_id = str(target_instance_id)[:128]
        self.source_hash = source_hash or heterogeneous_surface_v3_source_sha256()

    async def collect(self, raw_spec: dict[str, Any]) -> dict[str, Any]:
        spec = validate_heterogeneous_surface_v3_spec(raw_spec)
        async with httpx.AsyncClient(base_url=self.base_url, timeout=5.0, follow_redirects=False, cookies={}) as client:
            baseline = await client.get("/view?channel=plain&q=baseline")
            response = await client.get(spec["path"], headers=spec["payload"]["headers"])
        projection = {**observe_surface_sink(response.content, marker=spec["marker"], content_type=str(response.headers.get("content-type", "")), headers=dict(response.headers)), "oracle_name": V3_ORACLE, "surface_role": spec["surface_role"], "typed_surface_signal": bool(spec["surface_role"] == "attr" and False)}
        # The escaped attribute is observable as a sink projection; the oracle
        # intentionally checks the inert marker, not script execution.
        projection["typed_surface_signal"] = bool(spec["surface_role"] == "attr" and projection.get("marker_in_attribute"))
        reset = {"kind": "ephemeral_in_repo_heterogeneous_surface_v3_fixture", "fresh": True, "fresh_target": True, "state_change_allowed": False, "evaluator_state_hidden": True, "external_network": False, "target_instance_id": self.target_instance_id, "fixture_source_sha256": self.source_hash}
        shape = _shape(response)
        envelope = {"collector": V3_SCHEMA, "target": self.base_url, "path": urlsplit(spec["path"]).path, "method": "GET", "reset": reset, "baseline": _summary(baseline), "response": _summary(response), "surface_shape": shape, "oracle_projection": projection, "local_http_loopback": True, "script_execution": False, "network_access": False, "navigation": False, "database_touched": False, "real_sleep_performed": False, "credentials_accessed": False, "encoding": spec["encoding"], "payload_sha256": spec["payload"]["payload_sha256"]}
        envelope["evidence_hash"] = sha256_json(envelope)
        checked = validate_evidence(envelope)
        positive = bool(projection["typed_surface_signal"])
        return {"schema_version": V3_SCHEMA, "sample_id": f"{spec['source_id']}-{spec['lab_id']}-{spec['payload']['payload_sha256'][:12]}", "source_id": spec["source_id"], "lab_id": spec["lab_id"], "family": "xss", "payload": spec["payload"], "probe_artifact": {"original": spec["probe"], "encoding": spec["encoding"], "probe_sha256": hashlib.sha256(spec["probe"].encode()).hexdigest()}, "semantic": {"family": "xss", "surface": f"heterogeneous_v3_{spec['surface_role']}", "surface_role": spec["surface_role"], "expected_oracle": V3_ORACLE, "expected_signal": spec["expected_signal"]}, "evaluator_state_visible": False, "replay": {"target": self.base_url, "method": "GET", "path": spec["path"], "fresh_reset": reset, "transport": "httpx_loopback"}, "response_projection": _summary(response), "surface_shape": shape, "oracle_projection": projection, "evidence": checked["body"], "rule_ir_result": positive, "candidate_status": "typed_attribute_candidate" if positive else "surface_counterfactual", "safety": {"local_only": True, "read_only": True, "fresh_reset": True, "fresh_target": True, "external_network": False, "script_execution": False, "database_touched": False, "raw_body_stored": False, "credentials_stored": False, "state_mutated": False}, "pair": dict(spec["pair"]) if spec.get("pair") else None}

    async def collect_many(self, specs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return [await self.collect(spec) for spec in specs]


__all__ = ["V3_PORTS", "V3_BASE_URL", "V3_BASE_URLS", "HeterogeneousSurfaceV3Collector", "default_heterogeneous_surface_v3_specs", "heterogeneous_surface_v3_source_sha256", "make_heterogeneous_surface_v3_fixture_server", "validate_heterogeneous_surface_v3_spec"]
