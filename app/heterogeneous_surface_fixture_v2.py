"""Independent local heterogeneous-surface fixture for cross-source transfer.

This is deliberately a separate source file and response implementation from
``heterogeneous_surface_fixture.py``.  It keeps the same typed sink contract
while changing the endpoint, query names, markup, JSON/XML shape and header
name.  It is read-only, loopback-only and never executes the supplied marker.
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


V2_SCHEMA = "sift-heterogeneous-surface-fixture-v2"
V2_SPEC_SCHEMA = "sift-heterogeneous-surface-spec-v2"
V2_ORACLE = "fixture_heterogeneous_surface_oracle_v2"
V2_PORTS = (8803, 8804, 8805)
V2_BASE_URL = "http://127.0.0.1:8803"
V2_BASE_URLS = tuple(f"http://127.0.0.1:{port}" for port in V2_PORTS)
V2_SOURCE_PATH = Path(__file__).resolve()
V2_SURFACES = ("html_attribute", "html_text", "json_value", "xml_text", "plain_text", "response_header")
_VARIANTS = frozenset({"alpha", "beta", "gamma"})
_MARKER_RE = re.compile(r"^[A-Za-z0-9._-]{4,64}$")


def heterogeneous_surface_v2_source_sha256() -> str:
    return hashlib.sha256(V2_SOURCE_PATH.read_bytes()).hexdigest()


def _percent_encode_all(value: str) -> str:
    return "".join(f"%{byte:02X}" for byte in str(value).encode("utf-8"))


def _response_for_surface(surface: str, message: str, variant: str) -> tuple[int, str, str, dict[str, str]]:
    if surface == "html_attribute":
        if variant == "beta":
            body = f"<!doctype html><main><textarea data-v2-slot=\"render\" title=\"{html.escape(message, quote=True)}\">v2</textarea></main>"
        elif variant == "gamma":
            body = f"<!doctype html><main><button aria-label=\"{html.escape(message, quote=True)}\">v2</button></main>"
        else:
            body = f"<!doctype html><main><input name=\"render\" value=\"{html.escape(message, quote=True)}\"></main>"
        return 200, body, "text/html; charset=utf-8", {}
    if surface == "html_text":
        body = f"<!doctype html><main><article><strong>{html.escape(message, quote=True)}</strong></article></main>"
        return 200, body, "text/html; charset=utf-8", {}
    if surface == "json_value":
        key = "value" if variant != "gamma" else "rendered"
        body = json.dumps({"ok": 1, "render": {key: message}}, separators=(",", ":"), sort_keys=True)
        return 200, body, "application/json; charset=utf-8", {}
    if surface == "xml_text":
        key = "slot" if variant != "gamma" else "rendered"
        body = f"<?xml version=\"1.0\"?><render><{key}>{html.escape(message, quote=True)}</{key}></render>"
        return 200, body, "application/xml; charset=utf-8", {}
    if surface == "response_header":
        body = "<!doctype html><main><p>v2 header channel</p></main>"
        return 200, body, "text/html; charset=utf-8", {"X-Sift-V2-Echo": message}
    return 200, f"rendered-v2::{message}\n", "text/plain; charset=utf-8", {}


class _V2Handler(BaseHTTPRequestHandler):
    server_version = "sift-heterogeneous-surface-v2/1"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query, keep_blank_values=True)
        slot = unquote(str(query.get("slot", ["plain_text"])[0]))
        value = unquote(str(query.get("value", [""])[0]))
        variant = str(getattr(self.server, "fixture_variant", "alpha"))
        if parsed.path != "/render" or slot not in V2_SURFACES or variant not in _VARIANTS:
            status, body, content_type, extra_headers = 404, "not found", "text/plain; charset=utf-8", {}
        else:
            status, body, content_type, extra_headers = _response_for_surface(slot, value, variant)
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        for key, value in extra_headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        return


class HeterogeneousSurfaceV2Server(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], variant: str) -> None:
        if variant not in _VARIANTS:
            raise ValueError("unknown heterogeneous v2 fixture variant")
        super().__init__(address, _V2Handler)
        self.fixture_variant = variant


def make_heterogeneous_surface_v2_fixture_server(*, port: int = 8803, variant: str = "alpha") -> HeterogeneousSurfaceV2Server:
    if int(port) not in V2_PORTS:
        raise ValueError("heterogeneous v2 fixture port is not allow-listed")
    return HeterogeneousSurfaceV2Server(("127.0.0.1", int(port)), variant)


def _validate_path(path: str) -> tuple[str, dict[str, str]]:
    parsed = urlsplit(path)
    if parsed.path != "/render":
        raise ValueError("heterogeneous v2 fixture path is not allow-listed")
    values = {str(key): unquote(str(item[0])) for key, item in parse_qs(parsed.query, keep_blank_values=True).items() if item}
    if set(values) - {"slot", "value"}:
        raise ValueError("heterogeneous v2 fixture query contains an unknown field")
    if values.get("slot") not in V2_SURFACES:
        raise ValueError("heterogeneous v2 fixture surface is not allow-listed")
    return parsed.path, values


def validate_heterogeneous_surface_v2_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("heterogeneous v2 fixture spec must be an object")
    target = str(spec.get("target", V2_BASE_URL)).rstrip("/")
    if target not in V2_BASE_URLS:
        raise ValueError("heterogeneous v2 fixture target must be an allow-listed loopback URL")
    if str(spec.get("method", "GET")).upper() != "GET":
        raise ValueError("heterogeneous v2 fixture permits only GET")
    path = str(spec.get("path", "/render"))
    _, values = _validate_path(path)
    marker = str(spec.get("marker", "hetero-v2-marker"))
    if not _MARKER_RE.fullmatch(marker):
        raise ValueError("heterogeneous v2 fixture marker is invalid")
    source_id = str(spec.get("source_id", ""))
    lab_id = str(spec.get("lab_id", ""))
    surface = str(spec.get("surface_role", values.get("slot", "")))
    if not source_id or not lab_id or surface not in V2_SURFACES:
        raise ValueError("heterogeneous v2 fixture provenance fields are invalid")
    pair = dict(spec.get("pair") or {})
    if pair:
        if not str(pair.get("pair_id", "")) or str(pair.get("variant")) not in {"plain", "url_percent"}:
            raise ValueError("heterogeneous v2 fixture pair metadata is invalid")
        if str(pair.get("surface_role")) != surface:
            raise ValueError("heterogeneous v2 fixture pair surface mismatch")
        if not isinstance(pair.get("encoding_depth"), int) or not 0 <= pair["encoding_depth"] <= 2:
            raise ValueError("heterogeneous v2 fixture encoding_depth is invalid")
    probe = str(spec.get("probe", "v2-inert-surface-probe"))
    payload = build_detection_payload(target=target, method="GET", path=path, marker=marker, probe=probe, probe_kind="inert_dom_markup", expected={})
    return {
        "schema_version": V2_SPEC_SCHEMA,
        "target": target,
        "method": "GET",
        "path": path,
        "query": values,
        "marker": marker,
        "probe": probe,
        "encoding": str(spec.get("encoding", "plain")),
        "source_id": source_id,
        "lab_id": lab_id,
        "family": "xss",
        "surface_role": surface,
        "expected_oracle": V2_ORACLE,
        "expected_signal": str(spec.get("expected_signal", "typed_surface_sink")),
        "payload": payload,
        **({"pair": pair} if pair else {}),
    }


def _path_for(surface: str, marker: str, *, encoded: bool) -> str:
    slot = _percent_encode_all(surface) if encoded else surface
    value = _percent_encode_all(marker) if encoded else marker
    return f"/render?slot={slot}&value={value}"


def default_heterogeneous_surface_v2_specs(*, dataset_id: str = "fixture-pg13", target: str = V2_BASE_URL, marker: str = "hetero-v2-marker") -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for index, surface in enumerate(V2_SURFACES, start=1):
        pair_id = f"hetero-pg13-{surface}"
        signal = "attribute_sink" if surface == "html_attribute" else "reflected_counterfactual"
        plain = {
            "source_id": dataset_id,
            "lab_id": f"{surface}-plain",
            "target": target,
            "path": _path_for(surface, marker, encoded=False),
            "marker": marker,
            "probe": f"pg13-v2-probe-{index:02d}",
            "encoding": "plain",
            "surface_role": surface,
            "expected_signal": signal,
            "pair": {"pair_id": pair_id, "variant": "plain", "surface_role": surface, "encoding_depth": 0},
        }
        encoded = dict(plain)
        encoded.update({
            "lab_id": f"{surface}-url-percent",
            "path": _path_for(surface, marker, encoded=True),
            "encoding": "url_percent",
            "probe": _percent_encode_all(plain["probe"]),
            "pair": {"pair_id": pair_id, "variant": "url_percent", "surface_role": surface, "encoding_depth": 1},
        })
        specs.extend([plain, encoded])
    return specs


class _V2ShapeParser(HTMLParser):
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
    shape: dict[str, Any] = {"type": "other", "key_count": 0, "scalar_count": 0}
    try:
        parsed = response.json()
        if isinstance(parsed, dict):
            shape = {"type": "object", "key_count": len(parsed), "scalar_count": sum(not isinstance(item, (dict, list)) for item in parsed.values())}
    except (ValueError, json.JSONDecodeError):
        pass
    return {
        "status_code": int(response.status_code),
        "headers": {"content-type": str(response.headers.get("content-type", ""))},
        "body_length": len(body),
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "json_shape": shape,
    }


def _surface_shape(response: httpx.Response) -> dict[str, Any]:
    content_type = str(response.headers.get("content-type", "")).casefold()
    parser = _V2ShapeParser()
    try:
        parser.feed(response.content.decode("utf-8", errors="replace"))
        parser.close()
    except (TypeError, ValueError):
        pass
    content_class = "json" if "json" in content_type else "xml" if "xml" in content_type else "html" if "html" in content_type else "text" if "text" in content_type else "other"
    json_field_count = 0
    if "json" in content_type:
        try:
            value = response.json()
            json_field_count = len(value) if isinstance(value, dict) else 0
        except (ValueError, json.JSONDecodeError):
            pass
    return {
        "content_type_class": content_class,
        "status_class": f"{response.status_code // 100}xx",
        "html_tag_count": min(parser.tags, 64),
        "html_attribute_count": min(parser.attributes, 64),
        "script_count": min(parser.scripts, 32),
        "json_field_count": min(json_field_count, 32),
        "response_header_count": min(len(response.headers), 32),
        "body_length": min(len(response.content), 4096),
    }


class HeterogeneousSurfaceV2Collector:
    def __init__(self, *, base_url: str = V2_BASE_URL, target_instance_id: str = "unattested", source_hash: str | None = None) -> None:
        if base_url.rstrip("/") not in V2_BASE_URLS:
            raise ValueError("heterogeneous v2 collector target is not allow-listed")
        self.base_url = base_url.rstrip("/")
        self.target_instance_id = str(target_instance_id)[:128]
        self.source_hash = source_hash or heterogeneous_surface_v2_source_sha256()

    async def collect(self, raw_spec: dict[str, Any]) -> dict[str, Any]:
        spec = validate_heterogeneous_surface_v2_spec(raw_spec)
        async with httpx.AsyncClient(base_url=self.base_url, timeout=5.0, follow_redirects=False, cookies={}) as client:
            baseline = await client.get("/render?slot=plain_text&value=baseline")
            response = await client.get(spec["path"], headers=spec["payload"]["headers"])
        baseline_summary = _summary(baseline)
        response_summary = _summary(response)
        shape = _surface_shape(response)
        sink = observe_surface_sink(response.content, marker=spec["marker"], content_type=str(response.headers.get("content-type", "")), headers=dict(response.headers))
        projection = {
            **sink,
            "oracle_name": V2_ORACLE,
            "surface_role": spec["surface_role"],
            "typed_surface_signal": bool(spec["surface_role"] == "html_attribute" and sink.get("marker_in_attribute")),
            "status_changed": response.status_code != baseline.status_code,
            "body_length_delta_abs": abs(len(response.content) - len(baseline.content)),
            "external_redirect": False,
        }
        reset = {
            "kind": "ephemeral_in_repo_heterogeneous_surface_v2_fixture",
            "fresh": True,
            "fresh_target": True,
            "state_change_allowed": False,
            "evaluator_state_hidden": True,
            "external_network": False,
            "target_instance_id": self.target_instance_id,
            "fixture_source_sha256": self.source_hash,
        }
        envelope = {
            "collector": V2_SCHEMA,
            "target": self.base_url,
            "path": urlsplit(spec["path"]).path,
            "method": "GET",
            "reset": reset,
            "baseline": baseline_summary,
            "response": response_summary,
            "surface_shape": shape,
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
        positive = bool(projection["typed_surface_signal"])
        record = {
            "schema_version": V2_SCHEMA,
            "sample_id": f"{spec['source_id']}-{spec['lab_id']}-{spec['payload']['payload_sha256'][:12]}",
            "source_id": spec["source_id"],
            "lab_id": spec["lab_id"],
            "family": "xss",
            "payload": spec["payload"],
            "probe_artifact": {"original": spec["probe"], "encoding": spec["encoding"], "probe_sha256": hashlib.sha256(spec["probe"].encode()).hexdigest()},
            "semantic": {"family": "xss", "surface": f"heterogeneous_v2_{spec['surface_role']}", "surface_role": spec["surface_role"], "expected_oracle": V2_ORACLE, "expected_signal": spec["expected_signal"]},
            "evaluator_state_visible": False,
            "replay": {"target": self.base_url, "method": "GET", "path": spec["path"], "fresh_reset": reset, "transport": "httpx_loopback"},
            "response_projection": response_summary,
            "surface_shape": shape,
            "oracle_projection": projection,
            "evidence": checked["body"],
            "rule_ir_result": positive,
            "candidate_status": "typed_attribute_candidate" if positive else "surface_counterfactual",
            "safety": {"local_only": True, "read_only": True, "fresh_reset": True, "fresh_target": True, "external_network": False, "script_execution": False, "database_touched": False, "raw_body_stored": False, "credentials_stored": False, "state_mutated": False},
            "pair": dict(spec["pair"]) if spec.get("pair") else None,
        }
        if record["pair"] is None:
            record.pop("pair")
        return record

    async def collect_many(self, specs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return [await self.collect(spec) for spec in specs]


__all__ = [
    "V2_BASE_URL",
    "V2_BASE_URLS",
    "V2_ORACLE",
    "V2_PORTS",
    "V2_SURFACES",
    "HeterogeneousSurfaceV2Collector",
    "default_heterogeneous_surface_v2_specs",
    "heterogeneous_surface_v2_source_sha256",
    "make_heterogeneous_surface_v2_fixture_server",
    "validate_heterogeneous_surface_v2_spec",
]
