"""PG-277 loopback fixture with matched-shape, different-effect surfaces.

The attribute-positive and text-decoy responses have identical status,
content type, tag count, attribute count and body length.  Only the marker
channel differs.  This makes a deliberate projection collision for models
that receive coarse shape tokens and gives the question policy a reason to
ask for a more informative observation.
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

SCHEMA = "sift-counterfactual-surface-fixture-v1"
SPEC_SCHEMA = "sift-counterfactual-surface-spec-v1"
ORACLE = "fixture_counterfactual_attribute_oracle_v1"
PORTS = (8820, 8821, 8822)
BASE_URLS = tuple(f"http://127.0.0.1:{port}" for port in PORTS)
SOURCE_PATH = Path(__file__).resolve()
MODES = ("attr_sink", "attr_decoy", "header_decoy", "json_decoy", "absent_decoy")
VARIANTS = ("alpha", "beta", "gamma")
MARKER_RE = re.compile(r"^[A-Za-z0-9._-]{4,64}$")


def source_sha256() -> str:
    return hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()


def percent_encode(value: str) -> str:
    return "".join(f"%{byte:02X}" for byte in value.encode("utf-8"))


def _control(marker: str) -> str:
    # Same UTF-8 length as the marker, but guaranteed not to contain it.
    return "Z" * len(marker)


def _html_body(variant: str, attribute: str, text: str) -> str:
    attribute = html.escape(attribute, quote=True)
    text = html.escape(text, quote=True)
    if variant == "beta":
        return f"<!doctype html><section><span title=\"{attribute}\">{text}</span></section>"
    if variant == "gamma":
        return f"<!doctype html><main><output aria-label=\"{attribute}\">{text}</output></main>"
    return f"<!doctype html><div data-slot=\"{attribute}\">{text}</div>"


def render(mode: str, marker: str, variant: str) -> tuple[int, str, str, dict[str, str]]:
    control = _control(marker)
    if mode == "attr_sink":
        return 200, _html_body(variant, marker, control), "text/html; charset=utf-8", {}
    if mode == "attr_decoy":
        return 200, _html_body(variant, control, marker), "text/html; charset=utf-8", {}
    if mode == "header_decoy":
        return 200, _html_body(variant, control, control), "text/html; charset=utf-8", {"X-PG277-Marker": marker}
    if mode == "json_decoy":
        return 200, json.dumps({"ok": True, "value": marker}, separators=(",", ":"), sort_keys=True), "application/json; charset=utf-8", {}
    return 200, _html_body(variant, control, control), "text/html; charset=utf-8", {}


class Handler(BaseHTTPRequestHandler):
    server_version = "sift-counterfactual-surface/1"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        values = parse_qs(parsed.query, keep_blank_values=True)
        mode = unquote(str(values.get("mode", ["absent_decoy"])[0]))
        marker = unquote(str(values.get("m", [""])[0]))
        variant = str(getattr(self.server, "fixture_variant", "alpha"))
        if parsed.path != "/inspect" or mode not in MODES or variant not in VARIANTS or not MARKER_RE.fullmatch(marker):
            status, body, content_type, headers = 404, "not found", "text/plain; charset=utf-8", {}
        else:
            status, body, content_type, headers = render(mode, marker, variant)
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        return


class Server(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], variant: str) -> None:
        if variant not in VARIANTS:
            raise ValueError("unknown counterfactual fixture variant")
        super().__init__(address, Handler)
        self.fixture_variant = variant


def make_server(*, port: int, variant: str) -> Server:
    if int(port) not in PORTS:
        raise ValueError("counterfactual fixture port is not allow-listed")
    return Server(("127.0.0.1", int(port)), variant)


def _path(mode: str, marker: str, *, encoded: bool) -> str:
    value_mode = percent_encode(mode) if encoded else mode
    value_marker = percent_encode(marker) if encoded else marker
    return f"/inspect?mode={value_mode}&m={value_marker}"


def validate_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("counterfactual spec must be an object")
    target = str(spec.get("target", "")).rstrip("/")
    if target not in BASE_URLS or str(spec.get("method", "GET")).upper() != "GET":
        raise ValueError("counterfactual target/method is not allow-listed")
    path = str(spec.get("path", ""))
    parsed = urlsplit(path)
    values = {key: unquote(str(items[0])) for key, items in parse_qs(parsed.query, keep_blank_values=True).items() if items}
    if parsed.path != "/inspect" or set(values) != {"mode", "m"} or values["mode"] not in MODES:
        raise ValueError("counterfactual path is not allow-listed")
    marker = str(spec.get("marker", ""))
    if not MARKER_RE.fullmatch(marker) or values["m"] != marker:
        raise ValueError("counterfactual marker is invalid")
    source_id, lab_id = str(spec.get("source_id", "")), str(spec.get("lab_id", ""))
    if not source_id or not lab_id:
        raise ValueError("counterfactual provenance is incomplete")
    encoding = str(spec.get("encoding", "plain"))
    if encoding not in {"plain", "url_percent"}:
        raise ValueError("counterfactual encoding is invalid")
    probe = str(spec.get("probe", "pg277-inert-probe"))
    payload = build_detection_payload(target=target, method="GET", path=path, marker=marker, probe=probe, probe_kind="inert_dom_markup", expected={})
    return {"schema_version": SPEC_SCHEMA, "target": target, "method": "GET", "path": path, "mode": values["mode"], "marker": marker, "probe": probe, "encoding": encoding, "source_id": source_id, "lab_id": lab_id, "payload": payload}


def default_specs(*, dataset_id: str, target: str, marker: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for index, mode in enumerate(MODES, start=1):
        plain = {"source_id": dataset_id, "lab_id": f"{mode}-plain", "target": target, "method": "GET", "path": _path(mode, marker, encoded=False), "mode": mode, "marker": marker, "probe": f"pg277-probe-{index:02d}", "encoding": "plain"}
        encoded = dict(plain)
        encoded.update({"lab_id": f"{mode}-url-percent", "path": _path(mode, marker, encoded=True), "probe": percent_encode(plain["probe"]), "encoding": "url_percent"})
        specs.extend([plain, encoded])
    return specs


class ShapeParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags = 0
        self.attributes = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags += 1
        self.attributes += len(attrs)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)


def summary(response: httpx.Response) -> dict[str, Any]:
    body = response.content
    return {"status_code": int(response.status_code), "content_type": str(response.headers.get("content-type", "")), "body_length": len(body), "body_sha256": hashlib.sha256(body).hexdigest()}


def shape(response: httpx.Response) -> dict[str, Any]:
    content_type = str(response.headers.get("content-type", "")).casefold()
    parser = ShapeParser()
    parser.feed(response.content.decode("utf-8", errors="replace"))
    json_fields = 0
    if "json" in content_type:
        try:
            value = response.json()
            json_fields = len(value) if isinstance(value, dict) else 0
        except (ValueError, json.JSONDecodeError):
            pass
    content_class = "json" if "json" in content_type else "html" if "html" in content_type else "text"
    return {"content_type_class": content_class, "status_class": f"{response.status_code // 100}xx", "html_tag_count": min(parser.tags, 64), "html_attribute_count": min(parser.attributes, 64), "json_field_count": min(json_fields, 32), "response_header_count": min(len(response.headers), 32), "body_length": min(len(response.content), 4096)}


def marker_channel(response: httpx.Response, marker: str) -> tuple[str, dict[str, Any]]:
    projection = observe_surface_sink(response.content, marker=marker, content_type=str(response.headers.get("content-type", "")), headers=dict(response.headers))
    content_type = str(response.headers.get("content-type", "")).casefold()
    if "json" in content_type and marker in response.text:
        channel = "json"
    elif projection.get("marker_in_attribute"):
        channel = "attribute"
    elif projection.get("marker_in_html_text"):
        channel = "html_text"
    elif projection.get("marker_in_header"):
        channel = "header"
    elif projection.get("marker_in_json_value"):
        channel = "json"
    else:
        channel = "absent"
    return channel, projection


class Collector:
    def __init__(self, *, base_url: str, variant: str, target_instance_id: str, fixture_hash: str | None = None) -> None:
        if base_url.rstrip("/") not in BASE_URLS or variant not in VARIANTS:
            raise ValueError("counterfactual collector target is not allow-listed")
        self.base_url = base_url.rstrip("/")
        self.variant = variant
        self.target_instance_id = target_instance_id
        self.fixture_hash = fixture_hash or source_sha256()

    async def collect(self, raw_spec: dict[str, Any]) -> dict[str, Any]:
        spec = validate_spec(raw_spec)
        marker = spec["marker"]
        async with httpx.AsyncClient(base_url=self.base_url, timeout=5.0, follow_redirects=False, cookies={}) as client:
            negative = await client.get(_path("absent_decoy", marker, encoded=False))
            reference = await client.get(_path("attr_sink", marker, encoded=False))
            candidate = await client.get(spec["path"], headers=spec["payload"]["headers"])
        negative_channel, negative_projection = marker_channel(negative, marker)
        reference_channel, reference_projection = marker_channel(reference, marker)
        candidate_channel, candidate_projection = marker_channel(candidate, marker)
        typed_positive = bool(spec["mode"] == "attr_sink" and candidate_channel == "attribute" and reference_channel == "attribute" and negative_channel == "absent")
        reset = {"kind": "ephemeral_counterfactual_surface_fixture", "fresh": True, "fresh_target": True, "state_change_allowed": False, "evaluator_state_hidden": True, "external_network": False, "target_instance_id": self.target_instance_id, "fixture_source_sha256": self.fixture_hash}
        candidate_shape = shape(candidate)
        envelope = {"collector": SCHEMA, "target": self.base_url, "method": "GET", "reset": reset, "negative": summary(negative), "reference": summary(reference), "candidate": summary(candidate), "candidate_shape": candidate_shape, "observation_projection": {"negative_channel": negative_channel, "reference_channel": reference_channel, "candidate_channel": candidate_channel, "candidate_reference_match": candidate_channel == reference_channel, "negative_clean": negative_channel == "absent"}, "oracle_projection": {"oracle_name": ORACLE, "typed_positive": typed_positive}, "local_http_loopback": True, "script_execution": False, "network_access": False, "database_touched": False, "encoding": spec["encoding"], "payload_sha256": spec["payload"]["payload_sha256"]}
        envelope["evidence_hash"] = sha256_json(envelope)
        checked = validate_evidence(envelope)
        return {"schema_version": SCHEMA, "record_id": f"{spec['source_id']}-{spec['lab_id']}-{spec['payload']['payload_sha256'][:12]}", "source_id": spec["source_id"], "lab_id": spec["lab_id"], "variant": self.variant, "mode": spec["mode"], "encoding": spec["encoding"], "probe_artifact": {"encoding": spec["encoding"], "probe_sha256": hashlib.sha256(spec["probe"].encode()).hexdigest()}, "candidate_shape": candidate_shape, "observation_projection": envelope["observation_projection"], "oracle_projection": envelope["oracle_projection"], "rule_ir_result": typed_positive, "evidence": checked["body"], "safety": {"local_only": True, "read_only": True, "fresh_reset": True, "fresh_target": True, "external_network": False, "script_execution": False, "database_touched": False, "raw_body_stored": False, "raw_payload_stored": False, "state_mutated": False}}

    async def collect_many(self, specs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return [await self.collect(spec) for spec in specs]


__all__ = ["PORTS", "VARIANTS", "Collector", "default_specs", "make_server", "source_sha256"]
