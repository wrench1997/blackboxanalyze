"""Bounded read-only shadow collector for the local Juice Shop proxy.

This adapter is intentionally not a vulnerability scanner.  It exercises a
small, allow-listed set of GET surfaces from a second local application and
emits only structural projections.  No challenge/evaluator endpoint, cookie,
credential, raw body, or mutating request is allowed.
"""

from __future__ import annotations

import hashlib
import json
import re
from html.parser import HTMLParser
from typing import Any, Iterable
from urllib.parse import urlsplit

import httpx

from .detection_payload import build_detection_payload
from .juice_shop_adapter import DEFAULT_BASE_URL, PINNED_IMAGE, BLOCKED_AGENT_PATHS, stable_json_projection
from .maze_engine import sha256_json, validate_evidence


JUICE_SHOP_SHADOW_SCHEMA = "sift-juice-shop-shadow-collector-v1"
JUICE_SHOP_SHADOW_SPEC_SCHEMA = "sift-juice-shop-shadow-spec-v1"
JUICE_SHOP_SHADOW_ORACLE = "juice_shop_bounded_http_projection_v1"
SAFE_SHADOW_PATHS = frozenset({"/", "/robots.txt", "/rest/products/search"})
SAFE_SHADOW_QUERY_NAMES = frozenset({"q"})
SAFE_SHADOW_FAMILIES = frozenset({"logic"})
SQL_ERROR_PATTERNS = ("sql syntax", "mysql", "sqlite error", "odbc")
MARKER_RE = re.compile(r"^[A-Za-z0-9._-]{4,64}$")


class _HTMLShapeParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags = 0
        self.forms = 0
        self.scripts = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags += 1
        lower = tag.casefold()
        self.forms += int(lower == "form")
        self.scripts += int(lower == "script")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)


def _assert_target(value: str) -> str:
    parsed = urlsplit(str(value))
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port != 3100
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Juice Shop shadow target must be exactly http://127.0.0.1:3100")
    return DEFAULT_BASE_URL


def _params(raw: Any) -> dict[str, str]:
    params = dict(raw or {})
    if len(params) > 1:
        raise ValueError("Juice Shop shadow permits at most one query parameter")
    normalized: dict[str, str] = {}
    for key, value in params.items():
        if str(key).casefold() not in SAFE_SHADOW_QUERY_NAMES:
            raise ValueError("Juice Shop shadow query field is not allow-listed")
        text = str(value)
        if len(text) > 256 or any(token in text.casefold() for token in ("javascript:", "<script", "union select", "../")):
            raise ValueError("Juice Shop shadow query value is not an inert canary")
        normalized[str(key)] = text
    return normalized


def validate_juice_shop_shadow_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("Juice Shop shadow spec must be an object")
    target = _assert_target(str(spec.get("target", DEFAULT_BASE_URL)))
    method = str(spec.get("method", "GET")).upper()
    if method != "GET":
        raise ValueError("Juice Shop shadow only permits read-only GET")
    path = str(spec.get("path", ""))
    normalized_path = path.casefold().split("?", 1)[0].rstrip("/") or "/"
    if path not in SAFE_SHADOW_PATHS or any(normalized_path == blocked or normalized_path.startswith(f"{blocked}/") for blocked in BLOCKED_AGENT_PATHS):
        raise ValueError("Juice Shop shadow path is not allow-listed")
    marker = str(spec.get("marker", "js-shadow-probe"))
    if not MARKER_RE.fullmatch(marker):
        raise ValueError("Juice Shop shadow marker must be an inert identifier")
    params = _params(spec.get("params"))
    family = str(spec.get("family", "logic"))
    if family not in SAFE_SHADOW_FAMILIES:
        raise ValueError("Juice Shop shadow family is not supported")
    source_id = str(spec.get("source_id", ""))
    lab_id = str(spec.get("lab_id", ""))
    if not source_id or len(source_id) > 96 or not lab_id or len(lab_id) > 128:
        raise ValueError("Juice Shop shadow source_id and lab_id are required")
    probe = str(spec.get("probe", marker))
    payload = build_detection_payload(
        target=target,
        method="GET",
        path=path,
        marker=marker,
        probe=probe,
        probe_kind="http_canary",
        expected={},
    )
    return {
        "schema_version": JUICE_SHOP_SHADOW_SPEC_SCHEMA,
        "target": target,
        "method": "GET",
        "path": path,
        "params": params,
        "marker": marker,
        "probe": probe,
        "source_id": source_id,
        "lab_id": lab_id,
        "family": family,
        "surface": str(spec.get("surface", lab_id)),
        "expected_oracle": JUICE_SHOP_SHADOW_ORACLE,
        "expected_signal": "no_family_specific_oracle",
        "payload": payload,
    }


def default_juice_shop_shadow_specs(marker_prefix: str = "js-cross") -> list[dict[str, Any]]:
    expected_marker = f"{marker_prefix}-oracle"
    input_marker = f"{marker_prefix}-input"
    return [
        {
            "source_id": "juice-shop-shadow-pg05",
            "lab_id": "spa-shell",
            "surface": "juice_spa_shell",
            "path": "/",
            "params": {},
            "marker": expected_marker,
            "probe": input_marker,
        },
        {
            "source_id": "juice-shop-shadow-pg05",
            "lab_id": "robots-text",
            "surface": "juice_robots_text",
            "path": "/robots.txt",
            "params": {},
            "marker": expected_marker,
            "probe": input_marker,
        },
        {
            "source_id": "juice-shop-shadow-pg05",
            "lab_id": "search-unknown",
            "surface": "juice_search_unknown",
            "path": "/rest/products/search",
            "params": {"q": input_marker},
            "marker": expected_marker,
            "probe": input_marker,
        },
        {
            "source_id": "juice-shop-shadow-pg05",
            "lab_id": "search-common",
            "surface": "juice_search_common",
            "path": "/rest/products/search",
            "params": {"q": "apple"},
            "marker": expected_marker,
            "probe": "apple",
        },
        {
            "source_id": "juice-shop-shadow-pg05",
            "lab_id": "search-percent-input",
            "surface": "juice_search_percent_input",
            "path": "/rest/products/search",
            "params": {"q": f"{input_marker}%2520x"},
            "marker": expected_marker,
            "probe": f"{input_marker}%2520x",
        },
    ]


def _response_summary(response: httpx.Response) -> dict[str, Any]:
    body = response.content
    return {
        "status_code": int(response.status_code),
        "headers": {
            key: str(response.headers[key])
            for key in ("content-type", "content-length") if key in response.headers
        },
        "body_length": len(body),
        "body_sha256": hashlib.sha256(body).hexdigest(),
    }


def _body_shape(response: httpx.Response) -> dict[str, Any]:
    content_type = str(response.headers.get("content-type", "")).casefold()
    text = response.content.decode("utf-8", errors="replace")
    if "json" in content_type:
        try:
            value = stable_json_projection(response.json())
            if isinstance(value, dict):
                return {"kind": "object", "field_count": min(len(value), 64), "value_types": sorted({type(item).__name__ for item in value.values()})[:8]}
            if isinstance(value, list):
                return {"kind": "array", "length_bucket": min(len(value), 32)}
            return {"kind": type(value).__name__}
        except (ValueError, TypeError):
            return {"kind": "json_parse_error"}
    parser = _HTMLShapeParser()
    try:
        parser.feed(text)
        parser.close()
    except (TypeError, ValueError):
        pass
    return {"kind": "html_or_text", "tag_count": min(parser.tags, 512), "form_count": min(parser.forms, 64), "script_count": min(parser.scripts, 64)}


def _same_origin_external(location: str) -> tuple[bool, bool]:
    if not location:
        return False, False
    parsed = urlsplit(location)
    if parsed.scheme or parsed.netloc:
        return True, not (parsed.scheme == "http" and parsed.hostname == "127.0.0.1" and parsed.port in {None, 3100})
    return True, location.startswith("//")


def _projection(response: httpx.Response, baseline: httpx.Response, marker: str) -> dict[str, Any]:
    text = response.content.decode("utf-8", errors="replace")
    lowered = text.casefold()
    has_location, external = _same_origin_external(str(response.headers.get("location", "")))
    return {
        "marker_reflected": marker in text,
        "marker_count": min(text.count(marker), 8),
        "sql_error_shape": any(pattern in lowered for pattern in SQL_ERROR_PATTERNS),
        "external_redirect": external,
        "redirect_present": has_location,
        "status_changed": response.status_code != baseline.status_code,
        "body_length_delta": len(response.content) - len(baseline.content),
        "body_length_delta_abs": abs(len(response.content) - len(baseline.content)),
        "content_type_class": "json" if "json" in response.headers.get("content-type", "").casefold() else "html_or_text",
        "body_shape": _body_shape(response),
    }


class JuiceShopShadowCollector:
    def __init__(self, *, base_url: str = DEFAULT_BASE_URL, timeout_seconds: float = 5.0, target_instance_id: str = "unattested") -> None:
        self.base_url = _assert_target(base_url)
        self.timeout_seconds = min(max(float(timeout_seconds), 0.5), 5.0)
        self.target_instance_id = str(target_instance_id)[:128]
        self.fresh_target = False

    async def collect(self, raw_spec: dict[str, Any]) -> dict[str, Any]:
        spec = validate_juice_shop_shadow_spec(raw_spec)
        headers = dict(spec["payload"]["headers"])
        transport_error: httpx.HTTPError | None = None
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_seconds, follow_redirects=False, cookies={}) as client:
            try:
                baseline = await client.get(spec["path"], headers=headers)
                response = await client.get(spec["path"], params=spec["params"], headers=headers)
            except httpx.HTTPError as exc:
                # A stopped local container is an environment observation,
                # not a clean/negative surface.  Emit a bounded fail-closed
                # record so replay diagnostics and the test suite do not
                # crash or silently manufacture a label.
                transport_error = exc
        if transport_error is not None:
            empty_hash = hashlib.sha256(b"").hexdigest()
            baseline_summary = {"status_code": None, "headers": {}, "body_length": 0, "body_sha256": empty_hash, "json_shape": {"kind": "transport_error"}, "transport_error": True, "transport_error_type": type(transport_error).__name__}
            response_summary = dict(baseline_summary)
            projection = {"transport_error": True, "transport_error_type": type(transport_error).__name__, "marker_reflected": False, "marker_count": 0, "sql_error_shape": False, "external_redirect": False, "redirect_present": False, "status_changed": False, "body_length_delta": 0, "body_length_delta_abs": 0, "content_type_class": "unknown", "body_shape": {"kind": "transport_error"}}
        else:
            baseline_summary = _response_summary(baseline)
            response_summary = _response_summary(response)
            # The decoder receives only this bounded structural shape; no JSON
            # values or object keys are retained.
            response_summary["json_shape"] = _body_shape(response)
            baseline_summary["json_shape"] = _body_shape(baseline)
            projection = _projection(response, baseline, spec["marker"])
        reset = {
            "kind": "pinned_juice_shop_read_only",
            "fresh": False,
            "fresh_target": False,
            "state_change_allowed": False,
            "evaluator_state_hidden": True,
            "external_network": False,
            "target_instance_id": self.target_instance_id,
            "container_image_digest": PINNED_IMAGE.split("@", 1)[-1],
        }
        envelope = {
            "collector": JUICE_SHOP_SHADOW_SCHEMA,
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
        }
        envelope["evidence_hash"] = sha256_json(envelope)
        checked = validate_evidence(envelope)
        record = {
            "schema_version": JUICE_SHOP_SHADOW_SCHEMA,
            "sample_id": f"{spec['source_id']}-{spec['lab_id']}-{spec['payload']['payload_sha256'][:12]}",
            "source_id": spec["source_id"],
            "lab_id": spec["lab_id"],
            "family": spec["family"],
            "payload": spec["payload"],
            "probe_artifact": {
                "original": spec["probe"],
                "encoding": "cross_app_inert_canary",
                "probe_sha256": hashlib.sha256(spec["probe"].encode("utf-8")).hexdigest(),
            },
            "semantic": {
                "family": spec["family"],
                "surface": spec["surface"],
                "expected_oracle": spec["expected_oracle"],
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
            "rule_ir_result": False,
            "candidate_status": "environment_failure_abstain" if transport_error is not None else "unsupported_surface_abstain",
            "environment_failure": transport_error is not None,
            "safety": {
                "local_only": True,
                "read_only": True,
                "fresh_reset": False,
                "fresh_target": False,
                "external_network": False,
                "script_execution": False,
                "database_touched": False,
                "real_sleep_performed": False,
                "raw_body_stored": False,
                "credentials_stored": False,
            },
        }
        return record

    async def collect_many(self, specs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for spec in specs:
            records.append(await self.collect(spec))
        return records


__all__ = [
    "JUICE_SHOP_SHADOW_ORACLE",
    "JUICE_SHOP_SHADOW_SCHEMA",
    "JuiceShopShadowCollector",
    "SAFE_SHADOW_PATHS",
    "default_juice_shop_shadow_specs",
    "validate_juice_shop_shadow_spec",
]
