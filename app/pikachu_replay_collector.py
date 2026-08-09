"""Bounded, staged probes for an explicitly local Pikachu container.

This adapter is intentionally narrower than a penetration-testing tool.  It
only sends inert canaries to an exact loopback endpoint, keeps no cookies or
raw response bodies, and emits a bounded response projection suitable for
Rule IR training.  The collector does not execute JavaScript and never sends
RCE, SSRF, XXE, upload, traversal, authentication, or time-delay payloads.
"""

from __future__ import annotations

import hashlib
import json
import re
from html.parser import HTMLParser
from typing import Any, Iterable
from urllib.parse import quote, urlsplit

import httpx

from .detection_payload import ALLOWED_PROBE_KINDS, build_detection_payload
from .maze_engine import sha256_json, validate_evidence
from .rule_ir import canonical as canonical_rule_ir
from .rule_ir import complexity as rule_ir_complexity
from .rule_ir import evaluate as evaluate_rule_ir


PIKACHU_COLLECTOR_SCHEMA = "sift-pikachu-replay-collector-v1"
PIKACHU_SPEC_SCHEMA = "sift-pikachu-replay-spec-v1"
PIKACHU_BASE_URL = "http://127.0.0.1:8766"
PIKACHU_FRESH_BASE_URL = "http://127.0.0.1:8767"
PIKACHU_FRESH_BASE_URL_2 = "http://127.0.0.1:8768"
PIKACHU_IMAGE_DIGEST = "sha256:b32c7362bb102091bd4ef09c5c571db146bd57469d5598f0c8681ffeeb7907fe"

# These pages are read-only GET surfaces.  Other Pikachu pages remain in the
# inventory, but are explicitly represented as abstentions by the runner.
SAFE_PROBE_PATHS = frozenset({
    "/vul/xss/xss_reflected_get.php",
    "/vul/xss/xss_dom.php",
    "/vul/sqli/sqli_str.php",
    "/vul/sqli/sqli_search.php",
    "/vul/sqli/sqli_blind_b.php",
    "/vul/sqli/sqli_blind_t.php",
    "/vul/urlredirect/urlredirect.php",
})
SAFE_INVENTORY_PATHS = frozenset({
    "/",
    "/vul/xss/xss_reflected_get.php",
    "/vul/xss/xss_dom.php",
    "/vul/xss/xss_stored.php",
    "/vul/sqli/sqli_str.php",
    "/vul/sqli/sqli_search.php",
    "/vul/sqli/sqli_blind_b.php",
    "/vul/sqli/sqli_blind_t.php",
    "/vul/urlredirect/urlredirect.php",
    "/vul/ssrf/ssrf_curl.php",
    "/vul/fileinclude/fi_local.php",
    "/vul/dir/dir_list.php",
    "/vul/infoleak/findabc.php",
    "/vul/xxe/xxe_1.php",
    "/vul/rce/rce_ping.php",
    "/vul/unsafeupload/upload.php",
})
ALLOWED_QUERY_NAMES = frozenset({"message", "text", "name", "submit"})
ALLOWED_FAMILIES = frozenset({"xss", "injection", "access_control", "url_redirect", "logic"})
PAIR_VARIANTS = frozenset({"plain", "url_percent", "html_entity", "double_html_entity"})
SQL_ERROR_PATTERNS = (
    "you have an error in your sql syntax",
    "warning: mysql",
    "mysql_fetch",
    "mysql server version",
    "sql syntax",
    "sqlite error",
    "odbc",
)


class _HTMLShapeParser(HTMLParser):
    """Count bounded HTML structure without retaining source text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags = 0
        self.forms = 0
        self.inputs = 0
        self.scripts = 0
        self.title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags += 1
        lower = tag.casefold()
        if lower == "form":
            self.forms += 1
        elif lower in {"input", "textarea", "select", "button"}:
            self.inputs += 1
        elif lower == "script":
            self.scripts += 1
        elif lower == "title":
            self.title = True

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)


def _assert_pikachu_target(value: str) -> str:
    try:
        parsed = urlsplit(str(value))
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Pikachu target must be exactly http://127.0.0.1:8766") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or port not in {8766, 8767, 8768}
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Pikachu target must be exactly http://127.0.0.1:8766, :8767, or :8768")
    return f"http://127.0.0.1:{port}"


def _safe_params(raw: Any) -> dict[str, str]:
    params = dict(raw or {})
    if len(params) > 8:
        raise ValueError("Pikachu probe query has too many parameters")
    normalized: dict[str, str] = {}
    for key, value in params.items():
        name = str(key)
        if name.casefold() not in ALLOWED_QUERY_NAMES:
            raise ValueError(f"Pikachu probe query field is not allow-listed: {name}")
        text = str(value)
        if len(text) > 2048:
            raise ValueError("Pikachu probe query value is too large")
        if any(secret in name.casefold() for secret in ("password", "token", "cookie", "authorization")):
            raise ValueError("Pikachu probe query may not contain credentials")
        normalized[name] = text
    return normalized


def _validate_pair_metadata(raw: Any, *, family: str, surface: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("Pikachu pair metadata must be an object")
    pair_id = str(raw.get("pair_id", ""))
    variant = str(raw.get("variant", ""))
    surface_role = str(raw.get("surface_role", surface))
    depth = raw.get("encoding_depth")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{4,96}", pair_id):
        raise ValueError("Pikachu pair_id is invalid")
    if variant not in PAIR_VARIANTS:
        raise ValueError("Pikachu pair variant is not supported")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", surface_role):
        raise ValueError("Pikachu pair surface_role is invalid")
    if isinstance(depth, bool) or not isinstance(depth, int) or not 0 <= depth <= 3:
        raise ValueError("Pikachu pair encoding_depth must be an integer from 0 to 3")
    return {
        "pair_id": pair_id,
        "variant": variant,
        "surface_role": surface_role,
        "encoding_depth": depth,
    }


def validate_pikachu_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate one safe, one-at-a-time Pikachu GET probe specification."""

    if not isinstance(spec, dict):
        raise ValueError("Pikachu probe spec must be an object")
    target = _assert_pikachu_target(str(spec.get("target", PIKACHU_BASE_URL)))
    method = str(spec.get("method", "GET")).upper()
    if method != "GET":
        raise ValueError("staged Pikachu collector permits only read-only GET probes")
    path = str(spec.get("path", ""))
    if path not in SAFE_PROBE_PATHS or "://" in path or path.startswith("//"):
        raise ValueError("Pikachu path is not an allow-listed read-only probe surface")
    params = _safe_params(spec.get("params"))
    probe_kind = str(spec.get("probe_kind", "http_canary"))
    if probe_kind not in ALLOWED_PROBE_KINDS:
        raise ValueError("unknown Pikachu probe kind")
    marker = str(spec.get("marker", "pk-safe-probe"))
    probe = str(spec.get("probe", marker))
    payload = build_detection_payload(
        target=target,
        method="GET",
        path=path,
        marker=marker,
        probe_kind=probe_kind,
        probe=probe,
        expected=dict(spec.get("expected") or {}),
    )
    family = str(spec.get("family", ""))
    if family not in ALLOWED_FAMILIES:
        raise ValueError("Pikachu probe family is not supported")
    surface = str(spec.get("surface", path.strip("/").replace("/", "_")))
    pair = _validate_pair_metadata(spec.get("pair"), family=family, surface=surface)
    source_id = str(spec.get("source_id", ""))
    if not source_id or len(source_id) > 96:
        raise ValueError("Pikachu probe source_id is required")
    expected_oracle = str(spec.get("expected_oracle", "pikachu_bounded_http_projection_v1"))
    if expected_oracle != "pikachu_bounded_http_projection_v1":
        raise ValueError("Pikachu probes require the bounded HTTP projection oracle")
    normalized = {
        "schema_version": PIKACHU_SPEC_SCHEMA,
        "target": target,
        "method": method,
        "path": path,
        "params": params,
        "marker": marker,
        "probe_kind": probe_kind,
        "probe": probe,
        "encoding": str(spec.get("encoding", "query_percent_encoding")),
        "expected": json.loads(json.dumps(dict(spec.get("expected") or {}), ensure_ascii=False)),
        "family": family,
        "source_id": source_id,
        "lab_id": str(spec.get("lab_id", "pikachu-staged")),
        "surface": surface,
        "expected_oracle": expected_oracle,
        "expected_signal": str(spec.get("expected_signal", "bounded_response_projection")),
        "payload": payload,
    }
    if pair is not None:
        normalized["pair"] = pair
    return normalized


def _response_summary(response: httpx.Response) -> dict[str, Any]:
    body = response.content
    return {
        "status_code": int(response.status_code),
        "headers": {
            key: str(response.headers[key])
            for key in ("content-type", "content-length")
            if key in response.headers
        },
        "body_length": len(body),
        "body_sha256": hashlib.sha256(body).hexdigest(),
    }


def _same_origin_location(value: str) -> tuple[bool, bool]:
    """Return (has_location, is_external) without retaining its value."""

    if not value:
        return False, False
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc:
        return True, not (parsed.scheme == "http" and parsed.hostname == "127.0.0.1" and parsed.port in {None, 8766})
    return True, value.startswith("//")


def _bounded_projection(response: httpx.Response, baseline: httpx.Response, marker: str) -> dict[str, Any]:
    """Build a structural projection; response text is discarded afterwards."""

    body = response.content
    baseline_body = baseline.content
    text = body.decode("utf-8", errors="replace")
    lowered = text.casefold()
    parser = _HTMLShapeParser()
    try:
        parser.feed(text)
        parser.close()
    except (TypeError, ValueError):
        pass
    marker_pattern = re.escape(marker)
    marker_reflected = marker in text
    marker_in_html_text = bool(re.search(r">[^<]{0,512}" + marker_pattern + r"[^<]{0,512}<", text))
    marker_in_attribute = bool(re.search(r"(?:=|['\"])[^<>]{0,512}" + marker_pattern, text))
    marker_in_script = bool(re.search(r"<script\b[^>]*>[^<]{0,2048}" + marker_pattern, text, re.IGNORECASE | re.DOTALL))
    sql_error_shape = any(pattern in lowered for pattern in SQL_ERROR_PATTERNS)
    has_location, external_redirect = _same_origin_location(str(response.headers.get("location", "")))
    return {
        "marker_reflected": marker_reflected,
        "marker_count": min(text.count(marker), 8),
        "marker_in_html_text": marker_in_html_text,
        "marker_in_attribute": marker_in_attribute,
        "marker_in_script_source": marker_in_script,
        "sql_error_shape": sql_error_shape,
        "status_changed": response.status_code != baseline.status_code,
        "body_length_delta": len(body) - len(baseline_body),
        "body_length_delta_abs": abs(len(body) - len(baseline_body)),
        "redirect_present": has_location,
        "external_redirect": external_redirect,
        "html_tag_count": min(parser.tags, 512),
        "form_count": min(parser.forms, 64),
        "input_count": min(parser.inputs, 128),
        "script_count": min(parser.scripts, 64),
        "title_present": parser.title,
        "content_type_class": "html" if "html" in response.headers.get("content-type", "").casefold() else "other",
    }


def _rule_ir_for(spec: dict[str, Any]) -> dict[str, Any]:
    if spec["family"] == "xss":
        return {
            "op": "or",
            "args": [
                {"op": "eq", "left": {"op": "field", "path": "oracle_projection.marker_reflected"}, "right": {"op": "const", "value": True}},
                {"op": "eq", "left": {"op": "field", "path": "oracle_projection.marker_in_attribute"}, "right": {"op": "const", "value": True}},
                {"op": "eq", "left": {"op": "field", "path": "oracle_projection.marker_in_script_source"}, "right": {"op": "const", "value": True}},
            ],
        }
    if spec["family"] == "injection":
        return {
            "op": "or",
            "args": [
                {"op": "eq", "left": {"op": "field", "path": "oracle_projection.sql_error_shape"}, "right": {"op": "const", "value": True}},
                {"op": "ge", "left": {"op": "field", "path": "oracle_projection.body_length_delta_abs"}, "right": {"op": "const", "value": 256}},
            ],
        }
    if spec["family"] == "url_redirect":
        return {"op": "eq", "left": {"op": "field", "path": "oracle_projection.external_redirect"}, "right": {"op": "const", "value": True}}
    return {"op": "eq", "left": {"op": "field", "path": "response.status_code"}, "right": {"op": "const", "value": 200}}


def default_pikachu_probe_specs(marker: str = "pk-safe-probe-a1") -> list[dict[str, Any]]:
    """Create the first safe stage: one inert canary per approved surface."""

    return [
        {
            "source_id": "pikachu-local-pg03",
            "lab_id": "xss-reflected-get-safe-canary",
            "family": "xss",
            "surface": "xss_reflected_get",
            "path": "/vul/xss/xss_reflected_get.php",
            "params": {"message": marker, "submit": "submit"},
            "probe_kind": "http_canary",
            "marker": marker,
            "probe": marker,
            "expected_signal": "marker_reflection_only",
        },
        {
            "source_id": "pikachu-local-pg03",
            "lab_id": "xss-dom-safe-canary",
            "family": "xss",
            "surface": "xss_dom_value_source",
            "path": "/vul/xss/xss_dom.php",
            "params": {"text": marker},
            "probe_kind": "inert_dom_markup",
            "marker": marker,
            "probe": f'<span data-sift-marker="{marker}">x</span>',
            "expected_signal": "dom_source_reflection_without_execution",
        },
        {
            "source_id": "pikachu-local-pg03",
            "lab_id": "urlredirect-safe-baseline",
            "family": "url_redirect",
            "surface": "url_redirect_response",
            "path": "/vul/urlredirect/urlredirect.php",
            "params": {},
            "probe_kind": "http_canary",
            "marker": marker,
            "probe": marker,
            "expected_signal": "no_external_redirect",
        },
    ] + [
        {
            "source_id": "pikachu-local-pg03",
            "lab_id": f"sql-{surface}-safe-canary",
            "family": "injection",
            "surface": surface,
            "path": path,
            "params": {"name": marker, "submit": submit},
            "probe_kind": "sql_channel_class",
            "marker": marker,
            "probe": "plain",
            "expected_signal": "bounded_sql_error_or_shape_delta",
        }
        for surface, path, submit in (
            ("sqli_str", "/vul/sqli/sqli_str.php", "查询"),
            ("sqli_search", "/vul/sqli/sqli_search.php", "搜索"),
            ("sqli_blind_boolean", "/vul/sqli/sqli_blind_b.php", "查询"),
            ("sqli_blind_time", "/vul/sqli/sqli_blind_t.php", "查询"),
        )
    ]


def _pair_encoding(value: str, variant: str) -> str:
    """Encode only inert markup/identifier characters for pair training."""

    if variant == "plain":
        return value
    if variant == "url_percent":
        return quote(value, safe="")
    if variant == "html_entity":
        return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&#34;").replace(".", "&#46;")
    if variant == "double_html_entity":
        once = _pair_encoding(value, "html_entity")
        return once.replace("&", "&amp;")
    raise ValueError(f"unsupported pair encoding variant: {variant}")


def default_pikachu_paired_specs(marker_prefix: str = "pk-pair") -> list[dict[str, Any]]:
    """Build cross-surface, multi-encoding safe pairs for invariance training."""

    variants = (
        ("plain", 0),
        ("url_percent", 1),
        ("html_entity", 1),
        ("double_html_entity", 2),
    )
    specs: list[dict[str, Any]] = []
    xss_pair_id = f"{marker_prefix}-family-a-01"
    # Marker text is intentionally family-neutral; otherwise the decoder can
    # memorize a label from the canary instead of learning the response.
    xss_marker = f"{marker_prefix}-m01"
    xss_markup = f'<span data-sift-marker="{xss_marker}">x</span>'
    for path, field, surface_role in (
        ("/vul/xss/xss_reflected_get.php", "message", "reflected_get"),
        ("/vul/xss/xss_dom.php", "text", "dom_value_source"),
    ):
        for variant, depth in variants:
            value = _pair_encoding(xss_markup, variant)
            probe_kind = "inert_dom_markup" if variant == "plain" else "encoded_dom_markup"
            specs.append({
                "source_id": "pikachu-pair-pg04",
                "lab_id": f"{xss_pair_id}-{surface_role}-{variant}",
                "family": "xss",
                "surface": f"xss_{surface_role}",
                "path": path,
                "params": {field: value, **({"submit": "submit"} if field == "message" else {})},
                "probe_kind": probe_kind,
                "marker": xss_marker,
                "probe": value,
                "encoding": variant,
                "expected_signal": "marker_reflection_only",
                "pair": {
                    "pair_id": xss_pair_id,
                    "variant": variant,
                    "surface_role": surface_role,
                    "encoding_depth": depth,
                },
            })

    sql_pair_id = f"{marker_prefix}-family-b-01"
    sql_marker = f"{marker_prefix}-m02"
    sql_value = f"{sql_marker}.tag"
    for path, surface_role, submit in (
        ("/vul/sqli/sqli_str.php", "sqli_str", "查询"),
        ("/vul/sqli/sqli_search.php", "sqli_search", "搜索"),
        ("/vul/sqli/sqli_blind_b.php", "sqli_blind_boolean", "查询"),
        ("/vul/sqli/sqli_blind_t.php", "sqli_blind_time", "查询"),
    ):
        for variant, depth in variants:
            value = _pair_encoding(sql_value, variant)
            specs.append({
                "source_id": "pikachu-pair-pg04",
                "lab_id": f"{sql_pair_id}-{surface_role}-{variant}",
                "family": "injection",
                "surface": surface_role,
                "path": path,
                "params": {"name": value, "submit": submit},
                "probe_kind": "sql_channel_class",
                "marker": sql_marker,
                "probe": "plain",
                "encoding": variant,
                "expected_signal": "bounded_sql_error_or_shape_delta",
                "pair": {
                    "pair_id": sql_pair_id,
                    "variant": variant,
                    "surface_role": surface_role,
                    "encoding_depth": depth,
                },
            })
    return specs


def default_pikachu_counterfactual_specs(marker_prefix: str = "pk-cf") -> list[dict[str, Any]]:
    """Build local negative controls by substituting a non-matching marker.

    The request still carries an inert identifier, but the oracle marker is a
    different identifier.  A response may reflect the input and still must
    not be counted as evidence for this control.  This creates a real local
    no-signal observation rather than a synthetic label flip.
    """

    expected_marker = f"{marker_prefix}-expected"
    input_marker = f"{marker_prefix}-input"
    specs: list[dict[str, Any]] = []
    for variant in ("plain", "url_percent"):
        for path, field, surface_role in (
            ("/vul/xss/xss_reflected_get.php", "message", "reflected_get"),
            ("/vul/xss/xss_dom.php", "text", "dom_value_source"),
        ):
            raw_value = input_marker
            if surface_role == "dom_value_source":
                raw_value = f'<span data-sift-marker="{input_marker}">x</span>'
            value = _pair_encoding(raw_value, variant)
            specs.append({
                "source_id": "pikachu-counterfactual-pg05",
                "lab_id": f"negative-{surface_role}-marker-substitution-{variant}",
                "family": "xss",
                "surface": f"xss_{surface_role}",
                "path": path,
                "params": {field: value, **({"submit": "submit"} if field == "message" else {})},
                "probe_kind": "inert_dom_markup" if surface_role == "dom_value_source" else "http_canary",
                "marker": expected_marker,
                "probe": value,
                "encoding": f"counterfactual_marker_substitution_{variant}",
                "expected_signal": "no_matching_marker_signal",
            })
        for path, surface_role, submit in (
            ("/vul/sqli/sqli_str.php", "sqli_str", "查询"),
            ("/vul/sqli/sqli_search.php", "sqli_search", "搜索"),
            ("/vul/sqli/sqli_blind_b.php", "sqli_blind_boolean", "查询"),
            ("/vul/sqli/sqli_blind_t.php", "sqli_blind_time", "查询"),
        ):
            value = _pair_encoding(f"{input_marker}.tag", variant)
            specs.append({
                "source_id": "pikachu-counterfactual-pg05",
                "lab_id": f"negative-{surface_role}-marker-substitution-{variant}",
                "family": "injection",
                "surface": surface_role,
                "path": path,
                "params": {"name": value, "submit": submit},
                "probe_kind": "sql_channel_class",
                "marker": expected_marker,
                "probe": "plain",
                "encoding": f"counterfactual_marker_substitution_{variant}",
                "expected_signal": "no_matching_marker_or_sql_error_signal",
            })
    return specs


class PikachuReplayCollector:
    """Sequentially replay safe specs against an explicitly allow-listed local container."""

    def __init__(
        self,
        *,
        base_url: str = PIKACHU_BASE_URL,
        timeout_seconds: float = 5.0,
        target_instance_id: str = "unattested",
        fresh_target: bool = False,
    ) -> None:
        self.base_url = _assert_pikachu_target(base_url)
        self.timeout_seconds = min(max(float(timeout_seconds), 0.5), 5.0)
        self.target_instance_id = str(target_instance_id)[:128]
        self.fresh_target = bool(fresh_target)
        self.sequence = 0

    def _read_only_reset(self, spec: dict[str, Any]) -> dict[str, Any]:
        self.sequence += 1
        reset_id = hashlib.sha256(
            f"{spec['source_id']}:{spec['lab_id']}:{self.sequence}:{PIKACHU_IMAGE_DIGEST}".encode("utf-8")
        ).hexdigest()[:20]
        return {
            "kind": "pinned_container_read_only",
            "reset_id": reset_id,
            "sequence": self.sequence,
            "fresh": self.fresh_target,
            "fresh_target": self.fresh_target,
            "target_instance_id": self.target_instance_id,
            "state_change_allowed": False,
            "evaluator_state_hidden": True,
            "external_network": False,
            "container_image_digest": PIKACHU_IMAGE_DIGEST,
        }

    async def collect(self, raw_spec: dict[str, Any]) -> dict[str, Any]:
        spec = validate_pikachu_spec(raw_spec)
        reset = self._read_only_reset(spec)
        headers = dict(spec["payload"]["headers"])
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            follow_redirects=False,
            cookies={},
        ) as client:
            # Baseline and probe are intentionally adjacent and sequential;
            # this is the first step of the active-probe/belief-update loop.
            baseline = await client.get(spec["path"], headers=headers)
            response = await client.get(spec["path"], params=spec["params"], headers=headers)
        response_summary = _response_summary(response)
        baseline_summary = _response_summary(baseline)
        projection = _bounded_projection(response, baseline, spec["marker"])
        envelope = {
            "collector": PIKACHU_COLLECTOR_SCHEMA,
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
        rule_ir = _rule_ir_for(spec)
        rule_ir_canonical = canonical_rule_ir(rule_ir)
        rule_ir_result = bool(evaluate_rule_ir(rule_ir, {"response": response_summary, "oracle_projection": projection}))
        record = {
            "schema_version": PIKACHU_COLLECTOR_SCHEMA,
            "sample_id": f"{spec['source_id']}-{spec['lab_id']}-{spec['payload']['payload_sha256'][:12]}",
            "source_id": spec["source_id"],
            "lab_id": spec["lab_id"],
            "family": spec["family"],
            "payload": spec["payload"],
            "probe_artifact": {
                "original": spec["probe"],
                "encoding": spec["encoding"],
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
                "fresh_target": self.fresh_target,
                "target_instance_id": self.target_instance_id,
                "transport": "httpx_loopback",
            },
            "response_projection": response_summary,
            "oracle_projection": projection,
            "evidence": checked["body"],
            "rule_ir": rule_ir,
            "rule_ir_canonical": rule_ir_canonical,
            "rule_ir_complexity": rule_ir_complexity(rule_ir),
            "rule_ir_result": rule_ir_result,
            "candidate_status": "suspicious_surface_signal" if rule_ir_result else "clean_observation",
            "safety": {
                "local_only": True,
                "read_only": True,
                "fresh_reset": False,
                "fresh_target": self.fresh_target,
                "target_instance_id": self.target_instance_id,
                "external_network": False,
                "script_execution": False,
                "database_touched": False,
                "real_sleep_performed": False,
                "raw_body_stored": False,
                "credentials_stored": False,
            },
        }
        if spec.get("pair") is not None:
            record["pair"] = dict(spec["pair"])
        return record

    async def collect_many(self, specs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for spec in specs:
            records.append(await self.collect(spec))
        return records


__all__ = [
    "PIKACHU_BASE_URL",
    "PIKACHU_FRESH_BASE_URL",
    "PIKACHU_FRESH_BASE_URL_2",
    "PIKACHU_COLLECTOR_SCHEMA",
    "PIKACHU_IMAGE_DIGEST",
    "SAFE_INVENTORY_PATHS",
    "SAFE_PROBE_PATHS",
    "PikachuReplayCollector",
    "default_pikachu_probe_specs",
    "default_pikachu_paired_specs",
    "default_pikachu_counterfactual_specs",
    "validate_pikachu_spec",
]
