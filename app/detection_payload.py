"""Safe, detection-only payload manifests for authorized local network probes.

The AI may describe a benign canary request and its expected evidence.  This
module deliberately does not execute requests or construct exploit strings.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any
from urllib.parse import urlsplit


PAYLOAD_SCHEMA = "sift-detection-payload-v1"
ALLOWED_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "POST"})
ALLOWED_HEADER_NAMES = frozenset({"accept", "content-type", "cache-control", "x-sift-probe"})
ALLOWED_PROBE_KINDS = frozenset({
    "http_canary",
    "header_canary",
    "inert_dom_markup",
    "encoded_dom_markup",
    "sql_fragment_class",
    "sql_channel_class",
})
SAFE_SQL_FRAGMENT_CLASSES = frozenset({
    "plain",
    "quoted_value",
    "operator_like",
    "comment_like",
    "subquery_like",
    "blind_boolean",
    "row_shape",
    "syntax_error",
    "time_delay",
    "local_side_channel",
})
FORBIDDEN_MARKERS = (
    "<script",
    "javascript:",
    "union select",
    "drop table",
    ";--",
    "../",
    "powershell",
    "cmd.exe",
    "curl ",
    "wget ",
    "onerror=",
    "onload=",
    "onanimation",
)
MARKER_RE = re.compile(r"^[A-Za-z0-9._-]{4,64}$")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def payload_digest(payload: dict[str, Any]) -> str:
    body = dict(payload)
    body.pop("payload_sha256", None)
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


def _assert_local_base_url(value: str) -> str:
    parsed = urlsplit(str(value))
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("detection payload target must be an authorized local host")
    if parsed.username or parsed.password:
        raise ValueError("detection payload may not contain credentials")
    return str(value).rstrip("/")


def _assert_safe_text(value: str, label: str) -> str:
    normalized = str(value)
    lowered = normalized.casefold()
    if any(token in lowered for token in FORBIDDEN_MARKERS):
        raise ValueError(f"{label} contains a non-detection or destructive marker")
    return normalized


def validate_detection_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a non-executing detection payload manifest."""

    if not isinstance(payload, dict):
        raise ValueError("detection payload must be an object")
    target = _assert_local_base_url(str(payload.get("target", "http://127.0.0.1:3100")))
    method = str(payload.get("method", "GET")).upper()
    if method not in ALLOWED_METHODS:
        raise ValueError("detection payload permits only GET, HEAD, OPTIONS, or safe POST")
    path = str(payload.get("path", ""))
    if not path.startswith("/") or path.startswith("//") or "://" in path:
        raise ValueError("detection payload path must be origin-relative")
    if any(part in path.casefold() for part in ("/api/challenges", "/snippets")):
        raise ValueError("evaluator-only paths are not valid detection payload targets")
    headers: dict[str, str] = {}
    for key, value in dict(payload.get("headers") or {}).items():
        name = str(key).casefold()
        if name not in ALLOWED_HEADER_NAMES:
            raise ValueError(f"header is not permitted for detection payloads: {key}")
        headers[name] = _assert_safe_text(str(value), f"header {key}")
    raw_form = payload.get("form") or {}
    if method == "POST":
        if not isinstance(raw_form, dict) or not raw_form:
            raise ValueError("safe POST detection payload requires a non-empty form")
    elif raw_form:
        raise ValueError("form data is only valid for safe POST detection payloads")
    form: dict[str, str] = {}
    for key, value in dict(raw_form).items():
        name = str(key)
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", name):
            raise ValueError("safe POST form field name is invalid")
        if name.casefold() in {"password", "passwd", "secret", "token", "csrf", "cookie", "authorization", "file", "upload"}:
            raise ValueError("safe POST form may not contain credentials, tokens, or files")
        normalized_value = _assert_safe_text(str(value), f"form field {name}")
        if len(normalized_value) > 2048:
            raise ValueError("safe POST form value is too large")
        form[name] = normalized_value
    marker = str(payload.get("marker", "sift-probe"))
    if not MARKER_RE.fullmatch(marker) or any(token in marker.casefold() for token in FORBIDDEN_MARKERS):
        raise ValueError("marker must be a short inert identifier")
    expected = payload.get("expected") or {}
    if not isinstance(expected, dict):
        raise ValueError("expected detection evidence must be an object")
    expected = json.loads(_canonical(expected))
    probe_kind = str(payload.get("probe_kind", "http_canary"))
    if probe_kind not in ALLOWED_PROBE_KINDS:
        raise ValueError("unknown detection probe kind")
    raw_probe = payload.get("probe")
    probe = _assert_safe_text(marker if raw_probe is None else str(raw_probe), "probe")
    if len(probe) > 2048:
        raise ValueError("detection probe is too large")
    if probe_kind == "sql_channel_class" and probe not in SAFE_SQL_FRAGMENT_CLASSES:
        raise ValueError("SQL channel probe must use a known abstract fragment class")
    if probe_kind == "sql_fragment_class" and probe not in SAFE_SQL_FRAGMENT_CLASSES:
        raise ValueError("SQL fragment probe must use a known abstract fragment class")
    normalized = {
        "schema_version": PAYLOAD_SCHEMA,
        "target": target,
        "method": method,
        "path": _assert_safe_text(path, "path"),
        "headers": headers,
        "marker": marker,
        "probe_kind": probe_kind,
        "probe": probe,
        "expected": expected,
        "safety": {
            "authorized_scope": "local_only",
            "non_destructive": True,
            "no_script_execution": True,
            "no_database_write": True,
            "no_credential_access": True,
            "no_data_exfiltration": True,
            "no_external_network": True,
            "does_not_execute": True,
        },
    }
    # Keep the original GET/HEAD/OPTIONS canonical shape for catalog hash
    # compatibility.  POST manifests include their validated form explicitly.
    if method == "POST":
        normalized["form"] = form
    normalized["payload_sha256"] = payload_digest(normalized)
    return normalized


def build_detection_payload(
    *,
    path: str,
    expected: dict[str, Any] | None = None,
    marker: str = "sift-probe",
    target: str = "http://127.0.0.1:3100",
    method: str = "GET",
    headers: dict[str, str] | None = None,
    probe: str | None = None,
    probe_kind: str = "http_canary",
    form: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a validated manifest; no network operation is performed."""

    return validate_detection_payload({
        "target": target,
        "method": method,
        "path": path,
        "headers": headers or {"accept": "application/json", "x-sift-probe": marker},
        "marker": marker,
        "probe": marker if probe is None else probe,
        "probe_kind": probe_kind,
        "form": form or {},
        "expected": expected or {},
    })
