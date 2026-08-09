"""Bounded parameter-surface projection helpers for PG-224.

The browser crawl supplies route and field names.  This module decides whether
an inert local canary is safe to send, creates values only for the short-lived
request, and stores only response projections/wire placeholders.  It is
deliberately not a generic exploit builder.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import httpx

from .pg212_sql_response_oracle import build_sql_probe_values
from .pg179b_iterative_probe import validate_marker


PG224_SCHEMA = "pg224-pikachu-parameter-surface-projection-v1"
SAFE_SQL_PATHS = frozenset({
    "/vul/sqli/sqli_blind_b.php",
    "/vul/sqli/sqli_blind_t.php",
    "/vul/sqli/sqli_id.php",
    "/vul/sqli/sqli_search.php",
    "/vul/sqli/sqli_str.php",
    "/vul/sqli/sqli_widebyte.php",
    "/vul/sqli/sqli_x.php",
    "/vul/sqli/sqli_header/sqli_header_login.php",
})
SAFE_XSS_GET_PATHS = frozenset({
    "/vul/xss/xss_01.php",
    "/vul/xss/xss_02.php",
    "/vul/xss/xss_03.php",
    "/vul/xss/xss_04.php",
    "/vul/xss/xss_dom_x.php",
    "/vul/xss/xss_reflected_get.php",
})
SAFE_POST_PATHS = frozenset({
    "/vul/sqli/sqli_id.php",
    "/vul/sqli/sqli_widebyte.php",
})
BLOCKED_PATH_MARKERS = (
    "/rce/",
    "/ssrf/",
    "/xxe/",
    "/unsafeupload/",
    "/unsafedownload/",
    "/unserilization/",
    "/fileinclude/",
    "/burteforce/",
    "/csrf/",
    "/overpermission/",
)
BLOCKED_STATEFUL_PATHS = frozenset({
    "/vul/sqli/sqli_del.php",
    "/vul/xss/xss_stored.php",
    "/vul/xss/xssblind/xss_blind.php",
    "/vul/xss/xsspost/post_login.php",
})


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def route_policy(path: str, method: str, fields: Sequence[str]) -> dict[str, Any]:
    path = str(path)
    method = str(method).upper()
    names = [str(item) for item in fields]
    if any(name.casefold() in {"password", "token", "vcode", "csrf", "uploadfile", "o"} for name in names):
        return {"send_allowed": False, "reason": "credential_or_file_or_object_field", "oracle": "abstain"}
    if path in BLOCKED_STATEFUL_PATHS:
        return {"send_allowed": False, "reason": "stateful_surface_not_mutated", "oracle": "abstain"}
    if any(marker in path.casefold() for marker in BLOCKED_PATH_MARKERS):
        return {"send_allowed": False, "reason": "unsafe_family_preflight_only", "oracle": "abstain"}
    if method == "POST" and path not in SAFE_POST_PATHS:
        return {"send_allowed": False, "reason": "post_surface_not_read_only_allowlisted", "oracle": "abstain"}
    if method == "GET" and not (path in SAFE_SQL_PATHS or path in SAFE_XSS_GET_PATHS or "/dir/" in path or path == "/vul/urlredirect/urlredirect.php"):
        return {"send_allowed": False, "reason": "get_surface_not_read_only_allowlisted", "oracle": "abstain"}
    family = "injection" if path in SAFE_SQL_PATHS else "xss" if path in SAFE_XSS_GET_PATHS else "url_redirect" if "urlredirect" in path else "ordinary"
    return {"send_allowed": True, "reason": "bounded_read_only_canary", "oracle": "projection_only", "family": family}


def build_runtime_values(*, path: str, method: str, fields: Sequence[str], marker: str, probe_kind: str, control: bool = False) -> dict[str, str]:
    marker = validate_marker(marker)
    names = [str(item) for item in fields]
    if path in SAFE_SQL_PATHS:
        return build_sql_probe_values(field_names=names, marker=marker, probe_class="control" if control else "syntax_shape")
    values: dict[str, str] = {}
    for field in names:
        if field.casefold() == "submit":
            values[field] = "submit"
        else:
            values[field] = marker if control else f"{marker}-value"
    return values


def wire_placeholder(*, path: str, method: str, fields: Sequence[str], probe_kind: str) -> str:
    names = [str(item) for item in fields]
    values = []
    for field in names:
        if field.casefold() == "submit":
            value = "submit"
        elif path in SAFE_SQL_PATHS and probe_kind in {"sql_channel_class", "sql_fragment_class"}:
            value = "<RUNTIME_SQL_SHAPE>"
        else:
            value = "<RUNTIME_CANARY>"
        values.append(f"{field}={value}")
    if str(method).upper() == "GET":
        return f"GET <LOOPBACK_ORIGIN>{path}?{'&'.join(values)}"
    return f"POST <LOOPBACK_ORIGIN>{path}\\nContent-Type: application/x-www-form-urlencoded\\n\\n{'&'.join(values)}"


def project_response(response: httpx.Response, *, marker: str, baseline: Mapping[str, Any] | None = None) -> dict[str, Any]:
    marker = validate_marker(marker)
    body = bytes(response.content or b"")
    text = response.text.casefold()
    location = str(response.headers.get("location", ""))
    parsed = urlsplit(location) if location else None
    external_redirect = bool(parsed and parsed.hostname and parsed.hostname not in {"127.0.0.1", "localhost"})
    projection = {
        "status_code": int(response.status_code),
        "status_class": f"{int(response.status_code) // 100}xx",
        "content_type_class": "html" if "html" in str(response.headers.get("content-type", "")).casefold() else "other",
        "body_length_bucket": "0" if not body else "1-255" if len(body) <= 255 else "256-4095" if len(body) <= 4095 else "4096-65535" if len(body) <= 65535 else "65536+",
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "marker_reflected": marker.casefold() in text,
        "location_present": bool(location),
        "external_redirect": external_redirect,
        "same_origin_redirect": bool(location) and not external_redirect,
        "transport_error": False,
    }
    if baseline:
        projection["status_changed"] = projection["status_class"] != str(baseline.get("status_class"))
        projection["body_length_bucket_changed"] = projection["body_length_bucket"] != str(baseline.get("body_length_bucket"))
    else:
        projection["status_changed"] = False
        projection["body_length_bucket_changed"] = False
    projection["projection_sha256"] = sha256_json(projection)
    return {"schema_version": PG224_SCHEMA, "response_projection": projection, "raw_response_retained": False}


__all__ = ["BLOCKED_PATH_MARKERS", "PG224_SCHEMA", "SAFE_POST_PATHS", "SAFE_SQL_PATHS", "SAFE_XSS_GET_PATHS", "build_runtime_values", "project_response", "route_policy", "sha256_json", "wire_placeholder"]
