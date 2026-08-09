"""PG-195 request-surface adapter for bounded GET/POST replay.

The adapter is deliberately narrower than a generic HTTP client.  It accepts
only route/field metadata observed by the browser crawl, creates a runtime
inert marker, and returns projections plus a typed no-JavaScript DOM effect.
Request values and response text never enter the persisted record.
"""

from __future__ import annotations

import hashlib
from html import escape
from typing import Any, Mapping

import httpx

from .cross_lab_safe_catalog import sha256_json, validate_payload_manifest
from .pg179b_iterative_probe import _summary, validate_marker
from .pg193_browser_dom_oracle import run_browser_dom_oracle
from .pg185_pikachu_dom_adapter import inert_dom_probe


SCHEMA_VERSION = "sift-pg195-request-surface-adapter-v1"
_ALLOWED_METHODS = frozenset({"GET", "POST"})
_ALLOWED_ENCODINGS = frozenset({"identity", "url_percent", "html_entity", "json_string"})
_FORBIDDEN_FIELDS = frozenset({"password", "passwd", "secret", "token", "csrf", "cookie", "authorization", "file", "upload"})
_VALUE_FIELDS = frozenset({"message", "text", "content", "name", "username", "id", "title", "url", "submit"})


def _digest(value: Any) -> str:
    return hashlib.sha256(sha256_json(value).encode("utf-8")).hexdigest()


def build_surface_action_manifest(
    *,
    path: str,
    method: str,
    surface: str,
    field_names: list[str],
    probe_role: str,
    marker: str,
    encoding: str = "identity",
) -> dict[str, Any]:
    """Create a hash-bound GET or safe POST manifest from observed fields."""

    method = str(method).upper()
    if method not in _ALLOWED_METHODS:
        raise ValueError("PG-195 method must be GET or POST")
    if not path.startswith("/") or path.startswith("//") or "://" in path:
        raise ValueError("PG-195 path must be origin-relative")
    if probe_role not in {"control", "candidate"}:
        raise ValueError("PG-195 role must be control or candidate")
    marker = validate_marker(marker)
    encoding = str(encoding)
    if encoding not in _ALLOWED_ENCODINGS:
        raise ValueError("PG-195 encoding is not allow-listed")
    fields = sorted({str(item) for item in field_names if str(item)})
    if not fields or len(fields) > 16:
        raise ValueError("PG-195 requires bounded observed fields")
    if any(field.casefold() in _FORBIDDEN_FIELDS for field in fields):
        raise ValueError("PG-195 refuses credential or secret fields")
    marker_hash = hashlib.sha256(marker.encode("utf-8")).hexdigest()
    body = {
        "path": path,
        "method": method,
        "surface": surface,
        "fields": fields,
        "role": probe_role,
        "marker_sha256": marker_hash,
        "encoding": encoding,
    }
    manifest: dict[str, Any] = {
        "manifest_id": f"pg195-{surface}-{method.casefold()}-{probe_role}",
        "payload_sha256": sha256_json(body),
        "probe_ref": f"pg195-inert-surface-{surface}-{probe_role}",
        "probe_kind": "encoded_dom_markup" if encoding != "identity" else "inert_dom_markup",
        "route_template_id": f"pikachu-{surface}",
        "method": method,
        "placement": "query" if method == "GET" else "form",
        "encoding_chain": [encoding],
        "encoding_depth": int(encoding != "identity"),
        "marker_sha256": marker_hash,
        "max_bytes": 768,
        "safety": {
            "does_not_execute": True,
            "no_external_network": True,
            "no_script_execution": True,
            "no_database_write": True,
            "no_credential_access": True,
        },
    }
    if method == "POST":
        manifest["form_field_names"] = fields
        manifest["form_content_type"] = "application/x-www-form-urlencoded"
    return validate_payload_manifest(manifest)


def build_surface_values(*, field_names: list[str], probe_role: str, marker: str, encoding: str = "identity") -> dict[str, str]:
    """Build runtime-only values; no credential field is accepted."""

    marker = validate_marker(marker)
    if probe_role not in {"control", "candidate"}:
        raise ValueError("PG-195 role must be control or candidate")
    fields = {str(item) for item in field_names}
    if not fields or any(field.casefold() in _FORBIDDEN_FIELDS for field in fields):
        raise ValueError("PG-195 fields are not safe for replay")
    result: dict[str, str] = {}
    for field in sorted(fields):
        lowered = field.casefold()
        if lowered == "submit":
            result[field] = "submit"
        elif lowered == "id":
            result[field] = "1"
        elif lowered not in _VALUE_FIELDS:
            result[field] = marker
        elif probe_role == "candidate":
            result[field] = inert_dom_probe(marker, encoding=encoding)
        else:
            result[field] = marker
    return result


def project_surface_response(
    response: httpx.Response,
    *,
    marker: str,
    layout_variant: str,
    baseline_status: int | None = None,
    run_browser: bool = True,
) -> dict[str, Any]:
    """Return only bounded transport and no-JS DOM projections."""

    marker = validate_marker(marker)
    projection, signal, _ = _summary(response, marker=marker, baseline_status=baseline_status)
    # Parse the same response under a few inert layout shells.  The shells do
    # not add scripts, URLs, or handlers; they exercise layout generalization
    # without turning structure into an XSS claim.
    body = response.text
    shells = {
        "inline_html": "<main>{}</main>",
        "table_cell": "<table><tbody><tr><td>{}</td></tr></tbody></table>",
        "attribute_shell": '<section data-layout="pg195">{}</section>',
    }
    shell = shells.get(str(layout_variant), shells["inline_html"])
    browser = run_browser_dom_oracle(shell.format(body), marker=marker) if run_browser else {
        "marker_hits": 0,
        "element_count": 0,
        "script_tag_count": 0,
        "browser_dom_observed": False,
        "dom_change": False,
        "script_execution": False,
        "network_request_count": 0,
        "evidence_hash": sha256_json({"layout_variant": layout_variant, "marker_hits": 0}),
    }
    typed_surface_effect = bool(browser["dom_change"] and browser["marker_hits"] > 0 and not browser["script_execution"])
    oracle = {
        "oracle_id": "pg195-browser-dom-layout-nojs-v1",
        "modality": "typed_dom_surface_effect" if typed_surface_effect else "negative_or_untyped_surface",
        "typed_surface_effect": typed_surface_effect,
        "positive": False,
        "positive_authority": False,
        "confirmed_effect": "dom_structure" if typed_surface_effect else "none",
        "layout_variant": str(layout_variant),
        "signals": {
            "marker_reflected": bool(signal.get("marker_reflected")),
            "marker_location": str((projection.get("marker") or {}).get("location", "none")),
            "marker_hits": int(browser["marker_hits"]),
            "element_count": int(browser["element_count"]),
            "script_tag_count": int(browser["script_tag_count"]),
            "browser_dom_observed": bool(browser["browser_dom_observed"]),
            "dom_change": bool(browser["dom_change"]),
            "script_execution": False,
            "network_request_count": int(browser["network_request_count"]),
            "evidence_hash": str(browser["evidence_hash"]),
        },
        "safety": {
            "external_network": False,
            "script_execution": False,
            "database_write": False,
            "navigation": False,
            "raw_body_stored": False,
        },
    }
    oracle["projection_sha256"] = sha256_json(oracle)
    return {
        "schema_version": SCHEMA_VERSION,
        "response_projection": projection,
        "oracle_projection": oracle,
        "typed_surface_effect": typed_surface_effect,
        "raw_response_retained": False,
        "body_text": body,
        "signal": signal,
        "response_sha256": _digest(projection),
    }


def send_surface_request(
    client: httpx.Client,
    *,
    path: str,
    method: str,
    values: Mapping[str, str],
    marker: str,
    layout_variant: str,
    baseline_status: int | None = None,
    run_browser: bool = True,
) -> dict[str, Any]:
    method = str(method).upper()
    if method == "GET":
        response = client.get(path, params=dict(values), follow_redirects=False)
    elif method == "POST":
        response = client.post(path, data=dict(values), follow_redirects=False)
    else:
        raise ValueError("PG-195 request method is not allow-listed")
    result = project_surface_response(response, marker=marker, layout_variant=layout_variant, baseline_status=baseline_status, run_browser=run_browser)
    result["status"] = int(response.status_code)
    return result


__all__ = [
    "SCHEMA_VERSION",
    "build_surface_action_manifest",
    "build_surface_values",
    "project_surface_response",
    "send_surface_request",
]
