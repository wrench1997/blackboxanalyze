"""PG-185 read-only DOM-surface adapter for the local Pikachu target.

This adapter exists for the last-mile request loop, not for exploit delivery.
It creates an inert ``span`` marker in memory, sends it only to observed GET
parameters, and runs the detached DOM oracle over the in-memory response.  No
script is executed and no raw request/response value is returned or persisted.
"""

from __future__ import annotations

import hashlib
import json
from html import escape
from typing import Any, Mapping
from urllib.parse import quote

import httpx

from .cross_lab_safe_catalog import sha256_json, validate_payload_manifest
from .dom_oracle import run_dom_oracle
from .pg179b_iterative_probe import _summary, validate_marker


SCHEMA_VERSION = "sift-pg185-pikachu-dom-adapter-v1"
ALLOWED_ROLES = frozenset({"negative_control", "control", "candidate"})


def inert_dom_probe(marker: str, *, encoding: str = "identity") -> str:
    """Return a non-executing DOM marker; the value is runtime-only."""

    marker = validate_marker(marker)
    # Keep this construction explicit so it cannot grow an event handler,
    # URL-bearing attribute, or script source by accident.
    raw = f'<span data-sift-marker="{escape(marker, quote=True)}">{escape(marker)}</span>'
    encoding = str(encoding)
    if encoding == "identity":
        return raw
    if encoding == "html_entity":
        return escape(raw, quote=True)
    if encoding == "html_entity_depth2":
        return escape(escape(raw, quote=True), quote=True)
    if encoding == "url_percent":
        return quote(raw, safe="")
    if encoding == "json_string":
        return json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
    raise ValueError(f"unsupported PG-185 inert encoding: {encoding}")


def build_dom_action_manifest(
    *,
    path: str,
    surface: str,
    field_names: list[str],
    probe_role: str,
    marker: str,
    encoding_chain: list[str] | tuple[str, ...] = ("identity",),
) -> dict[str, Any]:
    """Bind a model role to an observed GET surface and validate it."""

    if not path.startswith("/") or path.startswith("//") or "://" in path:
        raise ValueError("PG-185 path must be origin-relative")
    if probe_role not in ALLOWED_ROLES:
        raise ValueError("PG-185 probe role is not allow-listed")
    marker = validate_marker(marker)
    fields = sorted({str(item) for item in field_names})
    if not fields:
        raise ValueError("PG-185 requires at least one observed GET parameter")
    encodings = [str(item) for item in encoding_chain]
    if not encodings or len(encodings) > 3 or any(item not in {"identity", "url_percent", "html_entity", "json_string"} for item in encodings):
        raise ValueError("PG-185 encoding chain is invalid")
    marker_hash = hashlib.sha256(marker.encode("utf-8")).hexdigest()
    body = {
        "path": path,
        "surface": surface,
        "method": "GET",
        "fields": fields,
        "role": probe_role,
        "marker_sha256": marker_hash,
        "probe_kind": "inert_dom_markup",
        "encoding_chain": encodings,
    }
    manifest = {
        "manifest_id": f"pg185-{surface}-get-{probe_role}",
        "payload_sha256": sha256_json(body),
        "probe_ref": f"pg185-inert-dom-{surface}-{probe_role}",
        "probe_kind": "inert_dom_markup",
        "route_template_id": f"pikachu-{surface}",
        "method": "GET",
        "placement": "query",
        "encoding_chain": encodings,
        "encoding_depth": sum(item != "identity" for item in encodings),
        "marker_sha256": marker_hash,
        "max_bytes": 512,
        "safety": {
            "does_not_execute": True,
            "no_external_network": True,
            "no_script_execution": True,
            "no_database_write": True,
            "no_credential_access": True,
        },
    }
    checked = validate_payload_manifest(manifest)
    if checked["payload_sha256"] != manifest["payload_sha256"]:
        raise ValueError("PG-185 payload binding changed during validation")
    return checked


def project_dom_response(
    response: httpx.Response,
    *,
    marker: str | None,
    baseline_status: int | None = None,
) -> dict[str, Any]:
    """Project one response and run a detached DOM oracle in memory."""

    marker = validate_marker(marker) if marker else None
    projection, signal, signal_hash = _summary(response, marker=marker, baseline_status=baseline_status)
    if marker:
        dom = run_dom_oracle(response.text, marker=marker).to_dict()
    else:
        dom = {
            "oracle": "controlled_detached_dom_v1",
            "browser_sink_observed": False,
            "dom_change": False,
            "marker_hits": 0,
            "script_execution": False,
            "network_access": False,
            "navigation": False,
            "database_touched": False,
        }
    typed_surface_effect = bool(
        marker
        and dom.get("browser_sink_observed")
        and dom.get("dom_change")
        and int(dom.get("marker_hits", 0)) > 0
        and not dom.get("script_execution")
    )
    oracle = {
        "oracle_id": "pg185-detached-dom-surface-v1",
        "oracle_contract_sha256": hashlib.sha256(b"pg185-detached-dom-surface-v1").hexdigest(),
        "modality": "typed_dom_surface_effect" if typed_surface_effect else "negative_or_untyped_surface",
        "candidate_signal": bool(signal.get("candidate_signal")),
        "typed_surface_effect": typed_surface_effect,
        # A DOM structure effect is intentionally not an XSS vulnerability
        # claim: no script execution or navigation is attempted.
        "positive": False,
        "positive_authority": False,
        "confirmed_effect": "dom_structure" if typed_surface_effect else "none",
        "signals": {
            "marker_reflected": bool(signal.get("marker_reflected")),
            "marker_in_attribute": bool(signal.get("marker_in_attribute")),
            "marker_in_script_source": bool(signal.get("marker_in_script_source")),
            "status_changed": bool(signal.get("status_changed")),
            "signal_sha256": signal_hash,
            "dom_evidence_hash": str(dom.get("evidence_hash", "")),
        },
        "safety": {
            "external_network": False,
            "script_execution": False,
            "database_write": False,
            "persistent_state_mutated": False,
            "credentials_accessed": False,
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
    }


def build_query(*, field_names: list[str], role: str, marker: str, encoding: str = "identity") -> tuple[dict[str, str], str | None]:
    """Build a query only from observed field names; return runtime marker."""

    fields = {str(item) for item in field_names}
    if not fields or not fields.issubset({"message", "submit", "text"}):
        raise ValueError("PG-185 query fields are not from the allow-listed observed surfaces")
    marker = validate_marker(marker)
    if role == "candidate":
        value = inert_dom_probe(marker, encoding=encoding)
        marker_for_oracle = marker
    else:
        value = marker
        marker_for_oracle = marker
    if "text" in fields:
        return {"text": value}, marker_for_oracle
    query = {"message": value}
    if "submit" in fields:
        query["submit"] = "submit"
    return query, marker_for_oracle


__all__ = [
    "ALLOWED_ROLES",
    "SCHEMA_VERSION",
    "build_dom_action_manifest",
    "build_query",
    "inert_dom_probe",
    "project_dom_response",
]
