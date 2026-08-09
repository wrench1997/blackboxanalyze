"""Projection-only iterative GET/POST probing for the local Pikachu track.

PG-179B records the request/response process, not exploit strings.  It uses
only bounded alphanumeric canaries and keeps every redirect hop as a bounded
shape.  A reflection, SQL-looking error, or redirect is a candidate signal;
without an execution/AST/redirect oracle it is never a positive.
"""

from __future__ import annotations

import hashlib
import re
from html.parser import HTMLParser
from typing import Any, Mapping
from urllib.parse import urljoin, urlsplit

import httpx

from .cross_lab_safe_catalog import sha256_json, validate_payload_manifest
from .failure_guided_scheduler import failure_signature
from .pg51_docker_replay import FORBIDDEN_INPUT_MARKERS, PIKACHU_IMAGE_DIGEST, SAFE_PATHS


PG179B_SCHEMA = "sift-pg179b-pikachu-iterative-probe-v1"
MAX_REDIRECT_HOPS = 6
SAFE_MARKER_RE = re.compile(r"^[A-Za-z0-9._-]{6,64}$")
SQL_ERROR_PATTERNS = ("sql syntax", "mysql", "sqlite error", "odbc")
SAFE_REDIRECT_PORTS = {8767, 8768, 8779, 8781, 3101, 3113, 8791}


class _ShapeParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags = 0
        self.forms = 0
        self.inputs = 0
        self.scripts = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags += 1
        lower = tag.casefold()
        self.forms += int(lower == "form")
        self.inputs += int(lower in {"input", "textarea", "select", "button"})
        self.scripts += int(lower == "script")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)


def validate_marker(marker: str) -> str:
    value = str(marker)
    if not SAFE_MARKER_RE.fullmatch(value) or any(token in value.casefold() for token in FORBIDDEN_INPUT_MARKERS):
        raise ValueError("PG-179B marker must be a bounded alphanumeric canary")
    return value


def _length_bucket(length: int) -> str:
    if length <= 0:
        return "0"
    if length <= 255:
        return "1-255"
    if length <= 4095:
        return "256-4095"
    if length <= 65535:
        return "4096-65535"
    return "65536+"


def _status_class(status: int) -> str:
    return f"{status // 100}xx" if 100 <= status <= 599 else "other"


def _location_shape(location: str, current_url: str) -> tuple[dict[str, Any] | None, bool]:
    if not location:
        return None, False
    absolute = urljoin(current_url, location)
    parsed = urlsplit(absolute)
    same_origin = parsed.scheme == "http" and parsed.hostname == "127.0.0.1" and (parsed.port or 80) in SAFE_REDIRECT_PORTS
    path = parsed.path or "/"
    segments = [segment for segment in path.split("/") if segment]
    return {
        # Keep redirect evidence as a shape only.  A canary can legitimately
        # appear in a Location path; persisting that value would turn a
        # projection into a raw probe echo.
        "path_shape": {
            "segment_count": min(len(segments), 32),
            "has_extension": bool(segments and "." in segments[-1]),
        },
        "query_keys": sorted({key for key, _ in httpx.QueryParams(parsed.query).multi_items()}),
    }, same_origin


def _summary(response: httpx.Response, *, marker: str | None = None, baseline_status: int | None = None) -> tuple[dict[str, Any], dict[str, Any], str]:
    body = bytes(response.content)
    body_text = body.decode("utf-8", errors="replace")
    content_type = str(response.headers.get("content-type", "")).split(";", 1)[0].casefold()
    content_class = content_type if content_type in {"html", "json", "text", "xml"} else "other"
    parser = _ShapeParser()
    try:
        parser.feed(body_text)
        parser.close()
    except (TypeError, ValueError):
        pass
    reflected = bool(marker and marker in body_text)
    in_script = bool(marker and re.search(r"<script\b[^>]*>[^<]{0,2048}" + re.escape(marker), body_text, re.IGNORECASE | re.DOTALL))
    in_attribute = bool(marker and re.search(r"(?:=|['\"])\s*[^<>]{0,256}" + re.escape(marker), body_text))
    location = str(response.headers.get("location", ""))
    location_shape, same_origin = _location_shape(location, str(response.url))
    projection = {
        "status_code": int(response.status_code),
        "status_class": _status_class(int(response.status_code)),
        "content_type_class": content_class,
        "body_length_bucket": _length_bucket(len(body)),
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "semantic_body_sha256": hashlib.sha256(body).hexdigest(),
        "shape": {"kind": content_class, "field_count": min(parser.tags, 512), "scalar_count": min(parser.forms, 64)},
        "header_names": sorted({str(key).casefold() for key in response.headers if str(key).casefold() in {"content-type", "location", "allow"}}),
        "marker": {"reflected": reflected, "location": "html_text" if reflected else "none", "count": min(body_text.count(marker or ""), 8) if marker else 0},
        "frame_policy": "unknown",
        "transport_error": False,
        "status_changed": baseline_status is not None and int(response.status_code) != int(baseline_status),
        "state_changed": False,
        "location_origin_changed": bool(location_shape and not same_origin),
    }
    projection["projection_sha256"] = sha256_json(projection)
    signal = {
        "marker_reflected": reflected,
        "marker_in_script_source": in_script,
        "marker_in_attribute": in_attribute,
        "marker_count": min(body_text.count(marker or ""), 8) if marker else 0,
        "sql_error_shape": any(pattern in body_text.casefold() for pattern in SQL_ERROR_PATTERNS),
        "redirect_present": bool(location_shape),
        "external_redirect": bool(location_shape and not same_origin),
        "status_changed": projection["status_changed"],
        "body_length_delta": 0,
        "html_tag_count": min(parser.tags, 512),
        "form_count": min(parser.forms, 64),
        "input_count": min(parser.inputs, 128),
        "script_count": min(parser.scripts, 64),
    }
    projection["status_chain"] = [int(response.status_code)]
    projection["redirect_chain"] = [location_shape] if location_shape else []
    projection["redirect_external_blocked"] = bool(location_shape and not same_origin)
    return projection, signal, sha256_json(signal)


def request_chain(
    client: httpx.Client,
    *,
    method: str,
    path: str,
    query: Mapping[str, str] | None = None,
    form: Mapping[str, str] | None = None,
    marker: str | None = None,
    baseline_status: int | None = None,
) -> dict[str, Any]:
    """Send one bounded request and follow only same-origin redirect hops."""

    method = str(method).upper()
    if method not in {"GET", "POST"}:
        raise ValueError("PG-179B permits only GET and POST")
    marker = validate_marker(marker) if marker else None
    current = path
    current_method = method
    first = True
    hops: list[dict[str, Any]] = []
    final_projection: dict[str, Any] | None = None
    final_signal: dict[str, Any] | None = None
    for _ in range(MAX_REDIRECT_HOPS):
        try:
            if current_method == "GET":
                response = client.get(current, params=query if first else None)
            else:
                response = client.post(current, data=form if first else None, headers={"content-type": "application/x-www-form-urlencoded"})
        except httpx.HTTPError as exc:
            projection = {
                "status_code": 0,
                "status_class": "transport_error",
                "content_type_class": "unknown",
                "body_length_bucket": "unknown",
                "body_sha256": hashlib.sha256(b"").hexdigest(),
                "semantic_body_sha256": hashlib.sha256(b"").hexdigest(),
                "shape": {"kind": "transport_error", "field_count": 0, "scalar_count": 0},
                "header_names": [],
                "marker": {"reflected": False, "location": "none", "count": 0},
                "frame_policy": "unknown",
                "transport_error": True,
                "status_changed": False,
                "state_changed": False,
                "location_origin_changed": False,
            }
            projection["projection_sha256"] = sha256_json(projection)
            hops.append({"method": current_method, "status": 0, "projection_sha256": projection["projection_sha256"], "error_class": type(exc).__name__})
            final_projection = projection
            final_signal = {"transport_error": True, "candidate_signal": False, "redirect_present": False, "external_redirect": False, "status_changed": False}
            break
        projection, signal, signal_hash = _summary(response, marker=marker, baseline_status=baseline_status)
        location_shape = projection["redirect_chain"][0] if projection["redirect_chain"] else None
        location_header = str(response.headers.get("location", ""))
        _, same_origin = _location_shape(location_header, str(response.url))
        hop = {
            "method": current_method,
            "status": int(response.status_code),
            "status_class": projection["status_class"],
            "content_type_class": projection["content_type_class"],
            "projection_sha256": projection["projection_sha256"],
            "location": location_shape,
            "location_same_origin": bool(same_origin) if location_header else None,
            "signal_sha256": signal_hash,
        }
        hops.append(hop)
        final_projection = projection
        final_signal = {**signal, "candidate_signal": bool(signal.get("marker_reflected") or signal.get("sql_error_shape") or signal.get("redirect_present") or signal.get("status_changed")), "redirect_hop_count": 0, "status_chain_sha256": ""}
        if int(response.status_code) not in {301, 302, 303, 307, 308} or not location_header or not same_origin:
            break
        current = urlsplit(urljoin(str(response.url), location_header)).path or "/"
        current_method = "GET" if int(response.status_code) in {301, 302, 303} else current_method
        first = False
    assert final_projection is not None and final_signal is not None
    final_projection["status_chain"] = [item["status"] for item in hops]
    final_projection["redirect_chain"] = [item["location"] for item in hops if item.get("location")]
    final_projection["redirect_hop_count"] = max(0, len(hops) - 1)
    final_projection["status_chain_sha256"] = sha256_json(hops)
    final_projection["projection_sha256"] = sha256_json({key: value for key, value in final_projection.items() if key not in {"status_chain_sha256", "projection_sha256"}})
    final_signal["redirect_hop_count"] = max(0, len(hops) - 1)
    final_signal["status_chain_sha256"] = final_projection["status_chain_sha256"]
    final_signal["candidate_signal"] = bool(final_signal.get("candidate_signal") or final_signal.get("external_redirect"))
    return {"projection": final_projection, "signal": final_signal, "hops": hops, "status_chain_sha256": final_projection["status_chain_sha256"]}


def surface_oracle(*, family: str, method: str, signal: Mapping[str, Any], oracle_contract_sha256: str, negative_control_pair_id: str | None = None) -> dict[str, Any]:
    candidate = bool(signal.get("candidate_signal"))
    return {
        "oracle_id": f"pg179b-{family}-surface-signal-v1",
        "oracle_contract_sha256": oracle_contract_sha256,
        "family": family,
        "modality": "surface_signal_without_typed_effect" if candidate else "negative_control",
        "candidate_signal": candidate,
        "positive": False,
        "positive_authority": False,
        "confirmed_effect": "none",
        "signals": {
            "method": method,
            "marker_reflected": bool(signal.get("marker_reflected")),
            "sql_error_shape": bool(signal.get("sql_error_shape")),
            "redirect_present": bool(signal.get("redirect_present")),
            "external_redirect": bool(signal.get("external_redirect")),
            "status_changed": bool(signal.get("status_changed")),
            "status_chain_sha256": str(signal.get("status_chain_sha256", "")),
        },
        **({"negative_control_pair_id": negative_control_pair_id} if negative_control_pair_id else {}),
        "safety": {
            "external_network": False,
            "script_execution": False,
            "database_write": False,
            "persistent_state_mutated": False,
            "credentials_accessed": False,
            "raw_body_stored": False,
        },
    }


def action_manifest(*, path: str, surface: str, family: str, method: str, field_names: list[str], probe_role: str, marker: str) -> dict[str, Any]:
    """Build and validate the model-facing bounded request manifest.

    The role decoder is deliberately not allowed to send a request directly.
    This function binds the role to the browser-observed route/field set and
    then runs the shared Catalog manifest validator before returning.  The
    marker itself is never returned as part of the manifest; only its digest
    crosses the model/evidence boundary.
    """
    method = str(method).upper()
    fields = sorted({str(item) for item in field_names})
    if method not in {"GET", "POST"}:
        raise ValueError("PG-179B action manifest permits only GET or POST")
    if not path.startswith("/") or "://" in path or path.startswith("//"):
        raise ValueError("PG-179B action manifest path must be origin-relative")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", surface):
        raise ValueError("PG-179B action manifest surface is invalid")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,32}", family):
        raise ValueError("PG-179B action manifest family is invalid")
    if probe_role not in {"negative_control", "control", "candidate"}:
        raise ValueError("PG-179B action manifest role is invalid")
    marker = validate_marker(marker)
    marker_hash = hashlib.sha256(marker.encode("utf-8")).hexdigest()
    body = {"path": path, "surface": surface, "family": family, "method": method, "fields": fields, "role": probe_role, "marker_sha256": marker_hash}
    manifest = {
        "manifest_id": f"pg179b-{surface}-{method.casefold()}-{probe_role}",
        "payload_sha256": sha256_json(body),
        "probe_ref": f"pg179b-safe-{family}-canary-{probe_role}",
        "probe_kind": "http_canary",
        "route_template_id": f"pikachu-{surface}",
        "method": method,
        "placement": "query" if method == "GET" else "form",
        "encoding_chain": ["identity"],
        "encoding_depth": 0,
        "marker_sha256": marker_hash,
        "max_bytes": 128,
        "safety": {"does_not_execute": True, "no_external_network": True, "no_script_execution": True, "no_database_write": True, "no_credential_access": True},
    }
    if method == "POST":
        manifest["form_field_names"] = fields
        manifest["form_content_type"] = "application/x-www-form-urlencoded"
    checked = validate_payload_manifest(manifest)
    if checked["payload_sha256"] != manifest["payload_sha256"]:
        raise ValueError("PG-179B payload manifest hash changed during validation")
    if checked["marker_sha256"] != marker_hash:
        raise ValueError("PG-179B payload marker binding changed during validation")
    if method == "POST" and checked.get("form_field_names") != fields:
        raise ValueError("PG-179B POST field binding changed during validation")
    return checked


def failure_for_step(*, method: str, role: str, signal: Mapping[str, Any], prior_records: list[dict[str, Any]], step_count: int, max_steps: int = 5) -> dict[str, Any]:
    record = {
        "method": method,
        "role": role,
        "candidate_signal": bool(signal.get("candidate_signal")),
        "positive": False,
        "positive_authority": False,
        "typed_available": False,
        "probe_round": step_count,
        "max_probe_rounds": max_steps,
    }
    signature = failure_signature(record, prior_records=prior_records, max_steps=max_steps, step_count=step_count)
    if method == "GET" and role == "candidate":
        signature["next_action"] = "probe_candidate_other_method" if bool(signal.get("candidate_signal")) else "repeat_matched_negative_pair"
    elif method == "POST" and role == "control":
        signature["next_action"] = "repeat_matched_negative_pair"
    elif method == "POST":
        signature["next_action"] = "abstain_unknown_oracle"
    return signature


__all__ = [
    "MAX_REDIRECT_HOPS",
    "PG179B_SCHEMA",
    "PIKACHU_IMAGE_DIGEST",
    "SAFE_PATHS",
    "action_manifest",
    "failure_for_step",
    "request_chain",
    "surface_oracle",
    "validate_marker",
]
