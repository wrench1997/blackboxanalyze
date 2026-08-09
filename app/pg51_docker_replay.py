"""Projection-only GET/POST adapter for the pinned local Pikachu Docker app.

This adapter intentionally confirms only bounded surface signals.  It never
claims JavaScript execution, SQL execution, redirect to an external origin, or
any other vulnerability effect.  Request values and response bodies are
discarded after projection; only hashes and typed booleans leave the adapter.
"""

from __future__ import annotations

import hashlib
import json
import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlsplit

import httpx

from .cross_lab_safe_catalog import ReadOnlySafeCatalogCollector, sha256_json
from .trace_aligned_dataset import validate_trace_step


PG51_SCHEMA = "sift-pg51-pikachu-docker-replay-v1"
PIKACHU_IMAGE_DIGEST = "sha256:b32c7362bb102091bd4ef09c5c571db146bd57469d5598f0c8681ffeeb7907fe"
BASE_URLS = frozenset({"http://127.0.0.1:8767", "http://127.0.0.1:8768"})
SAFE_METHODS = frozenset({"GET", "POST"})
SAFE_PATHS: dict[str, dict[str, Any]] = {
    "/vul/xss/xss_reflected_get.php": {"family": "xss", "surface": "xss_reflected_get", "query_name": "message", "submit": "submit"},
    "/vul/xss/xss_dom.php": {"family": "xss", "surface": "xss_dom_source", "query_name": "text", "submit": "submit"},
    "/vul/sqli/sqli_str.php": {"family": "injection", "surface": "sqli_string", "query_name": "name", "submit": "submit"},
    "/vul/sqli/sqli_search.php": {"family": "injection", "surface": "sqli_search", "query_name": "name", "submit": "submit"},
    "/vul/sqli/sqli_blind_b.php": {"family": "injection", "surface": "sqli_boolean", "query_name": "name", "submit": "submit"},
    "/vul/sqli/sqli_blind_t.php": {"family": "injection", "surface": "sqli_timing", "query_name": "name", "submit": "submit"},
    "/vul/urlredirect/urlredirect.php": {"family": "url_redirect", "surface": "url_redirect", "query_name": "name", "submit": "submit"},
}
SQL_ERROR_PATTERNS = ("sql syntax", "mysql", "sqlite error", "odbc")
FORBIDDEN_INPUT_MARKERS = ("<script", "union select", "javascript:", "../", "sleep(", "benchmark(")


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


def validate_target(base_url: str) -> str:
    value = str(base_url).rstrip("/")
    parsed = urlsplit(value)
    if value not in BASE_URLS or parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
        raise ValueError("PG-51 target must be an allow-listed loopback Pikachu URL")
    return value


def validate_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("PG-51 spec must be an object")
    target = validate_target(str(spec.get("target", "")))
    path = str(spec.get("path", ""))
    if path not in SAFE_PATHS:
        raise ValueError("PG-51 path is not allow-listed")
    method = str(spec.get("method", "GET")).upper()
    if method not in SAFE_METHODS:
        raise ValueError("PG-51 permits only GET and POST")
    marker = str(spec.get("marker", ""))
    if not re.fullmatch(r"[A-Za-z0-9._-]{6,64}", marker):
        raise ValueError("PG-51 marker must be an inert bounded identifier")
    if any(token in marker.casefold() for token in FORBIDDEN_INPUT_MARKERS):
        raise ValueError("PG-51 marker is not an inert identifier")
    return {"target": target, "path": path, "method": method, "marker": marker, **SAFE_PATHS[path]}


def _summary(response: httpx.Response) -> dict[str, Any]:
    body = bytes(response.content)
    content_type = str(response.headers.get("content-type", "")).split(";", 1)[0].casefold()
    return {"status_code": int(response.status_code), "status_class": f"{int(response.status_code) // 100}xx" if 100 <= int(response.status_code) <= 599 else "other", "content_type_class": content_type if content_type in {"html", "json", "text", "xml"} else "other", "body_length_bucket": "0" if not body else "1-255" if len(body) <= 255 else "256-4095" if len(body) <= 4095 else "4096-65535", "body_sha256": hashlib.sha256(body).hexdigest(), "semantic_body_sha256": hashlib.sha256(body).hexdigest(), "shape": {"kind": content_type, "field_count": 0, "scalar_count": 0}, "header_names": sorted({str(key).casefold() for key in response.headers if str(key).casefold() in {"content-type", "location", "allow"}}), "marker": {"reflected": False, "location": "none", "count": 0}, "frame_policy": "unknown", "transport_error": False, "status_changed": False, "state_changed": False, "location_origin_changed": False}


def projection(response: httpx.Response, baseline: httpx.Response, marker: str) -> dict[str, Any]:
    body = response.content.decode("utf-8", errors="replace")
    baseline_body = baseline.content
    lower = body.casefold()
    parser = _ShapeParser()
    try:
        parser.feed(body)
        parser.close()
    except (TypeError, ValueError):
        pass
    marker_reflected = marker in body
    marker_in_script = bool(re.search(r"<script\b[^>]*>[^<]{0,2048}" + re.escape(marker), body, re.IGNORECASE | re.DOTALL))
    marker_in_attribute = bool(re.search(r"(?:=|['\"])\s*[^<>]{0,256}" + re.escape(marker), body))
    location = str(response.headers.get("location", ""))
    parsed_location = urlsplit(location) if location else None
    external_redirect = bool(parsed_location and (parsed_location.scheme or parsed_location.netloc) and not (parsed_location.scheme == "http" and parsed_location.hostname == "127.0.0.1" and parsed_location.port in {None, 8767, 8768}))
    result = {"marker_reflected": marker_reflected, "marker_in_script_source": marker_in_script, "marker_in_attribute": marker_in_attribute, "marker_count": min(body.count(marker), 8), "sql_error_shape": any(pattern in lower for pattern in SQL_ERROR_PATTERNS), "external_redirect": external_redirect, "redirect_present": bool(location), "status_changed": response.status_code != baseline.status_code, "body_length_delta_abs": abs(len(response.content) - len(baseline_body)), "content_type_class": "html" if "html" in str(response.headers.get("content-type", "")).casefold() else "other", "html_tag_count": min(parser.tags, 512), "form_count": min(parser.forms, 64), "input_count": min(parser.inputs, 128), "script_count": min(parser.scripts, 64)}
    return result


def _response_projection(response: httpx.Response, baseline: httpx.Response, marker: str) -> tuple[dict[str, Any], dict[str, Any]]:
    signal = projection(response, baseline, marker)
    result = _summary(response)
    result["marker"] = {"reflected": bool(signal["marker_reflected"]), "location": "html_text" if signal["marker_reflected"] else "none", "count": int(signal["marker_count"])}
    result["status_changed"] = bool(signal["status_changed"])
    result["location_origin_changed"] = bool(signal["external_redirect"])
    result["shape"] = {"kind": result["content_type_class"], "field_count": int(signal["html_tag_count"]), "scalar_count": int(signal["form_count"])}
    result["projection_sha256"] = sha256_json(result)
    return result, signal


def typed_oracle(spec: dict[str, Any], projection_value: dict[str, Any]) -> dict[str, Any]:
    # This track has no browser-execution, SQL-AST, or redirect evaluator.
    # Source reflection is therefore only a candidate signal; it must never
    # become a positive authority merely because the HTML looks interesting.
    source_reflection = spec["family"] == "xss" and bool(projection_value.get("marker_in_script_source") or projection_value.get("marker_in_attribute"))
    candidate_signal = bool(projection_value.get("marker_reflected") or projection_value.get("sql_error_shape") or projection_value.get("external_redirect"))
    return {"oracle_id": f"pg51-{spec['surface']}-surface-signal-v1", "family": spec["family"], "modality": "surface_reflection_signal" if source_reflection else "negative_control", "candidate_signal": candidate_signal, "positive": False, "positive_authority": False, "confirmed_effect": "none", "authority_blocker": "execution_or_sql_ast_or_redirect_oracle_unavailable", "signals": {"surface": spec["surface"], "method": spec["method"], "marker_reflected": bool(projection_value.get("marker_reflected")), "marker_in_script_source": bool(projection_value.get("marker_in_script_source")), "marker_in_attribute": bool(projection_value.get("marker_in_attribute")), "sql_error_shape": bool(projection_value.get("sql_error_shape")), "external_redirect": bool(projection_value.get("external_redirect"))}, "safety": {"external_network": False, "script_execution": False, "database_write": False, "persistent_state_mutated": False, "credentials_accessed": False, "raw_body_stored": False}}


def payload_manifest(spec: dict[str, Any]) -> dict[str, Any]:
    basis = {key: spec[key] for key in ("path", "method", "family", "surface")}
    placement = "query" if spec["method"] == "GET" else "form"
    fields = [spec["query_name"], "submit"]
    return {"manifest_id": f"pg51-{spec['surface']}-{spec['method'].casefold()}", "payload_sha256": sha256_json({"basis": basis, "marker_sha256": hashlib.sha256(spec["marker"].encode()).hexdigest()}), "probe_ref": f"pg51-inert-{spec['family']}-canary", "probe_kind": "http_canary", "route_template_id": f"pikachu-{spec['surface']}", "method": spec["method"], "placement": placement, "encoding_chain": ["identity"], "encoding_depth": 0, "marker_sha256": hashlib.sha256(spec["marker"].encode()).hexdigest(), "max_bytes": 128, "form_field_names": fields if spec["method"] == "POST" else [], "form_content_type": "application/x-www-form-urlencoded" if spec["method"] == "POST" else "", "safety": {"does_not_execute": True, "no_external_network": True, "no_script_execution": True, "no_database_write": True, "no_credential_access": True}}


def rule_ir(spec: dict[str, Any]) -> dict[str, Any]:
    return {"rule_key": f"{spec['family']}.pg51.surface-signal", "grammar_version": "rule-ir-v1", "family_candidate": spec["family"], "operator_set": ["and", "present"], "required_slots": ["surface", "transport", "oracle"], "bound_slots": ["surface", "transport", "oracle"], "executable": False}


def collect_pair(*, source: dict[str, Any], registry: dict[str, Any], spec: dict[str, Any], client: httpx.Client, target_instance_id: str, reset_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    normalized = validate_spec(spec)
    path = normalized["path"]
    params = {normalized["query_name"]: normalized["marker"], "submit": normalized["submit"]}
    control_params = {"submit": normalized["submit"]}
    if normalized["method"] == "GET":
        baseline = client.get(path)
        control_response = client.get(path, params=control_params)
        candidate_response = client.get(path, params=params)
    else:
        headers = {"content-type": "application/x-www-form-urlencoded"}
        baseline = client.get(path)
        control_response = client.post(path, data=control_params, headers=headers)
        candidate_response = client.post(path, data=params, headers=headers)
    baseline_projection = _summary(baseline)
    baseline_projection["projection_sha256"] = sha256_json(baseline_projection)
    control_projection, control_signal = _response_projection(control_response, baseline, normalized["marker"])
    candidate_projection, candidate_signal = _response_projection(candidate_response, baseline, normalized["marker"])
    source_oracle = typed_oracle(normalized, candidate_signal)
    control_oracle = {**typed_oracle({**normalized, "marker": "control"}, control_signal), "positive": False, "positive_authority": False, "confirmed_effect": "none", "modality": "negative_control"}
    reset_common = {"kind": "fresh_pikachu_docker_round", "reset_id": reset_id, "target_instance_id": target_instance_id, "state_epoch": f"{target_instance_id}-epoch", "reset_adapter_sha256": source["reset_adapter_sha256"], "baseline_projection_sha256": baseline_projection["projection_sha256"], "fresh_target": True, "completed": True, "evaluator_state_hidden": True, "state_change_allowed": False, "external_network": False, "transport": "httpx_loopback"}
    def make_record(response_projection: dict[str, Any], oracle: dict[str, Any], role: str, suffix: str, control: dict[str, Any] | None) -> dict[str, Any]:
        reset = {**reset_common, "reset_id": f"{reset_id}-{suffix}"}
        manifest = payload_manifest(normalized)
        manifest["manifest_id"] = f"{manifest['manifest_id']}-{suffix}"
        return ReadOnlySafeCatalogCollector(source, registry=registry).collect(sample_id=f"pg51-{normalized['surface']}-{normalized['method'].casefold()}-{suffix}", sample_role=role, sampling_seed=51, reset=reset, payload_manifest=manifest, response_projection=response_projection, oracle_projection={"oracle_contract_sha256": source["oracle_contract_sha256"], **oracle}, rule_ir=rule_ir(normalized), negative_control=control)
    control = make_record(control_projection, control_oracle, "negative_control", "control", None)
    positive = bool(source_oracle["positive"])
    candidate = make_record(candidate_projection, source_oracle, "candidate" if positive else "negative_control", "candidate", {"control_sample_id": control["sample_id"], "control_evidence_hash": control["evidence"]["evidence_hash"], "intervention": "pg51-inert-marker-vs-control", "verdict": "confirmed_negative", "same_source": True, "same_surface": True} if positive else None)
    return control, candidate, {"baseline": baseline_projection, "candidate": candidate_projection, "positive": positive}


__all__ = ["BASE_URLS", "PIKACHU_IMAGE_DIGEST", "PG51_SCHEMA", "SAFE_PATHS", "collect_pair", "projection", "rule_ir", "typed_oracle", "validate_spec", "validate_target"]
