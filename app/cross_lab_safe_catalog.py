"""Read-only, projection-only Catalog scaffold for local cross-lab research.

The collector in this module never performs I/O against a target.  It accepts
only pre-bounded response/oracle projections produced by an authorized local
adapter and binds them to source, reset, payload-manifest, negative-control,
and Rule IR attestations with canonical SHA-256 evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any, Iterable


REGISTRY_SCHEMA = "sift-cross-lab-registry-v1"
CATALOG_SCHEMA = "sift-cross-lab-safe-catalog-v1"
SOURCE_SCHEMA = "sift-cross-lab-safe-source-v1"
SAMPLE_SCHEMA = "sift-cross-lab-safe-sample-v1"
EVIDENCE_SCHEMA = "sift-cross-lab-safe-evidence-v1"
RULE_IR_MANIFEST_SCHEMA = "sift-cross-lab-rule-ir-manifest-v1"

_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ALLOWED_SOURCE_TYPES = frozenset({"authorized_local_container", "in_repo_synthetic"})
_ALLOWED_LICENSES = frozenset({"internal-research", "local-container", "in-repo-synthetic"})
_ALLOWED_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "POST"})
_ALLOWED_FORM_CONTENT_TYPES = frozenset({"application/x-www-form-urlencoded", "application/json"})
_ALLOWED_PLACEMENTS = frozenset({"query", "header", "path_segment", "form", "none"})
_ALLOWED_ENCODINGS = frozenset({"identity", "url_percent", "html_entity", "json_string"})
_ALLOWED_PROBE_KINDS = frozenset({
    "http_canary",
    "header_canary",
    "inert_dom_markup",
    "encoded_dom_markup",
    "abstract_channel_class",
})
_ALLOWED_CONTENT_TYPES = frozenset({"html", "json", "text", "xml", "other", "unknown"})
_ALLOWED_STATUS_CLASSES = frozenset({"1xx", "2xx", "3xx", "4xx", "5xx", "transport_error", "other"})
_ALLOWED_LENGTH_BUCKETS = frozenset({"0", "1-255", "256-4095", "4096-65535", "65536+", "unknown"})
_ALLOWED_MARKER_LOCATIONS = frozenset({"none", "html_attribute", "html_text", "json_value", "response_header"})
_ALLOWED_FRAME_POLICIES = frozenset({"none", "allowall", "sameorigin", "deny", "ancestors_none", "other", "unknown"})
_ALLOWED_PROJECTION_SCHEMAS = frozenset({"canonical_effect_projection_v3_surface_signal"})
_ALLOWED_EFFECTS = frozenset({
    "none",
    "dom_structure",
    "interpreter_boundary",
    "authorization_boundary",
    "business_invariant",
    "history_binding",
    "redirect_origin",
    "frame_protection",
    "authentication_boundary",
    "validation_boundary",
    "command_canary",
})
_WEAK_MODALITIES = frozenset({"reflection", "syntax_error", "bounded_timing", "status_change", "transport_error"})
_FORBIDDEN_PROJECTION_KEYS = frozenset({
    "body",
    "raw_body",
    "body_preview",
    "response_body",
    "request_body",
    "probe",
    "raw_probe",
    "password",
    "secret",
    "token",
    "authorization",
    "cookie",
    "set-cookie",
    "credentials",
    "challenge_key",
    "evaluator_label",
})
_SAFETY_FLAGS = (
    "external_network",
    "script_execution",
    "database_write",
    "persistent_state_mutated",
    "credentials_accessed",
    "raw_body_stored",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_id(value: Any, label: str) -> str:
    text = str(value)
    if not _ID_RE.fullmatch(text):
        raise ValueError(f"{label} must be a bounded identifier")
    return text


def _require_hash(value: Any, label: str) -> str:
    text = str(value).casefold()
    if not _HASH_RE.fullmatch(text):
        raise ValueError(f"{label} must be a canonical SHA-256 hex digest")
    return text


def _bounded_scalar(value: Any, label: str) -> Any:
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, str):
        if len(value) > 160 or any(token in value for token in ("\r", "\n", "<", ">")) or "://" in value:
            raise ValueError(f"{label} contains an unbounded or raw string")
        return value
    raise ValueError(f"{label} contains an unsupported scalar")


def _bounded_mapping(value: Any, label: str, *, depth: int = 0) -> Any:
    if depth > 4:
        raise ValueError(f"{label} exceeds bounded projection depth")
    if isinstance(value, dict):
        if len(value) > 48:
            raise ValueError(f"{label} has too many fields")
        result: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            if key.casefold() in _FORBIDDEN_PROJECTION_KEYS or not _ID_RE.fullmatch(key):
                raise ValueError(f"{label} contains forbidden projection field: {key}")
            result[key] = _bounded_mapping(child, f"{label}.{key}", depth=depth + 1)
        return result
    if isinstance(value, list):
        if len(value) > 32:
            raise ValueError(f"{label} contains an oversized list")
        return [_bounded_mapping(item, label, depth=depth + 1) for item in value]
    return _bounded_scalar(value, label)


def registry_status(registry: dict[str, Any], target_id: str) -> dict[str, Any]:
    """Return PG-24 eligibility without turning absence into authorization."""

    if not isinstance(registry, dict) or registry.get("schema_version") != REGISTRY_SCHEMA:
        raise ValueError("unsupported PG-24 cross-lab registry")
    if not bool(registry.get("read_only")) or bool(registry.get("raw_probe_strings_stored")) or bool(registry.get("evaluator_labels_stored")):
        raise ValueError("PG-24 registry safety contract is not satisfied")
    target_id = _require_id(target_id, "target_id")
    entry = next((dict(item) for item in registry.get("targets", []) if str(item.get("target_id")) == target_id), None)
    if entry is None:
        return {
            "registered": False,
            "training_eligible": False,
            "training_role": "unregistered_evaluation_only",
            "entry": None,
        }
    safety = dict(entry.get("safety") or {})
    if bool(safety.get("external_network", False)) or bool(safety.get("raw_body_stored", False)):
        raise ValueError("registered target violates read-only bounded-data safety")
    return {
        "registered": True,
        "training_eligible": bool(entry.get("training_eligible", False)),
        "training_role": str(entry.get("training_role", "evaluation_only")),
        "entry": entry,
    }


def validate_source(source: dict[str, Any], *, registry: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(source, dict):
        raise ValueError("cross-lab source must be an object")
    target_id = _require_id(source.get("target_id"), "target_id")
    status = registry_status(registry, target_id)
    source_type = str(source.get("source_type", ""))
    if source_type not in _ALLOWED_SOURCE_TYPES:
        raise ValueError("cross-lab source type is not authorized")
    license_id = str(source.get("license", ""))
    if license_id not in _ALLOWED_LICENSES:
        raise ValueError("cross-lab source license is not approved")
    if str(source.get("authorization", "")) != "workspace_local_only":
        raise ValueError("cross-lab source authorization must be workspace_local_only")
    if not bool(source.get("read_only")) or source.get("external_network") is not False:
        raise ValueError("cross-lab source must explicitly attest read-only loopback use")
    scope = dict(source.get("loopback_scope") or {})
    if str(scope.get("scheme", "http")) not in {"http", "https"} or str(scope.get("host", "")) not in {"127.0.0.1", "localhost"}:
        raise ValueError("cross-lab source scope must be loopback only")
    try:
        port = int(scope.get("port", 0))
    except (TypeError, ValueError):
        port = 0
    if not 1 <= port <= 65535:
        raise ValueError("cross-lab source scope requires a bounded local port")
    digest_field = "container_image_digest" if source_type == "authorized_local_container" else "fixture_source_sha256"
    digest = str(source.get(digest_field, "")).casefold()
    if source_type == "authorized_local_container":
        if not _IMAGE_DIGEST_RE.fullmatch(digest):
            raise ValueError("container source requires a pinned image digest")
    else:
        digest = _require_hash(digest, "fixture_source_sha256")
    entry = status["entry"] or {}
    registered_image = str(entry.get("container_image", "")).casefold()
    if registered_image and "@sha256:" in registered_image:
        expected_image = "sha256:" + registered_image.rsplit("@sha256:", 1)[1]
        if digest != expected_image:
            raise ValueError("source image digest disagrees with PG-24 registry")
    normalized = {
        "schema_version": SOURCE_SCHEMA,
        "target_id": target_id,
        "app_family": _require_id(source.get("app_family"), "app_family"),
        "source_id": _require_id(source.get("source_id"), "source_id"),
        "source_type": source_type,
        "origin_ref": str(source.get("origin_ref", ""))[:160],
        "license": license_id,
        "authorization": "workspace_local_only",
        "loopback_scope": {"scheme": str(scope.get("scheme", "http")), "host": str(scope["host"]), "port": port},
        digest_field: digest,
        "collector_sha256": _require_hash(source.get("collector_sha256"), "collector_sha256"),
        "reset_adapter_sha256": _require_hash(source.get("reset_adapter_sha256"), "reset_adapter_sha256"),
        "oracle_contract_sha256": _require_hash(source.get("oracle_contract_sha256"), "oracle_contract_sha256"),
        "registry": {
            "schema_version": REGISTRY_SCHEMA,
            "registered": status["registered"],
            "training_eligible": status["training_eligible"],
            "training_role": status["training_role"],
        },
        "read_only": True,
        "external_network": False,
    }
    expected = sha256_json(normalized)
    declared = source.get("source_sha256")
    if declared is not None and _require_hash(declared, "source_sha256") != expected:
        raise ValueError("cross-lab source hash mismatch")
    normalized["source_sha256"] = expected
    return normalized


def validate_reset(reset: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(reset, dict):
        raise ValueError("fresh reset proof must be an object")
    if not bool(reset.get("fresh_target")) or not bool(reset.get("completed")):
        raise ValueError("fresh reset proof is incomplete")
    if not bool(reset.get("evaluator_state_hidden")):
        raise ValueError("fresh reset must hide evaluator state")
    if reset.get("external_network") is not False or reset.get("state_change_allowed") is not False:
        raise ValueError("fresh reset violates read-only local safety")
    adapter_hash = _require_hash(reset.get("reset_adapter_sha256"), "reset.reset_adapter_sha256")
    if adapter_hash != source["reset_adapter_sha256"]:
        raise ValueError("fresh reset adapter does not match source attestation")
    transport = str(reset.get("transport", ""))
    if transport and transport not in {"in_process_asgi", "httpx_loopback"}:
        raise ValueError("fresh reset transport is not an allow-listed local transport")
    normalized = {
        "reset_id": _require_id(reset.get("reset_id"), "reset_id"),
        "kind": _require_id(reset.get("kind"), "reset.kind"),
        "target_instance_id": _require_id(reset.get("target_instance_id"), "target_instance_id"),
        "state_epoch": _require_id(reset.get("state_epoch"), "reset.state_epoch"),
        "reset_adapter_sha256": adapter_hash,
        "baseline_projection_sha256": _require_hash(reset.get("baseline_projection_sha256"), "baseline_projection_sha256"),
        "fresh_target": True,
        "completed": True,
        "evaluator_state_hidden": True,
        "state_change_allowed": False,
        "external_network": False,
    }
    if transport:
        normalized["transport"] = transport
    normalized["reset_sha256"] = sha256_json(normalized)
    return normalized


def validate_payload_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("payload manifest must be an object")
    method = str(manifest.get("method", "")).upper()
    if method not in _ALLOWED_METHODS:
        raise ValueError("PG-25-B collector permits GET, HEAD, OPTIONS, or bounded safe POST")
    kind = str(manifest.get("probe_kind", ""))
    if kind not in _ALLOWED_PROBE_KINDS:
        raise ValueError("payload manifest probe kind is not allow-listed")
    placement = str(manifest.get("placement", "none"))
    if placement not in _ALLOWED_PLACEMENTS:
        raise ValueError("payload manifest placement is not allow-listed")
    encodings = [str(item) for item in manifest.get("encoding_chain", [])]
    if not encodings or len(encodings) > 3 or any(item not in _ALLOWED_ENCODINGS for item in encodings):
        raise ValueError("payload manifest encoding chain is invalid")
    depth = int(manifest.get("encoding_depth", -1))
    expected_depth = sum(item != "identity" for item in encodings)
    if depth != expected_depth:
        raise ValueError("payload manifest encoding depth disagrees with its chain")
    max_bytes = int(manifest.get("max_bytes", 0))
    if not 1 <= max_bytes <= 2048:
        raise ValueError("payload manifest size budget is invalid")
    form_field_names = [_require_id(item, "POST form field name") for item in manifest.get("form_field_names", [])]
    if len(form_field_names) != len(set(form_field_names)) or len(form_field_names) > 16:
        raise ValueError("POST form field names are duplicated or oversized")
    form_content_type = str(manifest.get("form_content_type", ""))
    if method == "POST":
        if not form_field_names:
            raise ValueError("POST payload manifest requires bounded form field names")
        if form_content_type not in _ALLOWED_FORM_CONTENT_TYPES:
            raise ValueError("POST payload manifest requires an allow-listed form content type")
    elif form_field_names or form_content_type:
        raise ValueError("form metadata is only valid for POST payload manifests")
    safety = dict(manifest.get("safety") or {})
    for key in ("does_not_execute", "no_external_network", "no_script_execution", "no_database_write", "no_credential_access"):
        if not bool(safety.get(key)):
            raise ValueError(f"payload manifest safety attestation missing: {key}")
    normalized = {
        "manifest_id": _require_id(manifest.get("manifest_id"), "payload manifest_id"),
        "payload_sha256": _require_hash(manifest.get("payload_sha256"), "payload_sha256"),
        "probe_ref": _require_id(manifest.get("probe_ref"), "probe_ref"),
        "probe_kind": kind,
        "route_template_id": _require_id(manifest.get("route_template_id"), "route_template_id"),
        "method": method,
        "placement": placement,
        "encoding_chain": encodings,
        "encoding_depth": depth,
        "marker_sha256": _require_hash(manifest.get("marker_sha256"), "marker_sha256"),
        "max_bytes": max_bytes,
        "safety": {key: True for key in ("does_not_execute", "no_external_network", "no_script_execution", "no_database_write", "no_credential_access")},
    }
    if method == "POST":
        normalized["form_field_names"] = form_field_names
        normalized["form_content_type"] = form_content_type
    normalized["manifest_sha256"] = sha256_json(normalized)
    return normalized


def validate_response_projection(projection: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(projection, dict):
        raise ValueError("response projection must be an object")
    allowed = {
        "status_code", "status_class", "content_type_class", "body_length_bucket",
        "body_sha256", "semantic_body_sha256", "shape", "header_names", "marker",
        "frame_policy",
        "transport_error", "status_changed", "state_changed", "location_origin_changed",
        "effect_surface", "effect_geometry", "projection_schema",
        "projection_sha256",
    }
    extra = sorted(set(projection) - allowed)
    if extra:
        raise ValueError("response projection contains non-bounded fields: " + ", ".join(extra))
    transport_error = bool(projection.get("transport_error", False))
    status_code = int(projection.get("status_code", 0))
    if transport_error:
        if status_code != 0:
            raise ValueError("transport error projection must use status_code 0")
    elif not 100 <= status_code <= 599:
        raise ValueError("response projection status code is invalid")
    status_class = str(projection.get("status_class", "other"))
    if status_class not in _ALLOWED_STATUS_CLASSES:
        raise ValueError("response projection status class is invalid")
    content_type = str(projection.get("content_type_class", "unknown"))
    if content_type not in _ALLOWED_CONTENT_TYPES:
        raise ValueError("response projection content type is invalid")
    length_bucket = str(projection.get("body_length_bucket", "unknown"))
    if length_bucket not in _ALLOWED_LENGTH_BUCKETS:
        raise ValueError("response projection length bucket is invalid")
    shape = _bounded_mapping(dict(projection.get("shape") or {}), "response_projection.shape")
    headers = sorted({_require_id(item, "response header name") for item in projection.get("header_names", [])})
    if len(headers) > 24:
        raise ValueError("response projection has too many header names")
    marker = dict(projection.get("marker") or {})
    marker_location = str(marker.get("location", "none"))
    if marker_location not in _ALLOWED_MARKER_LOCATIONS:
        raise ValueError("response marker location is invalid")
    frame_policy = str(projection.get("frame_policy", "unknown"))
    if frame_policy not in _ALLOWED_FRAME_POLICIES:
        raise ValueError("response frame policy is invalid")
    effect_surface = projection.get("effect_surface")
    if effect_surface is not None:
        if not isinstance(effect_surface, dict):
            raise ValueError("response effect_surface must be an object")
        expected_surface_keys = {
            "boolean_field_count", "true_boolean_count", "numeric_field_count",
            "nonzero_numeric_count", "array_field_count", "key_hash_buckets",
            "observation_schema", "observation_sha256",
        }
        if set(effect_surface) != expected_surface_keys:
            raise ValueError("response effect_surface schema is invalid")
        surface_counts = {
            "boolean_field_count": (0, 32),
            "true_boolean_count": (0, 32),
            "numeric_field_count": (0, 32),
            "nonzero_numeric_count": (0, 32),
            "array_field_count": (0, 8),
        }
        for key, (lower, upper) in surface_counts.items():
            value = effect_surface.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"response effect_surface.{key} is invalid")
        buckets = effect_surface.get("key_hash_buckets")
        if (
            not isinstance(buckets, list)
            or len(buckets) > 16
            or any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 64 for value in buckets)
            or buckets != sorted(set(buckets))
        ):
            raise ValueError("response effect_surface.key_hash_buckets is invalid")
        if str(effect_surface.get("observation_schema")) != "bounded_effect_shape_v1":
            raise ValueError("response effect_surface observation schema is invalid")
        surface_without_hash = {key: effect_surface[key] for key in expected_surface_keys if key != "observation_sha256"}
        if _require_hash(effect_surface.get("observation_sha256"), "response effect_surface.observation_sha256") != sha256_json(surface_without_hash):
            raise ValueError("response effect_surface hash mismatch")
        effect_surface = {
            **{key: int(effect_surface[key]) for key in surface_counts},
            "key_hash_buckets": list(buckets),
            "observation_schema": "bounded_effect_shape_v1",
            "observation_sha256": _require_hash(effect_surface.get("observation_sha256"), "response effect_surface.observation_sha256"),
        }
    effect_geometry = projection.get("effect_geometry")
    if effect_geometry is not None:
        if not isinstance(effect_geometry, dict):
            raise ValueError("response effect_geometry must be an object")
        expected_geometry_keys = {
            "object_count", "array_count", "array_item_count", "boolean_count",
            "true_boolean_count", "numeric_count", "nonzero_numeric_count",
            "string_count", "string_length_bucket_sum", "leaf_count", "max_depth",
            "geometry_schema", "geometry_sha256",
        }
        if set(effect_geometry) != expected_geometry_keys:
            raise ValueError("response effect_geometry schema is invalid")
        geometry_limits = {
            "object_count": 64, "array_count": 32, "array_item_count": 64,
            "boolean_count": 64, "true_boolean_count": 64, "numeric_count": 64,
            "nonzero_numeric_count": 64, "string_count": 64,
            "string_length_bucket_sum": 128, "leaf_count": 128, "max_depth": 16,
        }
        for key, upper in geometry_limits.items():
            value = effect_geometry.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= upper:
                raise ValueError(f"response effect_geometry.{key} is invalid")
        if str(effect_geometry.get("geometry_schema")) != "anonymous_value_type_geometry_v1":
            raise ValueError("response effect_geometry schema is invalid")
        geometry_without_hash = {key: effect_geometry[key] for key in expected_geometry_keys if key != "geometry_sha256"}
        if _require_hash(effect_geometry.get("geometry_sha256"), "response effect_geometry.geometry_sha256") != sha256_json(geometry_without_hash):
            raise ValueError("response effect_geometry hash mismatch")
        effect_geometry = {
            **{key: int(effect_geometry[key]) for key in geometry_limits},
            "geometry_schema": "anonymous_value_type_geometry_v1",
            "geometry_sha256": _require_hash(effect_geometry.get("geometry_sha256"), "response effect_geometry.geometry_sha256"),
        }
    projection_schema = projection.get("projection_schema")
    if projection_schema is not None:
        projection_schema = str(projection_schema)
        if projection_schema not in _ALLOWED_PROJECTION_SCHEMAS:
            raise ValueError("response projection schema is invalid")
        if effect_surface is None or effect_geometry is None:
            raise ValueError("surface projection schema requires both semantic channels")
    normalized = {
        "status_code": status_code,
        "status_class": status_class,
        "content_type_class": content_type,
        "body_length_bucket": length_bucket,
        "body_sha256": _require_hash(projection.get("body_sha256"), "response body_sha256"),
        "semantic_body_sha256": _require_hash(projection.get("semantic_body_sha256"), "semantic_body_sha256"),
        "shape": shape,
        "header_names": headers,
        "marker": {
            "reflected": bool(marker.get("reflected", False)),
            "location": marker_location,
            "count": max(0, min(8, int(marker.get("count", 0)))),
        },
        "frame_policy": frame_policy,
        "transport_error": transport_error,
        "status_changed": bool(projection.get("status_changed", False)),
        "state_changed": bool(projection.get("state_changed", False)),
        "location_origin_changed": bool(projection.get("location_origin_changed", False)),
    }
    if effect_surface is not None:
        normalized["effect_surface"] = effect_surface
    if effect_geometry is not None:
        normalized["effect_geometry"] = effect_geometry
    if projection_schema is not None:
        normalized["projection_schema"] = projection_schema
    normalized["projection_sha256"] = sha256_json(normalized)
    declared = projection.get("projection_sha256")
    if declared is not None and _require_hash(declared, "response projection_sha256") != normalized["projection_sha256"]:
        raise ValueError("response projection hash mismatch")
    return normalized


def validate_oracle_projection(projection: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(projection, dict):
        raise ValueError("oracle projection must be an object")
    contract_hash = _require_hash(projection.get("oracle_contract_sha256"), "oracle_contract_sha256")
    if contract_hash != source["oracle_contract_sha256"]:
        raise ValueError("oracle contract does not match source attestation")
    modality = _require_id(projection.get("modality"), "oracle modality")
    effect = str(projection.get("confirmed_effect", "none"))
    if effect not in _ALLOWED_EFFECTS:
        raise ValueError("oracle confirmed effect is not allow-listed")
    positive = bool(projection.get("positive", False))
    authority = bool(projection.get("positive_authority", False))
    if positive and (not authority or effect == "none" or modality in _WEAK_MODALITIES):
        raise ValueError("weak or unauthoritative oracle signal cannot be a confirmed positive")
    signals = _bounded_mapping(dict(projection.get("signals") or {}), "oracle_projection.signals")
    safety = dict(projection.get("safety") or {})
    if any(key not in safety for key in _SAFETY_FLAGS):
        raise ValueError("oracle projection safety attestation is incomplete")
    unsafe = [key for key in _SAFETY_FLAGS if bool(safety.get(key, False))]
    if unsafe:
        raise ValueError("oracle projection records unsafe side effects: " + ", ".join(unsafe))
    normalized = {
        "oracle_id": _require_id(projection.get("oracle_id"), "oracle_id"),
        "oracle_contract_sha256": contract_hash,
        "family": _require_id(projection.get("family"), "oracle family"),
        "modality": modality,
        "candidate_signal": bool(projection.get("candidate_signal", False)),
        "positive": positive,
        "positive_authority": authority,
        "confirmed_effect": effect,
        "signals": signals,
        "safety": {key: False for key in _SAFETY_FLAGS},
    }
    normalized["projection_sha256"] = sha256_json(normalized)
    declared = projection.get("projection_sha256")
    if declared is not None and _require_hash(declared, "oracle projection_sha256") != normalized["projection_sha256"]:
        raise ValueError("oracle projection hash mismatch")
    return normalized


def validate_rule_ir_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("Rule IR manifest must be an object")
    if bool(manifest.get("executable", True)):
        raise ValueError("cross-lab Rule IR must remain non-executable")
    required = [_require_id(item, "Rule IR slot") for item in manifest.get("required_slots", [])]
    bound = [_require_id(item, "Rule IR bound slot") for item in manifest.get("bound_slots", [])]
    if len(required) != len(set(required)) or len(bound) != len(set(bound)) or not set(bound).issubset(required):
        raise ValueError("Rule IR slot binding is inconsistent")
    operators = sorted({_require_id(item, "Rule IR operator") for item in manifest.get("operator_set", [])})
    if not operators or len(operators) > 16:
        raise ValueError("Rule IR operator set is empty or oversized")
    normalized = {
        "schema_version": RULE_IR_MANIFEST_SCHEMA,
        "rule_key": _require_id(manifest.get("rule_key"), "Rule IR rule_key"),
        "grammar_version": _require_id(manifest.get("grammar_version"), "Rule IR grammar_version"),
        "family_candidate": _require_id(manifest.get("family_candidate"), "Rule IR family_candidate"),
        "operator_set": operators,
        "required_slots": required,
        "bound_slots": bound,
        "binding_status": "bound" if required and set(required) == set(bound) else "partial" if bound else "unbound",
        "executable": False,
    }
    normalized["rule_ir_sha256"] = sha256_json(normalized)
    return normalized


def validate_negative_control(control: dict[str, Any] | None) -> dict[str, Any] | None:
    if control is None:
        return None
    if not isinstance(control, dict):
        raise ValueError("negative control attestation must be an object")
    if str(control.get("verdict", "")) != "confirmed_negative":
        raise ValueError("negative control must carry a confirmed-negative verdict")
    if not bool(control.get("same_source")) or not bool(control.get("same_surface")):
        raise ValueError("negative control must be matched on source and surface")
    return {
        "control_sample_id": _require_id(control.get("control_sample_id"), "negative control sample id"),
        "control_evidence_hash": _require_hash(control.get("control_evidence_hash"), "negative control evidence hash"),
        "intervention": _require_id(control.get("intervention"), "negative control intervention"),
        "verdict": "confirmed_negative",
        "same_source": True,
        "same_surface": True,
    }


def _derive_decision(
    *,
    source: dict[str, Any],
    sample_role: str,
    response: dict[str, Any],
    oracle: dict[str, Any],
    rule_ir: dict[str, Any],
    control: dict[str, Any] | None,
) -> dict[str, Any]:
    abstain_reasons: list[str] = []
    if not source["registry"]["training_eligible"]:
        abstain_reasons.append("source_evaluation_only")
    if response["transport_error"]:
        abstain_reasons.append("transport_error")
    if rule_ir["binding_status"] != "bound":
        abstain_reasons.append("rule_ir_slots_unbound")
    if sample_role == "negative_control":
        if oracle["positive"]:
            raise ValueError("negative control cannot carry a positive oracle result")
        evidence_status = "confirmed_negative"
    elif oracle["positive"]:
        if control is None:
            abstain_reasons.append("negative_control_missing")
        evidence_status = "confirmed_positive" if not any(reason != "source_evaluation_only" for reason in abstain_reasons) else "abstain"
    elif oracle["candidate_signal"]:
        evidence_status = "candidate"
    else:
        abstain_reasons.append("insufficient_family_evidence")
        evidence_status = "abstain"
    return {
        "evidence_status": evidence_status,
        "training_action": "accept" if not abstain_reasons and evidence_status in {"confirmed_positive", "confirmed_negative"} else "abstain",
        "abstain_reasons": sorted(set(abstain_reasons)),
        "oracle_revalidated": evidence_status in {"confirmed_positive", "confirmed_negative"},
    }


class ReadOnlySafeCatalogCollector:
    """Bind already-bounded local observations; never contact a target."""

    def __init__(self, source: dict[str, Any], *, registry: dict[str, Any]) -> None:
        self.source = validate_source(source, registry=registry)

    def collect(
        self,
        *,
        sample_id: str,
        sample_role: str,
        sampling_seed: int,
        reset: dict[str, Any],
        payload_manifest: dict[str, Any],
        response_projection: dict[str, Any],
        oracle_projection: dict[str, Any],
        rule_ir: dict[str, Any],
        negative_control: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if sample_role not in {"candidate", "negative_control"}:
            raise ValueError("sample role must be candidate or negative_control")
        checked_reset = validate_reset(reset, self.source)
        checked_payload = validate_payload_manifest(payload_manifest)
        checked_response = validate_response_projection(response_projection)
        checked_oracle = validate_oracle_projection(oracle_projection, self.source)
        checked_ir = validate_rule_ir_manifest(rule_ir)
        checked_control = validate_negative_control(negative_control)
        decision = _derive_decision(
            source=self.source,
            sample_role=sample_role,
            response=checked_response,
            oracle=checked_oracle,
            rule_ir=checked_ir,
            control=checked_control,
        )
        binding_sha256 = sha256_json({
            "rule_ir_sha256": checked_ir["rule_ir_sha256"],
            "oracle_projection_sha256": checked_oracle["projection_sha256"],
            "reset_sha256": checked_reset["reset_sha256"],
            "manifest_sha256": checked_payload["manifest_sha256"],
        })
        evidence_body = {
            "schema_version": EVIDENCE_SCHEMA,
            "source": {
                "source_id": self.source["source_id"],
                "source_sha256": self.source["source_sha256"],
                "target_id": self.source["target_id"],
            },
            "sampling_seed": int(sampling_seed),
            "reset": checked_reset,
            "payload_manifest": checked_payload,
            "response_projection": checked_response,
            "oracle_projection": checked_oracle,
            "rule_ir_binding_sha256": binding_sha256,
            "negative_control": checked_control,
            "decision": decision,
            "safety": {
                "local_only": True,
                "read_only": True,
                "raw_body_stored": False,
                "credentials_stored": False,
                "attack_string_stored": False,
                "external_network": False,
            },
        }
        evidence_hash = sha256_json(evidence_body)
        evidence = {**evidence_body, "evidence_hash_algorithm": "sha256-canonical-json", "evidence_hash": evidence_hash}
        record = {
            "schema_version": SAMPLE_SCHEMA,
            "sample_id": _require_id(sample_id, "sample_id"),
            "sample_role": sample_role,
            "source_id": self.source["source_id"],
            "source_sha256": self.source["source_sha256"],
            "target_instance_id": checked_reset["target_instance_id"],
            "sampling_seed": int(sampling_seed),
            "reset": checked_reset,
            "payload_manifest": checked_payload,
            "response_projection": checked_response,
            "oracle_projection": checked_oracle,
            "rule_ir": {**checked_ir, "binding_sha256": binding_sha256},
            "negative_control": checked_control,
            "decision": decision,
            "evidence": evidence,
        }
        # A real Catalog round-trip cannot preserve Python object aliases.
        # Break them here as well so mutating a top-level projection never
        # mutates the hash-bound evidence projection in memory.
        return json.loads(canonical_json(record))


def validate_sample(record: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict) or record.get("schema_version") != SAMPLE_SCHEMA:
        raise ValueError("unsupported cross-lab sample schema")
    if str(record.get("source_sha256", "")) != source["source_sha256"]:
        raise ValueError("sample source attestation mismatch")
    evidence = dict(record.get("evidence") or {})
    declared = _require_hash(evidence.get("evidence_hash"), "sample evidence hash")
    body = dict(evidence)
    body.pop("evidence_hash", None)
    body.pop("evidence_hash_algorithm", None)
    if declared != sha256_json(body):
        raise ValueError("sample evidence hash mismatch")
    checked_reset = validate_reset(dict(record.get("reset") or {}), source)
    checked_payload = validate_payload_manifest(dict(record.get("payload_manifest") or {}))
    checked_response = validate_response_projection(dict(record.get("response_projection") or {}))
    checked_oracle = validate_oracle_projection(dict(record.get("oracle_projection") or {}), source)
    checked_ir = validate_rule_ir_manifest(dict(record.get("rule_ir") or {}))
    checked_control = validate_negative_control(record.get("negative_control"))
    if str((record.get("reset") or {}).get("reset_sha256", "")) != checked_reset["reset_sha256"]:
        raise ValueError("sample reset hash mismatch")
    if str((record.get("payload_manifest") or {}).get("manifest_sha256", "")) != checked_payload["manifest_sha256"]:
        raise ValueError("sample payload manifest hash mismatch")
    if str((record.get("response_projection") or {}).get("projection_sha256", "")) != checked_response["projection_sha256"]:
        raise ValueError("sample response projection hash mismatch")
    if str((record.get("oracle_projection") or {}).get("projection_sha256", "")) != checked_oracle["projection_sha256"]:
        raise ValueError("sample oracle projection hash mismatch")
    expected_decision = _derive_decision(
        source=source,
        sample_role=str(record.get("sample_role", "")),
        response=checked_response,
        oracle=checked_oracle,
        rule_ir=checked_ir,
        control=checked_control,
    )
    if canonical_json(record.get("decision")) != canonical_json(expected_decision):
        raise ValueError("sample decision is not derivable from bounded evidence")
    for key in ("reset", "payload_manifest", "response_projection", "oracle_projection", "negative_control", "decision"):
        if canonical_json(record.get(key)) != canonical_json(evidence.get(key)):
            raise ValueError(f"sample {key} is not bound to evidence")
    if str((evidence.get("source") or {}).get("source_sha256", "")) != source["source_sha256"]:
        raise ValueError("sample evidence source hash mismatch")
    if str((record.get("rule_ir") or {}).get("rule_ir_sha256", "")) != checked_ir["rule_ir_sha256"]:
        raise ValueError("sample Rule IR manifest hash mismatch")
    expected_binding = sha256_json({
        "rule_ir_sha256": checked_ir["rule_ir_sha256"],
        "oracle_projection_sha256": checked_oracle["projection_sha256"],
        "reset_sha256": checked_reset["reset_sha256"],
        "manifest_sha256": checked_payload["manifest_sha256"],
    })
    if expected_binding != str((record.get("rule_ir") or {}).get("binding_sha256", "")) or expected_binding != str(evidence.get("rule_ir_binding_sha256", "")):
        raise ValueError("sample Rule IR binding mismatch")
    return copy.deepcopy(record)


def build_catalog(catalog_id: str, source: dict[str, Any], records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [validate_sample(dict(row), source) for row in records]
    if not rows:
        raise ValueError("cross-lab catalog must contain samples")
    ids = [row["sample_id"] for row in rows]
    hashes = [row["evidence"]["evidence_hash"] for row in rows]
    if len(ids) != len(set(ids)) or len(hashes) != len(set(hashes)):
        raise ValueError("cross-lab catalog contains duplicate sample or evidence identities")
    by_id = {row["sample_id"]: row for row in rows}
    for row in rows:
        if row["decision"]["evidence_status"] != "confirmed_positive":
            continue
        control = row.get("negative_control") or {}
        control_row = by_id.get(str(control.get("control_sample_id", "")))
        if control_row is None or control_row["decision"]["evidence_status"] != "confirmed_negative":
            raise ValueError("confirmed positive is missing its catalog negative control")
        if control_row["evidence"]["evidence_hash"] != control.get("control_evidence_hash"):
            raise ValueError("negative control evidence hash mismatch")
        if control_row["source_sha256"] != row["source_sha256"]:
            raise ValueError("negative control source mismatch")
    body = {
        "schema_version": CATALOG_SCHEMA,
        "catalog_id": _require_id(catalog_id, "catalog_id"),
        "source": copy.deepcopy(source),
        "samples": rows,
        "training_eligible": bool(source["registry"]["training_eligible"]) and all(row["decision"]["training_action"] == "accept" for row in rows),
        "safety": {
            "loopback_only": True,
            "read_only": True,
            "raw_body_stored": False,
            "credentials_stored": False,
            "attack_string_stored": False,
            "external_network": False,
        },
    }
    body["catalog_sha256"] = sha256_json(body)
    return body


__all__ = [
    "CATALOG_SCHEMA",
    "EVIDENCE_SCHEMA",
    "REGISTRY_SCHEMA",
    "RULE_IR_MANIFEST_SCHEMA",
    "SAMPLE_SCHEMA",
    "SOURCE_SCHEMA",
    "ReadOnlySafeCatalogCollector",
    "build_catalog",
    "canonical_json",
    "registry_status",
    "sha256_json",
    "validate_oracle_projection",
    "validate_payload_manifest",
    "validate_reset",
    "validate_response_projection",
    "validate_rule_ir_manifest",
    "validate_sample",
    "validate_source",
]
