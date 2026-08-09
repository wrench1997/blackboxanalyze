"""Pure-memory PG-377 WebGoat whole-page source-row adapter.

The adapter is the narrow boundary between an authorised local WebGoat
collector and the PG-331A source-row contract.  It accepts only a bounded,
de-identified page projection (HTML is parsed and discarded in memory), an
abstract request/response projection, a role/reset attestation and an
evaluator-side typed sidecar.  The returned model view contains the seven
ontology axes and the complete 107-field capture manifest; evaluator evidence
is kept under an explicit sidecar key and never copied into context tokens.

This module deliberately does not make requests, start a container, bind a
port, read a route, or create a wire.  Missing page/transport/evaluator
observations remain ``not_observed`` and force an ASK-safe target.  The
PG-368 method-shape artifact is explicitly rejected as a source-row input so a
coarse canary cannot be promoted by accident.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .pg331_loopback_adapter import _field_capture_manifest
from .pg331_source_row import collect_pg331_source_row, sha256_json
from .pg331_vulnerableapp_adapter import capture_vulnerableapp_projection
from .pg331_web_tokenizer import tokenize_web_observation


SCHEMA_VERSION = "pg377-webgoat-whole-page-source-row-adapter-v1"
AXES = (
    "document_structure",
    "navigation",
    "request_transport",
    "response_transport",
    "javascript_surface",
    "failure_feedback",
    "belief_and_replay",
)
ROLES = ("candidate", "reference", "negative", "replay")
METHODS = ("GET", "POST")
ONTOLOGY_PATH = Path(__file__).resolve().parents[1] / "research" / "pg331_web_token_ontology_v1.json"
ONTOLOGY = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8-sig"))
FIELD_MANIFEST_FIELDS = {
    str(axis): tuple(str(field) for field in list(spec.get("fields") or []))
    for axis, spec in dict(ONTOLOGY.get("axes") or {}).items()
}
FIELD_COUNT = sum(len(fields) for fields in FIELD_MANIFEST_FIELDS.values())
if FIELD_COUNT != 107:  # keep the contract explicit if the ontology changes
    raise RuntimeError(f"PG-377 expected 107 ontology fields, found {FIELD_COUNT}")

_RAW_KEYS = frozenset(
    {
        "url",
        "uri",
        "origin",
        "path",
        "location",
        "location_url",
        "payload",
        "raw_payload",
        "wire",
        "request_body",
        "request_value",
        "query_value",
        "form_value",
        "response",
        "response_body",
        "raw_response",
        "body",
        "body_text",
        "markup",
        "source_code",
        "oracle_answer",
        "evaluator_answer",
        "credential",
        "authorization",
        "cookie_value",
    }
)
_RAW_FRAGMENTS = ("raw_", "payload", "response_body", "body_text", "wire", "oracle", "evaluator")
_SYMBOL = re.compile(r"^[a-z0-9_.:-]{1,96}$")
_JS_OVERLAY_KEYS = frozenset({
    "schema_version",
    "source_sha256",
    "source_text_stored",
    "script_count",
    "local_fixture",
    "javascript_context",
    "js_semantic_tokens",
    "next_action",
    "safe_to_send",
    "ask_reason",
})
_JS_OVERLAY_FORBIDDEN = ("http://", "https://", "wire=", "payload=", "response_body=", "<script", "javascript:")


def _normalize_js_context_overlay(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Keep only abstract JS labels; reject source/wire leakage."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("PG-377 javascript_context_projection must be an object or None")
    unknown = set(str(key) for key in value) - _JS_OVERLAY_KEYS
    if unknown:
        raise ValueError("PG-377 javascript_context_projection contains unsupported fields")
    result: dict[str, Any] = {}
    for key in _JS_OVERLAY_KEYS:
        if key not in value:
            continue
        item = value[key]
        if key == "script_count":
            if isinstance(item, bool) or not isinstance(item, int) or item < 0 or item > 128:
                raise ValueError("PG-377 javascript script count is invalid")
            result[key] = item
        elif key == "js_semantic_tokens":
            if not isinstance(item, (list, tuple)) or len(item) > 128:
                raise ValueError("PG-377 javascript semantic token sequence invalid")
            tokens = [str(token) for token in item]
            if any(len(token) > 160 or any(fragment in token.casefold() for fragment in _JS_OVERLAY_FORBIDDEN) for token in tokens):
                raise ValueError("PG-377 javascript semantic token firewall")
            result[key] = tokens
        elif key == "javascript_context":
            if not isinstance(item, Mapping):
                raise ValueError("PG-377 javascript_context must be an object")
            allowed_context = {"source_kind", "parser_kind", "normalization_chain", "filter_shape", "guard_shape", "control_flow_shape", "event_shape", "ast_shape", "source_to_sink_shape", "sink_context", "external_or_dynamic_loader", "persistent_state", "dynamic_code", "tokens"}
            if set(str(key2) for key2 in item) - allowed_context:
                raise ValueError("PG-377 javascript_context contains unsupported fields")
            nested: dict[str, Any] = {}
            for key2, value2 in item.items():
                if key2 == "tokens":
                    if not isinstance(value2, (list, tuple)) or any(len(str(token)) > 160 or any(fragment in str(token).casefold() for fragment in _JS_OVERLAY_FORBIDDEN) for token in value2):
                        raise ValueError("PG-377 javascript context token firewall")
                    nested[str(key2)] = [str(token) for token in value2]
                elif isinstance(value2, (str, bool, int, float)) or value2 is None:
                    nested[str(key2)] = value2
                elif isinstance(value2, (list, tuple)) and len(value2) <= 32:
                    nested[str(key2)] = [str(item2) for item2 in value2]
                else:
                    raise ValueError("PG-377 javascript context value invalid")
            result[key] = nested
        elif isinstance(item, (str, bool, int, float)) or item is None:
            if isinstance(item, str) and any(fragment in item.casefold() for fragment in _JS_OVERLAY_FORBIDDEN):
                raise ValueError("PG-377 javascript overlay firewall")
            result[key] = item
        else:
            raise ValueError("PG-377 javascript overlay value invalid")
    result.setdefault("source_text_stored", False)
    if result.get("source_text_stored") is not False:
        raise ValueError("PG-377 raw javascript source is not allowed")
    return result


def _reject_raw(value: Any, path: str = "$", *, allowed_keys: frozenset[str] = frozenset()) -> None:
    """Reject raw/evaluator fields in projections before parsing them."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).casefold()
            if lowered not in allowed_keys and (lowered in _RAW_KEYS or any(fragment in lowered for fragment in _RAW_FRAGMENTS)):
                raise ValueError(f"PG-377 raw/literal field rejected: {path}.{key}")
            _reject_raw(child, f"{path}.{key}", allowed_keys=allowed_keys)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_raw(child, f"{path}[{index}]", allowed_keys=allowed_keys)
    elif isinstance(value, (bytes, bytearray)):
        raise ValueError("PG-377 raw bytes are not accepted")


def _symbol(value: Any, default: str = "unknown") -> str:
    if value in (None, ""):
        return default
    text = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    return text if _SYMBOL.fullmatch(text) else default


def _flag(value: Any) -> str:
    if value is True:
        return "present"
    if value is False:
        return "absent"
    return "unknown"


def _method(request_projection: Mapping[str, Any] | None) -> str:
    if request_projection is None:
        return "unknown"
    method = str(request_projection.get("method", "")).upper()
    if method not in METHODS:
        raise ValueError("PG-377 request_projection.method must be GET or POST")
    return method


def _reset_projection(reset: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return only abstract reset facts; identifiers stay evaluator-side."""

    if reset is None:
        return {
            "fresh_reset": "unknown",
            "network_mode": "unknown",
            "external_network": "unknown",
            "loopback_only": "unknown",
            "state_clean": "unknown",
            "volume_mount_count": "unknown",
            "container_restart_used": "unknown",
            "attested": False,
        }
    if not isinstance(reset, Mapping):
        raise ValueError("PG-377 role reset must be an object or None")
    _reject_raw(reset, "reset")
    allowed = {
        "reset_id",
        "fresh_reset",
        "target_instance_digest",
        "network_mode",
        "external_network",
        "loopback_only",
        "state_clean",
        "volume_mount_count",
        "bind_or_volume_mount_count",
        "container_restart_used",
        "database_health_gate",
    }
    unknown = sorted(str(key) for key in reset if str(key) not in allowed)
    if unknown:
        raise ValueError(f"PG-377 reset contains unsupported fields: {', '.join(unknown)}")
    volume = reset.get("volume_mount_count", reset.get("bind_or_volume_mount_count", "unknown"))
    if volume != "unknown":
        try:
            volume = int(volume)
        except (TypeError, ValueError) as error:
            raise ValueError("PG-377 reset volume count is invalid") from error
        if volume < 0 or volume > 64:
            raise ValueError("PG-377 reset volume count is out of bounds")
    values = {
        "fresh_reset": reset.get("fresh_reset", "unknown") if isinstance(reset.get("fresh_reset", "unknown"), bool) else "unknown",
        "network_mode": _symbol(reset.get("network_mode")),
        "external_network": reset.get("external_network", "unknown") if isinstance(reset.get("external_network", "unknown"), bool) else "unknown",
        "loopback_only": reset.get("loopback_only", "unknown") if isinstance(reset.get("loopback_only", "unknown"), bool) else "unknown",
        "state_clean": reset.get("state_clean", "unknown") if isinstance(reset.get("state_clean", "unknown"), bool) else "unknown",
        "volume_mount_count": volume,
        "container_restart_used": reset.get("container_restart_used", "unknown") if isinstance(reset.get("container_restart_used", "unknown"), bool) else "unknown",
    }
    values["attested"] = bool(
        values["fresh_reset"] is True
        and values["network_mode"] in {"none", "loopback"}
        and values["external_network"] is False
        and values["loopback_only"] is True
        and values["state_clean"] is True
        and values["volume_mount_count"] == 0
        and values["container_restart_used"] is False
    )
    return values


def _sidecar_flags(sidecar: Mapping[str, Any] | None) -> dict[str, str]:
    """Derive belief/replay flags from a typed evaluator sidecar only."""

    if sidecar is None:
        return {key: "unknown" for key in (
            "typed_available",
            "evidence_present",
            "negative_control",
            "fresh_reset",
            "replay_ready",
            "reference_present",
            "candidate_present",
            "evidence_hash_present",
        )}
    if not isinstance(sidecar, Mapping):
        raise ValueError("PG-377 evaluator_sidecar must be an object or None")
    # These flags are safe boolean/hash attestations emitted by the existing
    # evaluator-side builder.  They are not copied to model context.
    sidecar_safe_keys = frozenset(
        {
            "evaluator_id",
            "evaluator_version",
            "raw_payload_stored",
            "raw_response_stored",
            "oracle_answer_in_context",
            "training_eligible",
            "memory_promotion_allowed",
            "payload_catalog_promotion_allowed",
            "vulnerability_claim_allowed",
            "evidence_sha256",
            "evidence_hash",
            "evidence_hash_valid",
        }
    )
    _reject_raw(sidecar, "evaluator_sidecar", allowed_keys=sidecar_safe_keys)
    for key in ("raw_payload_stored", "raw_response_stored", "oracle_answer_in_context"):
        if sidecar.get(key) is True:
            raise ValueError(f"PG-377 evaluator_sidecar {key} must be false")
    checks = sidecar.get("checks") if isinstance(sidecar.get("checks"), Mapping) else {}
    source = {**dict(checks), **{str(key): value for key, value in sidecar.items()}}
    return {
        "typed_available": _flag(source.get("typed_effect", source.get("typed_effect_confirmed", source.get("typed_available")))),
        "evidence_present": _flag(source.get("evidence_hashes", source.get("evidence_present"))),
        "negative_control": _flag(source.get("negative_control_clean", source.get("negative_control"))),
        "fresh_reset": _flag(source.get("fresh_reset")),
        "replay_ready": _flag(source.get("replay_consistent", source.get("replay_ready"))),
        "reference_present": _flag(source.get("reference_present")),
        "candidate_present": _flag(source.get("candidate_present")),
        "evidence_hash_present": _flag(bool(sidecar.get("evidence_sha256") or sidecar.get("evidence_hash"))),
    }


def _belief_projection(*, role: str, method: str, sidecar: Mapping[str, Any] | None, reset: Mapping[str, Any] | None, has_observation: bool) -> dict[str, Any]:
    flags = _sidecar_flags(sidecar)
    reset_info = _reset_projection(reset)
    # A provided sidecar is still evaluator-only.  These values are bounded
    # availability/status symbols, not the sidecar's effect answer.
    return {
        "observation_presence": "present" if has_observation else "unknown",
        "observation_delta_axis": "response_shape" if has_observation else "unknown",
        "belief_prior_bucket": "unknown",
        "belief_posterior_bucket": "unknown",
        "belief_delta_axis": "response_shape" if has_observation else "unknown",
        "history_action": "baseline_observe",
        **flags,
        "fresh_reset": _flag(reset_info["fresh_reset"] if reset_info["fresh_reset"] != "unknown" else None),
        "step_budget": "unknown",
        "history_length": 1,
        "probe_count": 0,
        "probe_role": role,
        "process_step": "replay" if role == "replay" else "baseline",
        "method": method,
    }


def _target_projection(*, method: str, role: str, sidecar: Mapping[str, Any] | None, reset: Mapping[str, Any] | None, field_capture_manifest: Mapping[str, Any], observation: Mapping[str, Any]) -> dict[str, Any]:
    missing_fields = any(status in {"not_observed", "unknown"} for section in field_capture_manifest.values() for status in section.values())
    flags = _sidecar_flags(sidecar)
    reset_info = _reset_projection(reset)
    typed_complete = all(flags[key] == "present" for key in ("typed_available", "evidence_present", "negative_control", "replay_ready", "reference_present", "candidate_present")) and reset_info["attested"]
    failure = observation.get("failure_feedback") if isinstance(observation.get("failure_feedback"), Mapping) else {}
    failure_class = str(failure.get("failure_class", "none")).casefold()
    previous = str(failure.get("previous_action", ""))
    next_action = str(failure.get("next_action", ""))
    stuck_failure = failure_class not in {"", "none", "unknown"} and previous and previous == next_action
    ask = missing_fields or not typed_complete or method not in METHODS
    if stuck_failure:
        return {
            "question": "ask_failure",
            "next_action": "repair",
            "repair_action": "observe",
            "transport_ref": "unknown",
            "field_role_ref": "unknown",
            "encoding_ref": "unknown",
            "probe_variant_ref": "none",
            "safe_to_send": False,
        }
    return {
        "question": "ask_typed" if ask else "none",
        "next_action": "ask_typed" if ask else "assemble_rule_ir",
        "repair_action": "observe" if ask else "none",
        "transport_ref": "post_surface" if method == "POST" else "get_surface" if method == "GET" else "unknown",
        "field_role_ref": "parameter_role" if method in METHODS else "unknown",
        "encoding_ref": "form_urlencoded" if method == "POST" else "url_percent" if method == "GET" else "unknown",
        "probe_variant_ref": "unknown" if role == "replay" else "source_attested_candidate" if role == "candidate" else "reference" if role == "reference" else "negative_control",
        "safe_to_send": False,
    }


def _context_presence(tokens: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in tokens:
        if "=" not in str(token):
            continue
        key, value = str(token).split("=", 1)
        if key.endswith("_presence"):
            result[key] = value
    return result


def _validate_method_shape_guard(value: Any) -> None:
    if value is True:
        raise ValueError("PG-377 refuses PG-368 method-shape-only rows")
    if isinstance(value, str) and "pg368" in value.casefold():
        raise ValueError("PG-377 refuses PG-368 method-shape-only rows")


def capture_pg377_webgoat_source_row(
    *,
    html: str | None,
    headers: Mapping[str, Any] | None = None,
    request_projection: Mapping[str, Any] | None = None,
    response_projection: Mapping[str, Any] | None = None,
    role: str = "candidate",
    reset: Mapping[str, Any] | None = None,
    role_reset: Mapping[str, Any] | None = None,
    evaluator_sidecar: Mapping[str, Any] | None = None,
    typed_sidecar: Mapping[str, Any] | None = None,
    failure_projection: Mapping[str, Any] | None = None,
    belief_projection: Mapping[str, Any] | None = None,
    javascript_context_projection: Mapping[str, Any] | None = None,
    post_supported: bool = True,
    method_shape_only: bool = False,
    source_meta: Mapping[str, Any] | None = None,
    record_id: str | None = None,
    split: str = "unassigned",
    operator_reviewed: bool = False,
    hard_negative: bool = False,
) -> dict[str, Any]:
    """Convert one de-identified WebGoat page/projection to an abstract row."""

    _validate_method_shape_guard(method_shape_only)
    if reset is not None and role_reset is not None:
        raise ValueError("PG-377 pass only one of reset or role_reset")
    if evaluator_sidecar is not None and typed_sidecar is not None:
        raise ValueError("PG-377 pass only one of evaluator_sidecar or typed_sidecar")
    reset = reset if reset is not None else role_reset
    evaluator_sidecar = evaluator_sidecar if evaluator_sidecar is not None else typed_sidecar
    if role not in ROLES:
        raise ValueError("PG-377 role must be candidate, reference, negative, or replay")
    if html is not None and not isinstance(html, str):
        raise ValueError("PG-377 html must be text or None")
    if html is not None and len(html) > 2 * 1024 * 1024:
        raise ValueError("PG-377 html exceeds bounded in-memory limit")
    for value, name in ((headers, "headers"), (request_projection, "request_projection"), (response_projection, "response_projection"), (failure_projection, "failure_projection"), (belief_projection, "belief_projection")):
        if value is not None:
            if not isinstance(value, Mapping):
                raise ValueError(f"PG-377 {name} must be an object or None")
            _reject_raw(value, name)
    if evaluator_sidecar is not None:
        _sidecar_flags(evaluator_sidecar)  # validate before any context is built
    javascript_overlay = _normalize_js_context_overlay(javascript_context_projection)
    method = _method(request_projection)
    projection = capture_vulnerableapp_projection(
        html=html,
        headers=headers,
        request_projection=request_projection,
        response_projection=response_projection,
        post_supported=bool(post_supported),
        failure_projection=failure_projection,
        belief_projection=belief_projection,
    )
    observation = dict(projection["observation"])
    # A sidecar supplied by the evaluator is the only source for replay state.
    # Keep an explicit belief axis even when it is incomplete, so the tokenizer
    # emits ``unknown``/``not_observed`` and the target remains ASK-safe.
    if belief_projection is None:
        observation["belief_and_replay"] = _belief_projection(role=role, method=method, sidecar=evaluator_sidecar, reset=reset, has_observation=any(value is not None for value in observation.values())) if evaluator_sidecar is not None or reset is not None else None
    else:
        observation["belief_and_replay"] = dict(belief_projection)
        _reject_raw(observation["belief_and_replay"], "belief_projection")
    tokenized = tokenize_web_observation(observation)
    context_tokens = [str(token) for token in tokenized.get("context_tokens") or []]
    field_manifest = _field_capture_manifest(observation)
    target = _target_projection(method=method, role=role, sidecar=evaluator_sidecar, reset=reset, field_capture_manifest=field_manifest, observation=observation)
    reset_projection = _reset_projection(reset)
    sidecar_copy = dict(evaluator_sidecar) if isinstance(evaluator_sidecar, Mapping) else None
    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_kind": "pg377_full_page_projection",
        "method": method,
        "method_observed": method in METHODS,
        "role": role,
        "observation": observation,
        "context_tokens": context_tokens,
        "axis_presence": _context_presence(context_tokens),
        "field_capture_manifest": field_manifest,
        "field_capture_manifest_count": sum(len(fields) for fields in field_manifest.values()),
        "target_projection": target,
        "typed_projection": {"typed_available": _sidecar_flags(evaluator_sidecar)["typed_available"] == "present", "safe_to_send": False, "evaluator_only": True},
        "evaluator_sidecar": sidecar_copy,
        "javascript_context_overlay": javascript_overlay,
        "reset_attestation": reset_projection,
        "raw_html_stored": False,
        "raw_url_stored": False,
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "wire_created": False,
        "target_contacted": False,
        "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True},
        "training_eligible": False,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "full_page_observation": bool(html and html.strip()),
        "method_shape_only": False,
    }
    # Optional strict source-row materialization is deliberately explicit and
    # operator-reviewed.  Most PG-377 calls stop at this diagnostic view.
    if source_meta is not None and reset is not None and evaluator_sidecar is not None and record_id is not None:
        if not isinstance(source_meta, Mapping):
            raise ValueError("PG-377 source_meta must be an object")
        _reject_raw(source_meta, "source_meta")
        checks = evaluator_sidecar.get("checks") if isinstance(evaluator_sidecar.get("checks"), Mapping) else {}
        evaluator_projection = {
            "typed_available": bool(checks.get("typed_effect", False)),
            "negative_control": bool(checks.get("negative_control_clean", False)),
            "reference_present": bool(checks.get("reference_present", False)),
            "candidate_present": bool(checks.get("candidate_present", False)),
            "fresh_reset": bool(checks.get("fresh_reset", False)),
            "evidence_hash": str(evaluator_sidecar.get("evidence_sha256", "0" * 64)),
            "confirmed_positive": bool(evaluator_sidecar.get("confirmed_positive", False)),
            "effect_class": _symbol(evaluator_sidecar.get("effect_class"), "unknown"),
            "evaluator_version": _symbol(evaluator_sidecar.get("evaluator_id"), "pg377_sidecar"),
        }
        # The PG-331 collector retains the strict reset/evaluator contract and
        # makes missing fields/failure transitions explicit ASK targets.
        strict_reset = {key: reset[key] for key in ("fresh_reset", "reset_id", "target_instance_digest", "network_mode", "external_network", "loopback_only", "state_clean") if key in reset}
        source_row = collect_pg331_source_row(record_id=str(record_id), observation=observation, source_meta=source_meta, reset=strict_reset, evaluator=evaluator_projection, field_capture_manifest=field_manifest, target_projection=target, split=split, operator_reviewed=operator_reviewed, hard_negative=hard_negative)
        if javascript_overlay is not None:
            source_row["javascript_context_overlay"] = javascript_overlay
            source_row["record_sha256"] = sha256_json({key: value for key, value in source_row.items() if key != "record_sha256"})
        output["source_row"] = source_row
        output["training_eligible"] = bool(source_row.get("training_eligible"))
    output["record_sha256"] = sha256_json(output)
    return output


def validate_pg377_webgoat_source_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the adapter contract without contacting a target."""

    failures: list[str] = []
    if not isinstance(row, Mapping):
        return {"valid": False, "failures": ["row_not_mapping"]}
    if row.get("schema_version") != SCHEMA_VERSION:
        failures.append("schema_version")
    if row.get("source_kind") != "pg377_full_page_projection" or row.get("method_shape_only") is not False:
        failures.append("source_kind")
    try:
        _normalize_js_context_overlay(row.get("javascript_context_overlay"))
    except (TypeError, ValueError):
        failures.append("javascript_context_overlay")
    if row.get("role") not in ROLES:
        failures.append("role")
    method = row.get("method")
    if method not in (*METHODS, "unknown"):
        failures.append("method")
    observation = row.get("observation")
    if not isinstance(observation, Mapping) or set(observation) != set(AXES):
        failures.append("axes")
    manifest = row.get("field_capture_manifest")
    if not isinstance(manifest, Mapping) or set(str(key) for key in manifest) != set(FIELD_MANIFEST_FIELDS):
        failures.append("field_capture_manifest_axes")
    else:
        count = 0
        for axis, fields in FIELD_MANIFEST_FIELDS.items():
            section = manifest.get(axis)
            if not isinstance(section, Mapping) or set(str(key) for key in section) != set(fields):
                failures.append(f"field_capture_manifest:{axis}")
                continue
            count += len(section)
            if any(str(value) not in {"observed", "absent", "not_observed", "unknown"} for value in section.values()):
                failures.append(f"field_status:{axis}")
        if count != FIELD_COUNT or row.get("field_capture_manifest_count") != FIELD_COUNT:
            failures.append("field_count")
    if row.get("raw_html_stored") is not False or row.get("raw_url_stored") is not False or row.get("raw_payload_stored") is not False or row.get("raw_response_body_stored") is not False:
        failures.append("raw_storage")
    context_tokens = row.get("context_tokens")
    if not isinstance(context_tokens, list):
        failures.append("context_tokens")
    else:
        forbidden_context = ("http://", "https://", "payload=", "response_body=", "wire=", "oracle=", "evaluator=")
        if any(any(fragment in str(token).casefold() for fragment in forbidden_context) for token in context_tokens):
            failures.append("context_firewall")
    if row.get("wire_created") is not False or row.get("target_contacted") is not False:
        failures.append("execution_state")
    if row.get("context_firewall") != {"forbidden_token_count": 0, "sidecars_off_context": True}:
        failures.append("context_firewall")
    target = row.get("target_projection")
    if not isinstance(target, Mapping) or target.get("safe_to_send") is not False:
        failures.append("ask_safe_target")
    if row.get("full_page_observation") is not True:
        failures.append("full_page_observation")
    sidecar = row.get("evaluator_sidecar")
    if isinstance(sidecar, Mapping):
        try:
            _sidecar_flags(sidecar)
        except (TypeError, ValueError):
            failures.append("evaluator_sidecar")
    expected_hash = str(row.get("record_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        failures.append("record_sha256")
    else:
        body = dict(row)
        body.pop("record_sha256", None)
        if sha256_json(body) != expected_hash:
            failures.append("record_hash_mismatch")
    return {"valid": not failures, "failures": sorted(set(failures)), "field_count": FIELD_COUNT, "training_eligible": bool(row.get("training_eligible"))}


# Descriptive aliases keep callers independent of the historical ``capture``
# naming used by PG-331 adapters.
adapt_pg377_webgoat_source_row = capture_pg377_webgoat_source_row
build_pg377_webgoat_source_row = capture_pg377_webgoat_source_row
collect_pg377_source_row = capture_pg377_webgoat_source_row
capture_pg377_source_row = capture_pg377_webgoat_source_row
adapt_webgoat_source_row = capture_pg377_webgoat_source_row
capture_pg377_webgoat_projection = capture_pg377_webgoat_source_row
build_pg377_source_row = capture_pg377_webgoat_source_row


__all__ = [
    "AXES",
    "FIELD_COUNT",
    "FIELD_MANIFEST_FIELDS",
    "METHODS",
    "ROLES",
    "SCHEMA_VERSION",
    "adapt_pg377_webgoat_source_row",
    "adapt_webgoat_source_row",
    "build_pg377_webgoat_source_row",
    "capture_pg377_webgoat_source_row",
    "capture_pg377_webgoat_projection",
    "capture_pg377_source_row",
    "collect_pg377_source_row",
    "build_pg377_source_row",
    "validate_pg377_webgoat_source_row",
]
