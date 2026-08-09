"""PG-287-live evaluator projection boundary.

This module is the bridge between a target-side, authorised evaluator and the
PG-287 identifiability task.  It accepts only the already bounded PG-286
observation record, an abstract reference plan, and an attested observation of
the encoding/field role.  It never accepts a literal payload, request value,
response body, route family label, or oracle label in model context.

The output is still a collection candidate.  Cross-seed/source coverage and an
independent batch audit are required before any training or memory promotion.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "pg287-live-identifiability-collection-v1"
ENCODINGS = frozenset({"plain", "url_percent", "form_urlencoded", "json", "xml", "base64", "unknown"})
FIELD_ROLES = frozenset({"numeric", "url", "control", "text", "opaque", "none", "unknown"})
PROBE_CLASSES = frozenset({"sql", "xss", "redirect", "logic", "other"})
CHANNELS = frozenset({"query", "form", "body", "path", "header", "cookie", "unknown"})
WIRE_KINDS = frozenset({"query_param", "form_field", "body_field", "path_segment", "header_field", "cookie_value", "none"})
REPAIR_DELTAS = frozenset({"none", "encoding", "channel", "field", "method", "observe", "unknown"})
FINAL_ACTIONS = frozenset({"ask_typed", "negative_control", "candidate_probe", "repair", "abstain", "reference_probe"})
FIELD_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
HEX_DIGEST_RE = re.compile(r"^(?:sha256:)?[a-f0-9]{64}$")
RAW_KEYS = frozenset({
    "payload", "raw_payload", "payload_value", "probe_value", "request_body",
    "response_body", "raw_response", "body_text", "html", "query_value",
    "form_value", "cookie", "authorization_header", "credential", "location", "url",
})
FORBIDDEN_CONTEXT = ("family=", "oracle=", "typed_effect=", "positive=", "payload=", "literal=", "<script", "javascript:")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _contains_raw(value: Any, key: str = "") -> bool:
    if key.casefold() in RAW_KEYS:
        return True
    if isinstance(value, Mapping):
        return any(_contains_raw(child, str(child_key)) for child_key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_raw(child, key) for child in value)
    return False


def _digest(value: Any) -> str:
    return sha256_json(value)


def _required_digest(value: Any, label: str) -> str:
    digest = str(value or "")
    if not HEX_DIGEST_RE.fullmatch(digest):
        raise ValueError(f"PG-287 {label} must be a sha256 digest")
    return digest


def _attestation(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or _contains_raw(value):
        raise ValueError("PG-287 source attestation is missing or contains raw material")
    allowed = {"authorized", "authorization_id", "target_instance_digest", "image_digest", "source_digest", "collector_id"}
    unknown = sorted(str(key) for key in value if str(key) not in allowed)
    if unknown:
        raise ValueError(f"PG-287 source attestation contains unsupported fields: {', '.join(unknown)}")
    if value.get("authorized") is not True:
        raise ValueError("PG-287 source attestation must explicitly be authorized")
    result = {str(key): value[key] for key in allowed if key in value}
    for key in ("target_instance_digest", "image_digest", "source_digest"):
        result[key] = _required_digest(result.get(key), key)
    for key in ("authorization_id", "collector_id"):
        if not isinstance(result.get(key), str) or not result[key] or len(result[key]) > 96:
            raise ValueError(f"PG-287 {key} is required")
    return result


def _plan(value: Mapping[str, Any], method: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or _contains_raw(value):
        raise ValueError("PG-287 reference plan is missing or contains raw material")
    allowed = {"next_action", "final_action", "method", "probe_class", "channel", "encoding", "wire_kind", "field_slot", "repair_delta", "safe_to_send", "oracle_required"}
    unknown = sorted(str(key) for key in value if str(key) not in allowed)
    if unknown:
        raise ValueError(f"PG-287 reference plan contains unsupported fields: {', '.join(unknown)}")
    action = str(value.get("final_action", value.get("next_action", "")))
    if action not in FINAL_ACTIONS:
        raise ValueError("PG-287 reference plan action is not allow-listed")
    plan_method = str(value.get("method", method)).upper()
    if plan_method not in {"GET", "POST"} or plan_method != method:
        raise ValueError("PG-287 reference plan method must match the observed surface")
    probe = str(value.get("probe_class", "other"))
    channel = str(value.get("channel", "unknown"))
    encoding = str(value.get("encoding", "unknown"))
    wire = str(value.get("wire_kind", "none"))
    repair = str(value.get("repair_delta", "none"))
    if probe not in PROBE_CLASSES or channel not in CHANNELS or encoding not in ENCODINGS or wire not in WIRE_KINDS or repair not in REPAIR_DELTAS:
        raise ValueError("PG-287 reference plan contains an unsupported slot")
    field_slot = str(value.get("field_slot", "unknown"))
    if field_slot not in {"unknown", "observed_or_runtime_canary", "query_param", "form_field", "body_field", "path_segment", "header_field", "cookie_value"}:
        raise ValueError("PG-287 field_slot is not allow-listed")
    safe = value.get("safe_to_send", False)
    if not isinstance(safe, bool):
        raise ValueError("PG-287 safe_to_send must be boolean")
    if action == "ask_typed" and safe:
        raise ValueError("PG-287 ask_typed must not be safe_to_send")
    return {
        "final_action": action,
        "method": plan_method,
        "probe_class": probe,
        "channel": channel,
        "encoding": encoding,
        "wire_kind": wire,
        "field_slot": field_slot,
        "repair_delta": repair,
        "safe_to_send": safe,
        "oracle_required": bool(value.get("oracle_required", True)),
    }


def _target_tokens(plan: Mapping[str, Any], *, ambiguous: bool) -> list[str]:
    method = str(plan["method"])
    if ambiguous:
        plan = {**plan, "final_action": "ask_typed", "probe_class": "other", "channel": "unknown", "encoding": "unknown", "wire_kind": "none", "field_slot": "unknown", "repair_delta": "none", "safe_to_send": False}
    return [
        "[TARGET_BOS]",
        f"plan={plan['final_action']}",
        f"method={method}",
        f"probe_class={plan['probe_class']}",
        f"channel={plan['channel']}",
        f"encoding={plan['encoding']}",
        f"wire={plan['wire_kind']}",
        f"field_slot={plan['field_slot']}",
        f"repair_delta={plan['repair_delta']}",
        "family_agnostic=1",
        f"final_action={plan['final_action']}",
        f"safe_to_send={int(bool(plan['safe_to_send']))}",
        "[TARGET_EOS]",
    ]


def collect_pg287_live_record(
    *,
    observation_record: Mapping[str, Any],
    observed_binding: Mapping[str, Any],
    reference_plan: Mapping[str, Any],
    source_attestation: Mapping[str, Any],
    remote_probe: Mapping[str, Any],
    split: str = "unassigned",
    hard_negative: bool = False,
    operator_reviewed: bool = False,
) -> dict[str, Any]:
    """Turn one live PG-286 observation into a bounded PG-287 candidate."""

    if not isinstance(observation_record, Mapping) or _contains_raw(observation_record):
        raise ValueError("PG-287 observation record is missing or contains raw material")
    if str(observation_record.get("schema_version", "")) != "pg286-live-observation-collection-v1":
        raise ValueError("PG-287 requires a PG-286 live observation record")
    if str(remote_probe.get("status", "unavailable")) != "available":
        raise ValueError("PG-287 live collection requires an available authorized remote Docker probe")
    if observation_record.get("token_evidence_status") != "complete" or observation_record.get("decision") != "eligible_for_cross_seed_review":
        raise ValueError("PG-287 live observation is not a complete cross-seed candidate")
    if observation_record.get("raw_payload_stored") or observation_record.get("raw_response_body_stored"):
        raise ValueError("PG-287 raw material flags must be false")
    surface = observation_record.get("surface")
    reset = observation_record.get("reset")
    if not isinstance(surface, Mapping) or not isinstance(reset, Mapping):
        raise ValueError("PG-287 surface and reset projections are required")
    method = str(surface.get("method", "")).upper()
    if method not in {"GET", "POST"}:
        raise ValueError("PG-287 live surface method must be GET or POST")
    if reset.get("fresh_target") is not True or not str(reset.get("reset_id", "")):
        raise ValueError("PG-287 live record requires a fresh reset id")
    attested = _attestation(source_attestation)
    if not isinstance(observed_binding, Mapping) or _contains_raw(observed_binding):
        raise ValueError("PG-287 observed binding is missing or contains raw material")
    binding_keys = {"encoding", "field_role", "evidence_hash", "observation_id"}
    if any(str(key) not in binding_keys for key in observed_binding):
        raise ValueError("PG-287 observed binding contains unsupported fields")
    encoding = str(observed_binding.get("encoding", "unknown"))
    field_role = str(observed_binding.get("field_role", "unknown"))
    if encoding not in ENCODINGS or field_role not in FIELD_ROLES:
        raise ValueError("PG-287 observed binding slot is not allow-listed")
    evidence_hash = _required_digest(observation_record.get("evidence_hash"), "observation evidence")
    if str(observed_binding.get("evidence_hash", evidence_hash)) != evidence_hash:
        raise ValueError("PG-287 observed binding must bind to the PG-286 evidence hash")
    if not isinstance(observed_binding.get("observation_id", ""), str) or not observed_binding["observation_id"]:
        raise ValueError("PG-287 observation_id is required")
    plan = _plan(reference_plan, method)
    ambiguous = encoding == "unknown" or field_role in {"unknown", "none"}
    if ambiguous and plan["final_action"] != "ask_typed":
        raise ValueError("PG-287 ambiguous observation must have ask_typed reference target")
    if not ambiguous and plan["final_action"] == "ask_typed":
        raise ValueError("PG-287 resolved observation cannot have ask_typed target")
    split = str(split)
    if split not in {"train", "route_dev", "family_holdout", "unassigned"}:
        raise ValueError("PG-287 split is not allow-listed")
    base_tokens = [str(token) for token in list(observation_record.get("context_tokens") or [])]
    if not base_tokens or "ir_family_agnostic=1" not in base_tokens or "[CTX_END]" not in base_tokens:
        raise ValueError("PG-287 observation context is incomplete")
    if any(any(bad.casefold() in token.casefold() for bad in FORBIDDEN_CONTEXT) for token in base_tokens):
        raise ValueError("PG-287 observation context contains forbidden label/literal material")
    if any(token.startswith(("encoding_observed=", "observation_sufficiency=", "observed_field_role=")) for token in base_tokens):
        raise ValueError("PG-287 observation context already contains identifiability binding")
    context_tokens = [token for token in base_tokens if token != "[CTX_END]"]
    context_tokens.extend([f"encoding_observed={encoding}", f"observed_field_role={field_role}", f"observation_sufficiency={'ambiguous' if ambiguous else 'resolved'}", "[CTX_END]"])
    if any(any(bad.casefold() in token.casefold() for bad in FORBIDDEN_CONTEXT) for token in context_tokens):
        raise ValueError("PG-287 generated context contains forbidden material")
    record = {
        "schema_version": SCHEMA_VERSION,
        "record_id": f"pg287-live:{observation_record.get('record_id', 'unknown')}:{observed_binding['observation_id']}",
        "source_record_id": str(observation_record.get("record_id", "")),
        "source_group_id": _digest({"source_digest": attested["source_digest"], "target_instance_digest": attested["target_instance_digest"]}),
        "source": "pg287_live_evaluator",
        "split": split,
        "variant": "ambiguous" if ambiguous else "resolved",
        "method": method,
        "context_tokens": context_tokens,
        "target_tokens": _target_tokens(plan, ambiguous=ambiguous),
        "target": {**plan, "next_action": plan["final_action"], "encoding": "unknown" if ambiguous else plan["encoding"], "oracle_required": True},
        "hard_negative": bool(hard_negative),
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "oracle_label_in_context": False,
        "literal_probe_in_context": False,
        "training_eligible": bool(operator_reviewed and not hard_negative),
        "memory_promotion_allowed": False,
        "source_evidence_hash": evidence_hash,
        "source_attestation_sha256": _digest(attested),
        "observed_binding_sha256": _digest({"encoding": encoding, "field_role": field_role, "evidence_hash": evidence_hash, "observation_id": observed_binding["observation_id"]}),
        "quality": {
            "remote_docker_status": str(remote_probe.get("status")),
            "fresh_reset": True,
            "reset_id": str(reset.get("reset_id")),
            "operator_reviewed": bool(operator_reviewed),
            "observation_complete": True,
            "source_authorized": True,
            "typed_effect": str(observation_record.get("typed_effect_type", "")),
        },
        "promotion": {
            "training_eligible": bool(operator_reviewed and not hard_negative),
            "cross_seed_review_required": True,
            "independent_audit_required": True,
            "memory_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
    }
    record["record_sha256"] = _digest(record)
    return record


__all__ = ["SCHEMA_VERSION", "collect_pg287_live_record", "sha256_json"]
