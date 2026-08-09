"""Evidence binding for abstract Rule IR slots.

Binding is intentionally a separate layer from the abstract AST.  A shadow
probe can bind an observation reference (status/body shape/query) but cannot
prove evaluator state or a trusted policy that was never visible.  Unavailable
slots therefore remain explicitly ``unbound`` instead of being guessed.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Iterator
from urllib.parse import parse_qs, unquote, urlsplit

from .rule_ir_decoder import validate_abstract_rule_ir


def iter_policy_slots(expr: dict[str, Any]) -> Iterator[str]:
    if not isinstance(expr, dict):
        return
    if expr.get("op") == "policy_slot":
        name = expr.get("name")
        if isinstance(name, str):
            yield name
        return
    for value in expr.values():
        if isinstance(value, dict):
            yield from iter_policy_slots(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield from iter_policy_slots(item)


def shadow_evidence(action: dict[str, Any], raw: dict[str, Any], projection: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a bounded, target-neutral evidence envelope from one shadow row."""

    path = str(action.get("path", ""))
    parsed = urlsplit(path)
    query = parse_qs(parsed.query, keep_blank_values=True)
    selected_headers = {
        str(key): str(value)
        for key, value in dict(raw.get("headers") or {}).items()
        if str(key).casefold() in {"content-type", "content-length", "location"}
    }
    return {
        "source": "local_shadow_probe",
        "action": {"method": str(action.get("method", "GET")).upper(), "path": path},
        "status_code": int(raw.get("status_code", 0) or 0),
        "headers": selected_headers,
        "body_length": int(raw.get("body_length", 0) or 0),
        "body_shape": str(raw.get("body_shape") or (projection or {}).get("body_shape") or "unknown"),
        "transport_error": raw.get("transport_error"),
        "query": {key: [unquote(item) for item in values[:4]] for key, values in query.items()},
    }


def evidence_digest(evidence: dict[str, Any]) -> str:
    payload = json.dumps(evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _query_value(evidence: dict[str, Any], names: tuple[str, ...]) -> str | None:
    query = evidence.get("query") or {}
    for name in names:
        values = query.get(name)
        if values:
            return str(values[0])
    return None


def _slot_binding(slot: str, evidence: dict[str, Any]) -> dict[str, Any]:
    status_code = int(evidence.get("status_code", 0) or 0)
    shape = str(evidence.get("body_shape", "unknown"))
    path = str((evidence.get("action") or {}).get("path", ""))
    sensitive_shapes = {"prometheus", "diagnostic", "traceback", "directory_listing", "source_map", "security_policy_text"}
    response_slots = {
        "sensitive_operational_artifact_is_public",
        "unsafe_or_deprecated_surface_is_enabled",
        "sensitive_data_visible_without_need",
        "subject_authenticated",
        "identity_proof_valid",
        "subject_authorized_for_resource",
        "credential_policy_satisfied",
        "value_is_in_declared_domain",
    }
    if slot in response_slots and (evidence.get("transport_error") or status_code <= 0):
        return {
            "status": "unbound",
            "value": None,
            "evidence_ref": None,
            "reason": "transport did not produce a trustworthy HTTP response",
        }
    if slot == "sensitive_operational_artifact_is_public":
        return {"status": "bound", "value": shape in sensitive_shapes, "evidence_ref": "body_shape", "reason": "bounded response-shape observation"}
    if slot == "unsafe_or_deprecated_surface_is_enabled":
        return {"status": "bound", "value": status_code in {200, 301, 302, 307, 308}, "evidence_ref": "status_code", "reason": "surface returned an HTTP response; safety classification remains separate"}
    if slot == "sensitive_data_visible_without_need":
        return {"status": "bound", "value": shape in {"directory_listing", "diagnostic", "source_map"}, "evidence_ref": "body_shape", "reason": "bounded exposure-shape observation"}
    if slot in {"subject_authenticated", "identity_proof_valid"}:
        return {"status": "bound", "value": status_code not in {401, 403}, "evidence_ref": "status_code", "reason": "HTTP denial/acceptance signal; not an evaluator-state claim"}
    if slot == "subject_authorized_for_resource":
        return {"status": "bound", "value": status_code not in {401, 403}, "evidence_ref": "status_code", "reason": "resource response signal; authorization remains provisional"}
    if slot == "credential_policy_satisfied":
        return {"status": "bound", "value": status_code not in {400, 401, 403}, "evidence_ref": "status_code", "reason": "credential endpoint response signal"}
    if slot == "representation_is_canonical":
        return {"status": "bound", "value": "%25" not in path and "%2525" not in path, "evidence_ref": "action.path", "reason": "bounded encoding-shape check"}
    if slot == "value_is_in_declared_domain":
        return {"status": "bound", "value": status_code not in {400, 422, 500}, "evidence_ref": "status_code", "reason": "input response signal"}
    if slot == "untrusted_data_cannot_change_interpreter_structure":
        return {"status": "bound", "value": not any(token in path.casefold() for token in ("union", "or%201%3d1", "--")), "evidence_ref": "action.path", "reason": "payload marker observation; no exploit claim"}
    if slot == "candidate_url":
        value = _query_value(evidence, ("to", "url", "next", "redirect"))
        return {"status": "bound" if value is not None else "unbound", "value": value, "evidence_ref": "query", "reason": "redirect candidate extracted from local action"}
    if slot == "trusted_origin":
        return {"status": "unbound", "value": None, "evidence_ref": None, "reason": "trusted policy origin is not visible in a shadow response"}
    if slot == "untrusted_text":
        value = _query_value(evidence, ("q", "query", "search", "payload"))
        return {"status": "bound" if value is not None else "unbound", "value": value, "evidence_ref": "query", "reason": "bounded request payload extraction"}
    return {"status": "unbound", "value": None, "evidence_ref": None, "reason": "slot has no direct shadow-visible binding"}


def bind_rule_ir_slots(rule_ir: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    """Bind every policy slot to evidence or explicitly mark it unavailable."""

    validate_abstract_rule_ir(rule_ir)
    slots = list(dict.fromkeys(iter_policy_slots(rule_ir)))
    bindings = {slot: _slot_binding(slot, evidence) for slot in slots}
    bound_count = sum(item["status"] == "bound" for item in bindings.values())
    return {
        "schema_version": "sift-rule-ir-evidence-binding-v1",
        "status": "fully_bound" if bound_count == len(bindings) else "partially_bound",
        "abstract_rule_ir": copy.deepcopy(rule_ir),
        "bindings": bindings,
        "bound_slot_count": bound_count,
        "slot_count": len(bindings),
        "evidence": evidence,
        "evidence_hash": evidence_digest(evidence),
        "evidence_hash_algorithm": "sha256-canonical-json",
        "executable": False,
    }


def validate_binding(binding: dict[str, Any]) -> dict[str, Any]:
    """Recompute binding derivations and provenance before persistence."""

    if binding.get("schema_version") != "sift-rule-ir-evidence-binding-v1":
        raise ValueError("unsupported Rule IR binding schema")
    evidence = binding.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("binding evidence must be an object")
    if binding.get("evidence_hash") != evidence_digest(evidence):
        raise ValueError("binding evidence hash mismatch")
    if binding.get("evidence_hash_algorithm") != "sha256-canonical-json":
        raise ValueError("unsupported binding evidence hash algorithm")
    rule_ir = binding.get("abstract_rule_ir")
    if not isinstance(rule_ir, dict):
        raise ValueError("binding abstract Rule IR must be an object")
    validate_abstract_rule_ir(rule_ir)
    slots = list(dict.fromkeys(iter_policy_slots(rule_ir)))
    expected_bindings = {slot: _slot_binding(slot, evidence) for slot in slots}
    bindings = binding.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError("binding slots must be an object")
    if bindings != expected_bindings:
        raise ValueError("binding derivation mismatch")
    bound_count = sum(isinstance(item, dict) and item.get("status") == "bound" for item in bindings.values())
    if binding.get("bound_slot_count") != bound_count or binding.get("slot_count") != len(bindings):
        raise ValueError("binding slot counts are inconsistent")
    expected_status = "fully_bound" if bound_count == len(bindings) else "partially_bound"
    if binding.get("status") != expected_status or binding.get("executable") is not False:
        raise ValueError("binding status is inconsistent")
    return {"valid": True, "bound_slot_count": bound_count, "slot_count": len(bindings), "evidence_hash": binding["evidence_hash"]}
