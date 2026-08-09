"""PG-302 symbolic slot-reference assembly and deterministic binding.

The causal model predicts *which observable slot to copy* rather than
memorizing every method/field/encoding combination.  Binding is a transparent
Rule-IR operation; it only yields abstract values and never a wire string.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .pg293_failure_next_action import TARGET_BOS, TARGET_EOS, sha256_json
from .pg301_payload_assembly import TARGET_KEYS, audit_assembly_records, assembly_target_for_context, canonical_assembly_context, target_map


SCHEMA_VERSION = "pg302-symbolic-assembly-v1"
SYMBOLIC_TARGET_KEYS = ("question", "next_action", "repair_action", "transport_ref", "field_role_ref", "encoding_ref", "canary", "oracle", "stop_condition", "safe_to_send")
_REFS = frozenset({"surface_method", "surface_field_role", "surface_encoding", "none"})


def symbolic_target_for_context(tokens: Sequence[str]) -> list[str]:
    base = target_map(assembly_target_for_context(tokens))
    transport_ref = "surface_method" if base.get("transport") in {"GET", "POST"} else "none"
    field_ref = "surface_field_role" if base.get("field_role") not in {None, "unknown", "none"} else "none"
    encoding_ref = "surface_encoding" if base.get("encoding") not in {None, "unknown", "none"} else "none"
    values = {
        "question": base.get("question", "none"),
        "next_action": base.get("next_action", "abstain"),
        "repair_action": base.get("repair_action", "none"),
        "transport_ref": transport_ref,
        "field_role_ref": field_ref,
        "encoding_ref": encoding_ref,
        "canary": base.get("canary", "none"),
        "oracle": base.get("oracle", "typed"),
        "stop_condition": base.get("stop_condition", "await_observation"),
        "safe_to_send": base.get("safe_to_send", "0"),
    }
    return [TARGET_BOS, *[f"{key}={values[key]}" for key in SYMBOLIC_TARGET_KEYS], TARGET_EOS]


def symbolic_record(row: Mapping[str, Any]) -> dict[str, Any]:
    context = canonical_assembly_context(list(row.get("context_tokens") or []))
    target = symbolic_target_for_context(context)
    values = target_map(target)
    projected = {
        "schema_version": SCHEMA_VERSION,
        "record_id": str(row.get("record_id") or sha256_json(context + target)[:16]),
        "split": str(row.get("split") or "train"),
        "training_eligible": bool(row.get("training_eligible", False)),
        "context_tokens": context,
        "target_tokens": target,
        "question": values.get("question", "none"),
        "safe_to_send": values.get("safe_to_send") == "1",
        "trace_step": int(row.get("trace_step", 0) or 0),
        "counterfactual_group": str(row.get("counterfactual_group") or "none"),
        "hard_negative": bool(row.get("hard_negative", False)),
        "record_sha256": "",
    }
    projected["record_sha256"] = sha256_json(projected)
    return projected


def _context_values(tokens: Sequence[str]) -> dict[str, str]:
    return target_map(canonical_assembly_context(tokens))


def bind_symbolic_plan(symbolic_tokens: Sequence[str], context_tokens: Sequence[str]) -> list[str] | None:
    """Bind slot references to bounded abstract values; reject malformed plans."""

    values = target_map(symbolic_tokens)
    if not symbolic_tokens or str(symbolic_tokens[0]) != TARGET_BOS or str(symbolic_tokens[-1]) != TARGET_EOS:
        return None
    if any(key not in values for key in SYMBOLIC_TARGET_KEYS):
        return None
    if any(values[key] not in _REFS for key in ("transport_ref", "field_role_ref", "encoding_ref")):
        return None
    context = _context_values(context_tokens)
    bound = {
        "question": values["question"],
        "next_action": values["next_action"],
        "repair_action": values["repair_action"],
        "transport": "none" if values["transport_ref"] == "none" else context.get(values["transport_ref"], "unknown"),
        "field_role": "unknown" if values["field_role_ref"] == "none" else context.get(values["field_role_ref"], "unknown"),
        "encoding": "unknown" if values["encoding_ref"] == "none" else context.get(values["encoding_ref"], "unknown"),
        "canary": values["canary"],
        "oracle": values["oracle"],
        "stop_condition": values["stop_condition"],
        "safe_to_send": values["safe_to_send"],
    }
    # Unknown/invalid bindings cannot be safe to send; retain the model's
    # decision only when every referenced slot resolved to an allowed value.
    if bound["transport"] not in {"GET", "POST", "none"} or bound["field_role"] not in {"query_param", "form_field", "header_value", "path_segment", "unknown"} or bound["encoding"] not in {"url_percent", "form_urlencoded", "json_string", "base64_marker", "identity", "unknown"}:
        bound["safe_to_send"] = "0"
    if bound["safe_to_send"] == "1" and (bound["transport"] == "none" or bound["field_role"] == "unknown" or bound["encoding"] == "unknown"):
        bound["safe_to_send"] = "0"
    return [TARGET_BOS, *[f"{key}={bound[key]}" for key in TARGET_KEYS], TARGET_EOS]


def audit_symbolic_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    base = audit_assembly_records(records)
    bad_refs: list[int] = []
    for index, row in enumerate(records):
        values = target_map(list(row.get("target_tokens") or []))
        if any(values.get(key) not in _REFS for key in ("transport_ref", "field_role_ref", "encoding_ref")):
            bad_refs.append(index)
    checks = dict(base.get("checks") or {})
    checks.pop("abstract_target_shape", None)
    checks["symbolic_target_shape"] = all(str(token).startswith(("[TARGET_", "question=", "next_action=", "repair_action=", "transport_ref=", "field_role_ref=", "encoding_ref=", "canary=", "oracle=", "stop_condition=", "safe_to_send=")) for row in records for token in row.get("target_tokens", []))
    checks["reference_values_bounded"] = not bad_refs
    checks["forbidden_fields_absent"] = not base.get("forbidden_indices")
    checks["observable_slots_complete"] = not base.get("incomplete_indices")
    checks["records_present"] = bool(records)
    return {"schema_version": f"{SCHEMA_VERSION}-audit", "checks": checks, "bad_reference_indices": bad_refs, "status": "passed" if all(checks.values()) else "failed"}


__all__ = [
    "SCHEMA_VERSION",
    "SYMBOLIC_TARGET_KEYS",
    "audit_symbolic_records",
    "bind_symbolic_plan",
    "sha256_json",
    "symbolic_record",
    "symbolic_target_for_context",
]
