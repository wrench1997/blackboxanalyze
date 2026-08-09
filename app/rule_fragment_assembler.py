"""Family-free Rule IR fragment validation and copy/paste assembly.

This module deliberately treats a rule as a composition of small, bounded
evidence fragments.  A fragment contains only probe-bank geometry, transport
method, a fresh-reset commitment and a response/effect relation.  It never
stores a vulnerability family, an oracle result, a request value or a body.

Assembly is a proposal: it is non-executable, requires an independent typed
oracle, and can never promote training data or long-term memory by itself.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from .active_probe_signature import PROBE_IDS, model_input_has_forbidden_field
from .probe_binding_attestation import CANONICAL_BINDING_SHA256, binding_attestation_valid


FRAGMENT_SCHEMA_VERSION = "generic-rule-fragment-v1"
ASSEMBLY_SCHEMA_VERSION = "generic-rule-assembly-v1"
_METHODS = frozenset({"GET", "POST"})
_ATOMS = frozenset({
    "effect_present",
    "input_only_anomaly",
    "no_effect",
    "ambiguous_effect",
    "probe_binding_valid",
    "negative_control_clear",
})
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _hash_is_valid(value: Any) -> bool:
    return bool(_HASH_RE.fullmatch(str(value or "")))


def _effect_slots(model_input: Mapping[str, Any]) -> tuple[str, ...]:
    pattern = list(model_input.get("delta_pattern") or [])
    if len(pattern) != len(PROBE_IDS):
        raise ValueError("fragment model input has an invalid response delta pattern")
    return tuple(PROBE_IDS[index] for index, changed in enumerate(pattern) if bool(changed))


def _input_only_anomaly(model_input: Mapping[str, Any]) -> bool:
    extension = model_input.get("causal_extension")
    if not isinstance(extension, Mapping):
        return False
    pattern = extension.get("input_changed_response_unchanged_pattern")
    return isinstance(pattern, list) and len(pattern) == len(PROBE_IDS) and any(bool(item) for item in pattern)


def _fresh_reset_valid(reset: Mapping[str, Any]) -> bool:
    return (
        bool(reset.get("fresh_target"))
        and bool(reset.get("completed"))
        and _hash_is_valid(reset.get("reset_sha256"))
    )


def fragment_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one bounded catalog row to a generic reusable fragment."""

    model_input = row.get("model_input")
    if not isinstance(model_input, Mapping):
        raise ValueError("fragment row is missing model_input")
    if model_input_has_forbidden_field(model_input):
        raise ValueError("fragment model input contains an evaluator or raw field")
    method = str(model_input.get("method", row.get("method", ""))).upper()
    if method not in _METHODS:
        raise ValueError("fragment method is outside the GET/POST allow-list")
    evidence_hash = str(row.get("evidence_sha256", ""))
    if not _hash_is_valid(evidence_hash):
        raise ValueError("fragment requires a 256-bit evidence hash")
    reset = row.get("fresh_reset")
    if not isinstance(reset, Mapping) or not _fresh_reset_valid(reset):
        raise ValueError("fragment requires a completed fresh reset commitment")
    if not bool(row.get("negative_control_matched")):
        raise ValueError("fragment requires a matched negative control")
    binding_valid = binding_attestation_valid(model_input, expected_sha256=CANONICAL_BINDING_SHA256)
    effect_slots = _effect_slots(model_input)
    if len(effect_slots) == 1:
        relation_atom = "effect_present"
        slot: str | None = effect_slots[0]
    elif len(effect_slots) > 1:
        relation_atom = "ambiguous_effect"
        slot = None
    elif _input_only_anomaly(model_input):
        relation_atom = "input_only_anomaly"
        slot = None
    else:
        relation_atom = "no_effect"
        slot = None
    atoms = [relation_atom]
    if binding_valid:
        atoms.append("probe_binding_valid")
    if bool(row.get("negative_control_matched")):
        atoms.append("negative_control_clear")
    base = {
        "schema_version": FRAGMENT_SCHEMA_VERSION,
        "method": method,
        "relation_atom": relation_atom,
        "slot": slot,
        "atoms": sorted(set(atoms)),
        "binding_sha256": str((model_input.get("probe_binding") or {}).get("binding_sha256", "")),
        "evidence_sha256": evidence_hash,
        "reset_sha256": str(reset.get("reset_sha256")),
        "fresh_target": True,
        "negative_control_clear": True,
    }
    fragment = dict(base)
    fragment["fragment_id"] = _sha256(base)
    return fragment


def _fragment_error(fragment: Mapping[str, Any], *, expected_binding_sha256: str) -> str | None:
    if str(fragment.get("schema_version", "")) != FRAGMENT_SCHEMA_VERSION:
        return "unknown_fragment_schema"
    if str(fragment.get("method", "")).upper() not in _METHODS:
        return "invalid_fragment_method"
    if str(fragment.get("relation_atom", "")) not in _ATOMS:
        return "unsupported_fragment_atom"
    slot = fragment.get("slot")
    if slot is not None and str(slot) not in PROBE_IDS:
        return "invalid_fragment_slot"
    atoms = {str(atom) for atom in fragment.get("atoms", [])}
    if not atoms or not atoms.issubset(_ATOMS) or str(fragment.get("relation_atom")) not in atoms:
        return "invalid_fragment_atoms"
    if not _hash_is_valid(fragment.get("evidence_sha256")) or not _hash_is_valid(fragment.get("reset_sha256")):
        return "invalid_fragment_evidence_hash"
    if not bool(fragment.get("fresh_target")) or not bool(fragment.get("negative_control_clear")):
        return "unsafe_fragment_context"
    if str(fragment.get("binding_sha256", "")) != str(expected_binding_sha256) or "probe_binding_valid" not in atoms:
        return "invalid_fragment_binding"
    expected_id = dict(fragment)
    expected_id.pop("fragment_id", None)
    if str(fragment.get("fragment_id", "")) != _sha256(expected_id):
        return "fragment_id_mismatch"
    return None


def _assembly_expression(atoms: Sequence[str]) -> dict[str, Any]:
    normalized = sorted({str(atom) for atom in atoms})
    return {
        "op": "and",
        "args": [{"op": "atom", "name": atom} for atom in normalized],
    }


def assemble_rule_fragments(
    fragments: Sequence[Mapping[str, Any]],
    *,
    supported_slots: Sequence[str] = (),
    expected_binding_sha256: str = CANONICAL_BINDING_SHA256,
) -> dict[str, Any]:
    """Assemble bounded fragments into a non-executable Rule IR proposal.

    The output is canonicalized by fragment id, so copying the same fragments
    in another order cannot create a new rule.  Any evidence/binding/slot
    conflict returns an abstention rather than a partial candidate.
    """

    items = [dict(fragment) for fragment in fragments]
    common = {
        "schema_version": ASSEMBLY_SCHEMA_VERSION,
        "executable": False,
        "typed_oracle_required": True,
        "promotion_eligible": False,
    }
    if len(items) < 2:
        return {**common, "decision": "abstain", "reason": "insufficient_fragments"}
    errors = [_fragment_error(item, expected_binding_sha256=expected_binding_sha256) for item in items]
    if any(error is not None for error in errors):
        return {**common, "decision": "abstain", "reason": next(error for error in errors if error is not None)}
    evidence = [str(item["evidence_sha256"]) for item in items]
    if len(set(evidence)) != len(evidence):
        return {**common, "decision": "abstain", "reason": "duplicate_evidence"}
    methods = {str(item["method"]).upper() for item in items}
    if methods != set(_METHODS):
        return {**common, "decision": "abstain", "reason": "get_post_pair_required"}
    if len(items) != 2:
        return {**common, "decision": "abstain", "reason": "exactly_two_transport_fragments_required"}
    if items[0]["method"] == items[1]["method"]:
        return {**common, "decision": "abstain", "reason": "duplicate_transport_method"}
    relations = {str(item["relation_atom"]) for item in items}
    if relations != {"effect_present"}:
        if "input_only_anomaly" in relations:
            reason = "input_only_fragment_cannot_supply_effect"
        elif "ambiguous_effect" in relations:
            reason = "ambiguous_effect_fragment"
        else:
            reason = "effect_pair_required"
        return {**common, "decision": "abstain", "reason": reason}
    slots = {str(item.get("slot")) for item in items}
    if len(slots) != 1 or "None" in slots:
        return {**common, "decision": "abstain", "reason": "slot_conflict"}
    slot = next(iter(slots))
    if supported_slots and slot not in {str(value) for value in supported_slots}:
        return {**common, "decision": "abstain", "reason": "unseen_effect_slot"}
    bindings = {str(item["binding_sha256"]) for item in items}
    if len(bindings) != 1:
        return {**common, "decision": "abstain", "reason": "binding_conflict"}
    atoms = ["effect_present", "probe_binding_valid", "get_post_repeat", "negative_control_clear"]
    expression = _assembly_expression(atoms)
    ordered_ids = sorted(str(item["fragment_id"]) for item in items)
    assembly = {
        **common,
        "decision": "await_typed_oracle",
        "reason": "two_method_effect_pair_reassembled",
        "slot": slot,
        "methods": ["GET", "POST"],
        "atoms": atoms,
        "expression": expression,
        "fragment_ids": ordered_ids,
        "evidence_sha256": sorted(evidence),
        "canonical_sha256": _sha256({"expression": expression, "fragment_ids": ordered_ids, "slot": slot}),
    }
    return assembly


__all__ = [
    "ASSEMBLY_SCHEMA_VERSION",
    "FRAGMENT_SCHEMA_VERSION",
    "assemble_rule_fragments",
    "fragment_from_row",
]
