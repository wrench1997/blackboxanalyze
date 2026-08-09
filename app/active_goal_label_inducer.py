"""Oracle-blind induction of generic goals and labels from active signatures.

This module answers a deliberately narrow research question: can a learner
invent a reusable *abstract effect label* from a bounded active-probe
signature, without being told a vulnerability family or an evaluator result?
The labels name only action slots (``p0`` ... ``p8``), never XSS/SQL/auth/etc.
An independent runner must still replay the frozen proposal and call a typed
oracle before any sample can be considered for training or memory.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from .active_probe_signature import PROBE_IDS, model_input_has_forbidden_field
from .probe_binding_attestation import (
    BINDING_SCHEMA_VERSION,
    CANONICAL_BINDING_SHA256,
    binding_attestation_valid,
)
from .pg105_observable_projection import SCHEMA_VERSION as CAUSAL_PROJECTION_SCHEMA


SCHEMA_VERSION = "active-auto-goal-label-inducer-v1"
_METHODS = frozenset({"GET", "POST"})
COMPOSITION_SCHEMA_VERSION = "generic-rule-composition-v1"
REQUIRED_COMPOSITION_ATOMS = (
    "effect_present",
    "probe_binding_valid",
    "get_post_repeat",
    "negative_control_clear",
)
_COMPOSITION_ATOM_RE = re.compile(
    r"^(?:effect_present|probe_binding_valid|get_post_repeat|negative_control_clear|candidate_without_surface_delta|supported_active_slot:p[0-8])$"
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def proposal_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def compose_rule_ir(atoms: Sequence[str]) -> dict[str, Any]:
    """Build a generic, non-executable Rule IR composition from reusable atoms.

    This is intentionally an annotation contract rather than an executable
    vulnerability rule.  Atoms are de-duplicated and sorted so copy/paste
    assembly and fragment order cannot create a spurious new concept.
    """

    normalized = sorted({str(atom) for atom in atoms})
    if not normalized:
        raise ValueError("composition requires at least one generic atom")
    if any(not _COMPOSITION_ATOM_RE.fullmatch(atom) for atom in normalized):
        raise ValueError("composition contains an unsupported or family-specific atom")
    expression = {
        "op": "and",
        "args": [{"op": "atom", "name": atom} for atom in normalized],
    }
    digest = hashlib.sha256(_canonical(expression).encode("utf-8")).hexdigest()
    return {
        "schema_version": COMPOSITION_SCHEMA_VERSION,
        "executable": False,
        "typed_oracle_required": True,
        "atoms": normalized,
        "expression": expression,
        "canonical_sha256": digest,
    }


def _composition_projection(value: Mapping[str, Any], slots: Sequence[str], *, binding_valid: bool) -> dict[str, Any]:
    """Describe observed fragments without claiming the full goal is proven."""

    effect_slots = set(observed_effect_slots(value))
    observed: list[str] = []
    if effect_slots:
        observed.append("effect_present")
    if binding_valid:
        observed.append("probe_binding_valid")
    if len(slots) == 1:
        if slots[0] in effect_slots:
            observed.append(f"supported_active_slot:{slots[0]}")
        else:
            observed.append("candidate_without_surface_delta")
    elif any(slot not in effect_slots for slot in slots):
        observed.append("candidate_without_surface_delta")
    missing = [atom for atom in REQUIRED_COMPOSITION_ATOMS if atom not in observed]
    return {
        "schema_version": COMPOSITION_SCHEMA_VERSION,
        "executable": False,
        "status": "proposal_only",
        "required_atoms": list(REQUIRED_COMPOSITION_ATOMS),
        "observed_atoms": sorted(observed),
        "missing_atoms": missing,
        "rule_ir": compose_rule_ir(observed) if observed else None,
    }


def _signature(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if model_input_has_forbidden_field(value):
        raise ValueError("active goal/label inducer received an evaluator or raw field")
    if list(value.get("probe_order") or []) != list(PROBE_IDS):
        raise ValueError("active goal/label inducer requires the canonical probe bank")
    pattern = list(value.get("delta_pattern") or [])
    if len(pattern) != len(PROBE_IDS):
        raise ValueError("active goal/label inducer received an invalid delta pattern")
    method = str(value.get("method", ""))
    if method not in _METHODS:
        raise ValueError("active goal/label inducer received an invalid method")
    signs = list(value.get("geometry_sign_pattern") or [])
    if len(signs) != len(PROBE_IDS) or any(not isinstance(row, list) or len(row) != 11 for row in signs):
        raise ValueError("active goal/label inducer received an invalid geometry pattern")
    attention = value.get("attention_pattern")
    extension = value.get("causal_extension")
    if attention is not None:
        if not isinstance(attention, list) or len(attention) != len(PROBE_IDS) or any(not isinstance(item, bool) for item in attention):
            raise ValueError("active goal/label inducer received an invalid attention pattern")
    if extension is not None:
        if not isinstance(extension, Mapping) or str(extension.get("schema_version", "")) != CAUSAL_PROJECTION_SCHEMA:
            raise ValueError("active goal/label inducer received an invalid causal extension")
        for field in (
            "input_changed_pattern",
            "input_changed_response_unchanged_pattern",
            "relation_code_pattern",
            "input_shape_delta_pattern",
            "numeric_bucket_delta_pattern",
        ):
            pattern = extension.get(field)
            if not isinstance(pattern, list) or len(pattern) != len(PROBE_IDS):
                raise ValueError("active goal/label inducer received an incomplete causal extension")
        if any(int(code) not in range(4) for code in extension["relation_code_pattern"]):
            raise ValueError("active goal/label inducer received an invalid causal relation code")
    return value


def observed_effect_slots(value: Mapping[str, Any]) -> tuple[str, ...]:
    """Return slots with an actual bounded response/surface difference."""

    signature = _signature(value)
    return tuple(PROBE_IDS[index] for index, changed in enumerate(signature["delta_pattern"]) if bool(changed))


def active_slots(value: Mapping[str, Any]) -> tuple[str, ...]:
    """Return response-effect or generic input/response-anomaly slots."""

    signature = _signature(value)
    pattern = signature.get("attention_pattern")
    if not isinstance(pattern, list):
        pattern = signature["delta_pattern"]
    return tuple(PROBE_IDS[index] for index, changed in enumerate(pattern) if bool(changed))


class ActiveGoalLabelInducer:
    """Induce a small generic goal/label ontology from unlabeled signatures."""

    def __init__(
        self,
        *,
        minimum_support: int = 2,
        require_get_post: bool = True,
        require_binding_attestation: bool = False,
        expected_binding_sha256: str = CANONICAL_BINDING_SHA256,
    ) -> None:
        if int(minimum_support) < 1:
            raise ValueError("minimum_support must be positive")
        self.minimum_support = int(minimum_support)
        self.require_get_post = bool(require_get_post)
        self.require_binding_attestation = bool(require_binding_attestation)
        self.expected_binding_sha256 = str(expected_binding_sha256)
        self._proposal: dict[str, Any] | None = None

    def fit(self, rows: Sequence[Mapping[str, Any]]) -> "ActiveGoalLabelInducer":
        if not rows:
            raise ValueError("active goal/label inducer requires non-empty design rows")
        support: Counter[str] = Counter()
        methods: dict[str, set[str]] = defaultdict(set)
        pattern_counts: Counter[tuple[str, ...]] = Counter()
        ambiguous_count = 0
        for row in rows:
            value = row.get("model_input") if isinstance(row.get("model_input"), Mapping) else row.get("signature")
            if not isinstance(value, Mapping):
                raise ValueError("active goal/label inducer row is missing model_input")
            if self.require_binding_attestation and not binding_attestation_valid(value, expected_sha256=self.expected_binding_sha256):
                raise ValueError("active goal/label inducer training row lacks a valid probe binding attestation")
            slots = active_slots(value)
            pattern_counts[slots] += 1
            if len(slots) != 1:
                ambiguous_count += 1
                continue
            support[slots[0]] += 1
            methods[slots[0]].add(str(value["method"]))

        discovered: list[dict[str, Any]] = []
        for slot in PROBE_IDS:
            count = int(support[slot])
            method_set = sorted(methods[slot])
            stable = count >= self.minimum_support and (not self.require_get_post or method_set == ["GET", "POST"])
            if not stable:
                continue
            discovered.append({
                "slot": slot,
                "label_id": f"AUTO_EFFECT_SLOT_{slot.upper()}",
                "support_count": count,
                "support_rate": round(count / len(rows), 6),
                "method_coverage": method_set,
                "stable_under_get_post": method_set == ["GET", "POST"],
                "decision": "confirm_candidate",
            })
        supported_slots = [item["slot"] for item in discovered]
        labels = [
            {
                "label_id": "AUTO_NO_OBSERVED_EFFECT",
                "definition": {"predicate": "active_slot_count_equals", "value": 0},
                "decision": "reject",
            },
            *discovered,
            {
                "label_id": "AUTO_UNSEEN_OR_AMBIGUOUS_EFFECT",
                "definition": {"predicate": "active_slot_unseen_or_count_not_equal_one"},
                "decision": "abstain",
            },
        ]
        self._proposal = {
            "schema_version": SCHEMA_VERSION,
            "proposal_id": "pg103-active-auto-goal-label-v1",
            "proposal_inputs": {
                "design_row_count": len(rows),
                "oracle_visible": False,
                "family_visible": False,
                "raw_probe_visible": False,
                "raw_response_visible": False,
                "induction_objective": "stable_single_active_slot_with_get_post_support",
                "binding_attestation_required": self.require_binding_attestation,
            },
            "goal": {
                "goal_id": "auto_repeatable_effect_slot_discovery_v1",
                "intent": "discover a repeatable bounded effect without assigning a vulnerability family",
                "success_condition": [
                    "one supported active slot is observed",
                    "the same generic label repeats on GET and POST or a second compatible probe",
                    "matched negative controls do not receive a candidate decision",
                    "fresh reset, loopback safety and evidence hash checks remain true",
                ],
                "failure_condition": [
                    "no observable active slot",
                    "the effect is not repeated",
                    "a matched negative control also receives the candidate decision",
                ],
                "abstain_condition": [
                    "more than one active slot is observed",
                    "the active slot was not supported during design",
                    "GET/POST support is incomplete or conflicting",
                ],
                "budget": {"max_steps": 2, "requires_fresh_reset": True, "requires_get_post_pair": True},
            },
            "labels": labels,
            "discovered_effect_slots": discovered,
            "supported_slots": supported_slots,
            "probe_binding": {
                "required": self.require_binding_attestation,
                "schema_version": BINDING_SCHEMA_VERSION,
                "binding_sha256": self.expected_binding_sha256,
            },
            "composition_contract": {
                "required_atoms": list(REQUIRED_COMPOSITION_ATOMS),
                "rule_ir": compose_rule_ir(REQUIRED_COMPOSITION_ATOMS),
                "executable": False,
                "typed_oracle_required": True,
                "family_classification_forbidden": True,
            },
            "induction_stats": {
                "pattern_count": len(pattern_counts),
                "single_slot_pattern_count": sum(int(len(pattern) == 1) for pattern in pattern_counts),
                "ambiguous_row_count": ambiguous_count,
                "minimum_support": self.minimum_support,
                "require_get_post": self.require_get_post,
                "pattern_support": {"|".join(pattern) if pattern else "NONE": count for pattern, count in sorted(pattern_counts.items())},
            },
            "audit": {
                "labels_are_generic_action_effect_aliases": True,
                "vulnerability_family_names_not_generated": True,
                "requires_independent_typed_oracle": True,
                "training_promotion_allowed": False,
                "memory_promotion_allowed": False,
            },
        }
        self._proposal["proposal_sha256"] = proposal_digest(self._proposal)
        return self

    def proposal(self) -> dict[str, Any]:
        if self._proposal is None:
            raise RuntimeError("active goal/label inducer is not fitted")
        return json.loads(json.dumps(self._proposal, ensure_ascii=False))

    def predict(self, value: Mapping[str, Any], *, guarded: bool = True) -> dict[str, Any]:
        proposal = self.proposal()
        slots = active_slots(value)
        binding_valid = binding_attestation_valid(value, expected_sha256=self.expected_binding_sha256)
        composition = _composition_projection(value, slots, binding_valid=binding_valid)

        def emit(**fields: Any) -> dict[str, Any]:
            result = dict(fields)
            result["composition"] = composition
            result["composition_atoms"] = list(composition["observed_atoms"])
            if result.get("decision") == "confirm_candidate":
                # A single active signature is only a reusable effect fragment;
                # repeatability, a matched negative and a typed oracle are still
                # required before anything can be promoted.
                result["composition_decision"] = "await_typed_oracle"
            else:
                result["composition_decision"] = result.get("decision")
            result["promotion_eligible"] = False
            return result

        if self.require_binding_attestation and not binding_valid:
            if guarded:
                return emit(
                    label_id="AUTO_UNATTESTED_PROBE_BANK",
                    decision="abstain",
                    active_slots=list(slots),
                    guarded=True,
                    reason="invalid_probe_binding_attestation",
                )
        if not guarded:
            if len(slots) == 1:
                return emit(
                    label_id=f"AUTO_EFFECT_SLOT_{slots[0].upper()}",
                    decision="confirm_candidate",
                    active_slots=list(slots),
                    guarded=False,
                )
            if not slots:
                return emit(
                    label_id="AUTO_NO_OBSERVED_EFFECT",
                    decision="reject",
                    active_slots=[],
                    guarded=False,
                )
            return emit(
                label_id="AUTO_UNSEEN_OR_AMBIGUOUS_EFFECT",
                decision="confirm_candidate",
                active_slots=list(slots),
                guarded=False,
                reason="raw_multi_or_ambiguous_effect",
            )
        supported = set(str(slot) for slot in proposal.get("supported_slots", []))
        if not slots:
            return emit(
                label_id="AUTO_NO_OBSERVED_EFFECT",
                decision="reject",
                active_slots=[],
                guarded=True,
                reason="no_observable_active_effect",
            )
        effect_slots = set(observed_effect_slots(value))
        if not effect_slots:
            # An input-only change is a suspicious relation, never a known
            # effect.  This prevents a decoy parameter change from borrowing
            # a supported slot label.
            return emit(
                label_id="AUTO_UNSEEN_OR_AMBIGUOUS_EFFECT",
                decision="abstain",
                active_slots=list(slots),
                guarded=True,
                reason="input_changed_without_surface_effect",
            )
        if len(slots) != 1:
            return emit(
                label_id="AUTO_UNSEEN_OR_AMBIGUOUS_EFFECT",
                decision="abstain",
                active_slots=list(slots),
                guarded=True,
                reason="ambiguous_active_effect",
            )
        if slots[0] not in supported:
            return emit(
                label_id="AUTO_UNSEEN_OR_AMBIGUOUS_EFFECT",
                decision="abstain",
                active_slots=list(slots),
                guarded=True,
                reason="unseen_active_slot",
            )
        return emit(
            label_id=f"AUTO_EFFECT_SLOT_{slots[0].upper()}",
            decision="confirm_candidate",
            active_slots=list(slots),
            guarded=True,
            reason="supported_repeatable_effect",
        )


__all__ = [
    "ActiveGoalLabelInducer",
    "REQUIRED_COMPOSITION_ATOMS",
    "COMPOSITION_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "active_slots",
    "compose_rule_ir",
    "observed_effect_slots",
    "proposal_digest",
]
