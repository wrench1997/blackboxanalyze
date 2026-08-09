"""Family-free, fail-closed scheduling from bounded probe failures.

The scheduler deliberately consumes projections rather than raw requests,
responses, payloads, credentials, or target identifiers.  A failure is an
observation about which *gate* is still unresolved; it is not a vulnerability
label.  The returned action names are abstract, allow-listed replay actions
that a local collector may implement with safe probes.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "sift-failure-signature-v1"
_FAILURE_KINDS = frozenset(
    {
        "typed_positive",
        "candidate_without_typed_effect",
        "oracle_unavailable",
        "method_disagreement",
        "no_surface_delta",
        "budget_exhausted",
    }
)
_NEXT_ACTIONS = frozenset(
    {
        "replay_other_method",
        "repeat_matched_negative_pair",
        "probe_candidate_other_method",
        "abstain_candidate_only",
        "abstain_unknown_oracle",
        "stop_confirmed_positive",
        "abstain_budget_exhausted",
    }
)
_KEY_FEATURES = ("failure_kind", "failed_gate", "candidate_signal", "observed_method", "methods_seen", "probe_budget")
_NON_FAILURE_KINDS = frozenset({"typed_positive"})
_HISTORY_ACTIONS = frozenset(
    {
        *_NEXT_ACTIONS,
        "assemble_abstract_plan",
        "repair_abstract_plan",
        "request_observation",
        "probe_candidate_same_method",
        "candidate_request",
        "reference_request",
        "negative_control",
        "candidate_failed",
        "inspect_binding",
        "inspect_environment",
        "recheck_oracle",
        "repair_candidate",
        "replay_confirmed",
        "abstain",
    }
)


def _key_feature_weights(kind: str, gate: str, *, candidate: bool, method_count: int, remaining_probe_budget: int) -> dict[str, float]:
    """Assign bounded, auditable emphasis to the features that disambiguate the next probe."""

    weights = {feature: 1.0 for feature in _KEY_FEATURES}
    if gate == "typed_effect":
        weights["failed_gate"] = 2.6
        weights["candidate_signal"] = 2.2 if candidate else 1.2
        weights["methods_seen"] = 1.8
    elif gate == "matched_negative_control":
        weights["failed_gate"] = 2.5
        weights["observed_method"] = 1.8
        weights["methods_seen"] = 2.2
    elif gate == "cross_channel_consistency":
        weights["failed_gate"] = 2.4
        weights["methods_seen"] = 2.5
    elif gate == "surface_delta":
        weights["failed_gate"] = 2.0
        weights["candidate_signal"] = 1.8
    if remaining_probe_budget > 0:
        weights["probe_budget"] = min(3.0, 1.0 + 0.5 * remaining_probe_budget)
    total = sum(weights.values())
    return {key: round(value / total, 6) for key, value in weights.items()}


def key_feature_weights_for_signature(signature: Mapping[str, Any]) -> dict[str, float]:
    """Return the normalized feature focus used to assemble a replay step.

    Older traces may not contain the optional weight projection.  Recomputing
    it from the bounded signature keeps those traces usable without treating
    a missing weight field as an unstructured feature or retaining raw data.
    """

    supplied = signature.get("key_feature_weights")
    if isinstance(supplied, Mapping) and set(supplied) == set(_KEY_FEATURES):
        values = {key: float(supplied[key]) for key in _KEY_FEATURES}
        if all(0.0 <= value <= 1.0 for value in values.values()) and abs(sum(values.values()) - 1.0) <= 1e-5:
            return values
    methods = {
        str(item).upper()
        for item in signature.get("methods_seen", [])
        if str(item).upper() in {"GET", "POST"}
    }
    return _key_feature_weights(
        str(signature.get("kind", "")),
        str(signature.get("failed_gate", "")),
        candidate=bool(signature.get("candidate_signal")),
        method_count=len(methods),
        remaining_probe_budget=max(0, int(signature.get("remaining_probe_budget", 0) or 0)),
    )


def validate_failure_transition(previous_action: str, signature: Mapping[str, Any]) -> dict[str, Any]:
    """Require a changed bounded action after an observed failure.

    This is intentionally a separate contract from the failure classifier.  A
    model can emit a plausible ``repair_action`` token while repeating the
    exact action that just failed; that is not active diagnosis.  New traces
    may include ``previous_action`` and use this gate, while legacy traces
    without that field remain readable for historical comparison.
    """

    previous = str(previous_action or "")
    next_action = str(signature.get("next_action", ""))
    kind = str(signature.get("kind", ""))
    if next_action not in _HISTORY_ACTIONS:
        raise ValueError("failure transition next_action is not allow-listed")
    if not previous:
        return {
            "previous_action": "",
            "next_action": next_action,
            "action_changed": None,
            "repair_transition_required": False,
            "repair_transition_valid": True,
        }
    if previous not in _HISTORY_ACTIONS:
        raise ValueError("failure transition previous_action is not allow-listed")
    changed = previous != next_action
    required = kind not in _NON_FAILURE_KINDS
    if required and not changed:
        raise ValueError("failure transition must change the next action")
    return {
        "previous_action": previous,
        "next_action": next_action,
        "action_changed": changed,
        "repair_transition_required": required,
        "repair_transition_valid": True,
    }


def _method_set(records: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(record.get("method", "")).upper() for record in records if record.get("method")}


def failure_signature(
    record: Mapping[str, Any],
    *,
    prior_records: Sequence[Mapping[str, Any]] = (),
    max_steps: int = 4,
    step_count: int = 1,
) -> dict[str, Any]:
    """Summarize the current unresolved gate and select one bounded next step.

    The result contains no raw probe/body data.  ``prior_records`` is used only
    to detect GET/POST disagreement and to ensure the scheduler progresses
    instead of replaying the same method forever.
    """

    methods = _method_set([*prior_records, record])
    method = str(record.get("method", "")).upper()
    candidate = bool(record.get("candidate_signal"))
    authority = bool(record.get("positive_authority"))
    typed_available = bool(record.get("typed_available", record.get("oracle_available", False)))
    positive = bool(record.get("positive"))
    role = str(record.get("role", ""))
    probe_round = max(1, int(record.get("probe_round", 1) or 1))
    max_probe_rounds = max(probe_round, int(record.get("max_probe_rounds", 1) or 1))
    remaining_probe_budget = max(0, max_probe_rounds - probe_round)
    prior_candidates = {
        str(item.get("method", "")).upper(): bool(item.get("candidate_signal"))
        for item in prior_records
        if item.get("role") == "candidate"
    }
    disagreement = method in prior_candidates and prior_candidates[method] != candidate

    if role == "control" and not candidate:
        kind = "no_surface_delta"
        gate = "matched_negative_control"
        next_action = "repeat_matched_negative_pair"
    elif authority and positive:
        kind = "typed_positive"
        gate = "typed_effect"
        # Keep the action name canonical across implementations.  The first
        # typed-positive observation still needs the candidate replay on the
        # other channel; ``replay_other_method`` is reserved for an unknown
        # oracle where no typed effect is available yet.
        next_action = "probe_candidate_other_method" if len(methods) < 2 else "stop_confirmed_positive"
    elif not typed_available:
        kind = "oracle_unavailable"
        gate = "typed_effect"
        next_action = "replay_other_method" if len(methods) < 2 or remaining_probe_budget > 0 else "abstain_unknown_oracle"
    elif disagreement:
        kind = "method_disagreement"
        gate = "cross_channel_consistency"
        next_action = "repeat_matched_negative_pair" if len(methods) < 2 else "abstain_candidate_only"
    elif candidate and not positive:
        kind = "candidate_without_typed_effect"
        gate = "typed_effect"
        next_action = "probe_candidate_other_method" if len(methods) < 2 or remaining_probe_budget > 0 else "abstain_candidate_only"
    else:
        kind = "no_surface_delta"
        gate = "surface_delta"
        next_action = "probe_candidate_other_method" if len(methods) < 2 or remaining_probe_budget > 0 else "abstain_candidate_only"

    # Preserve the causal failure category at the budget boundary.  The
    # budget is a modifier of the next action, not a replacement for the
    # evidence that caused the branch.
    if step_count >= max_steps and next_action not in {"stop_confirmed_positive", "abstain_candidate_only", "abstain_unknown_oracle"}:
        next_action = "abstain_budget_exhausted"

    if kind not in _FAILURE_KINDS or next_action not in _NEXT_ACTIONS:  # pragma: no cover
        raise AssertionError("failure scheduler emitted a non-allow-listed value")
    key_feature_weights = _key_feature_weights(kind, gate, candidate=candidate, method_count=len(methods), remaining_probe_budget=remaining_probe_budget)
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "failed_gate": gate,
        "observed_method": method,
        "methods_seen": sorted(methods),
        "candidate_signal": candidate,
        "typed_available": typed_available,
        "positive_authority": authority,
        "probe_round": probe_round,
        "remaining_probe_budget": remaining_probe_budget,
        "key_feature_weights": key_feature_weights,
        "key_features_ranked": [key for key, _ in sorted(key_feature_weights.items(), key=lambda item: (-item[1], item[0]))],
        "next_action": next_action,
        "model_visible": True,
        "raw_probe_retained": False,
        "raw_response_retained": False,
        "memory_promotion_allowed": False,
    }
    if "previous_action" in record:
        result.update(validate_failure_transition(str(record.get("previous_action", "")), result))
    return result


def validate_failure_signature(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the small projection stored in a trace step."""

    if str(value.get("schema_version", "")) != SCHEMA_VERSION:
        raise ValueError("failure_signature schema version is invalid")
    kind = str(value.get("kind", ""))
    action = str(value.get("next_action", ""))
    if kind not in _FAILURE_KINDS:
        raise ValueError("failure_signature kind is not allow-listed")
    if action not in _NEXT_ACTIONS:
        raise ValueError("failure_signature next_action is not allow-listed")
    probe_round = int(value.get("probe_round", 1))
    remaining = int(value.get("remaining_probe_budget", 0))
    if probe_round < 1 or probe_round > 16 or remaining < 0 or remaining > 16:
        raise ValueError("failure_signature probe budget is invalid")
    weights = value.get("key_feature_weights")
    if weights is not None:
        if not isinstance(weights, dict) or set(weights) != set(_KEY_FEATURES):
            raise ValueError("failure_signature key feature weights are invalid")
        if any(not 0.0 <= float(item) <= 1.0 for item in weights.values()) or abs(sum(float(item) for item in weights.values()) - 1.0) > 1e-5:
            raise ValueError("failure_signature key feature weights must be normalized")
        ranked = value.get("key_features_ranked")
        expected_ranked = [key for key, _ in sorted(((key, float(weights[key])) for key in _KEY_FEATURES), key=lambda item: (-item[1], item[0]))]
        if ranked is not None and ranked != expected_ranked:
            raise ValueError("failure_signature key feature ranking is inconsistent")
    methods = value.get("methods_seen")
    if not isinstance(methods, list) or any(str(item).upper() not in {"GET", "POST"} for item in methods):
        raise ValueError("failure_signature methods_seen is invalid")
    if bool(value.get("memory_promotion_allowed")) or bool(value.get("raw_probe_retained")) or bool(value.get("raw_response_retained")):
        raise ValueError("failure_signature retention contract failed")
    if "previous_action" in value:
        transition = validate_failure_transition(str(value.get("previous_action", "")), value)
        if value.get("action_changed") != transition["action_changed"] or value.get("repair_transition_valid") is not True:
            raise ValueError("failure transition metadata is inconsistent")
    return dict(value)


__all__ = ["SCHEMA_VERSION", "failure_signature", "key_feature_weights_for_signature", "validate_failure_signature", "validate_failure_transition"]
