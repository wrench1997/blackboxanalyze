"""Bounded automatic goal/label induction for the PG-96 experiment.

This module is deliberately *not* a vulnerability oracle and it is not an
LLM.  It is a small, deterministic inductive-synthesis baseline that answers
one research question: can a learner invent a useful, vocabulary-invariant
intermediate label from model-visible observation differences without being
given a family name or the evaluator result?

The same contract can later be used to validate a JSON proposal produced by a
language model.  Keeping the proposal grammar and the evaluator separate is
important: a proposal may suggest a goal or a label, but only an independent
typed oracle can decide whether a target effect was actually present.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "auto-goal-label-proposal-v1"
ALLOWED_PHASES = {"screen", "confirm", "error", "timeout"}
_SAFE_METHODS = {"GET", "POST"}
_NUMERIC_SURFACE_FIELDS = (
    "array_field_count",
    "boolean_field_count",
    "nonzero_numeric_count",
    "numeric_field_count",
    "true_boolean_count",
)
_NUMERIC_GEOMETRY_FIELDS = (
    "array_count",
    "array_item_count",
    "boolean_count",
    "leaf_count",
    "max_depth",
    "nonzero_numeric_count",
    "numeric_count",
    "object_count",
    "string_count",
    "string_length_bucket_sum",
    "true_boolean_count",
)
_FORBIDDEN_INPUT_KEYS = {
    "family",
    "hypothesis",
    "surface",
    "oracle_projection",
    "decision",
    "belief_before",
    "belief_after",
    "next_action",
    "target_instance_id",
    "route_template_id",
    "probe_ref",
    "probe_sha256",
    "body_sha256",
    "semantic_body_sha256",
    "projection_sha256",
    "observation_sha256",
    "geometry_sha256",
}


def _bounded_id(value: Any, label: str) -> str:
    text = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", text):
        raise ValueError(f"{label} is not a bounded identifier")
    return text


def _phase_from_probe_ref(value: Any) -> str:
    phase = str(value).rsplit("-", 1)[-1]
    if phase not in ALLOWED_PHASES:
        raise ValueError("probe phase is not allow-listed")
    return phase


def _sign_delta(before: Any, after: Any) -> str:
    if before == after:
        return "same"
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        if after > before:
            return "increase"
        if after < before:
            return "decrease"
    if isinstance(before, bool) and isinstance(after, bool):
        return "true" if after else "false"
    return "change"


def _feature_token(prefix: str, field: str, relation: str) -> str:
    return f"DELTA_{prefix}_{field}_{relation}"


def _projection_delta(control: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[str, ...]:
    """Return bounded, evaluator-free difference tokens for a matched pair."""

    tokens: list[str] = []
    for field in ("status_class", "content_type_class", "body_length_bucket"):
        relation = _sign_delta(control.get(field), candidate.get(field))
        if relation != "same":
            tokens.append(_feature_token("RESPONSE", field.upper(), relation.upper()))

    for group_name, fields, prefix in (
        ("effect_surface", _NUMERIC_SURFACE_FIELDS, "SURFACE"),
        ("effect_geometry", _NUMERIC_GEOMETRY_FIELDS, "GEOMETRY"),
    ):
        left = control.get(group_name) or {}
        right = candidate.get(group_name) or {}
        for field in fields:
            relation = _sign_delta(left.get(field), right.get(field))
            if relation != "same":
                tokens.append(_feature_token(prefix, field.upper(), relation.upper()))

    for field in ("location_origin_changed", "state_changed", "transport_error"):
        relation = _sign_delta(control.get(field), candidate.get(field))
        if relation != "same":
            tokens.append(_feature_token("RESPONSE", field.upper(), relation.upper()))
    return tuple(sorted(set(tokens)))


def make_visible_pair(control_step: Mapping[str, Any], candidate_step: Mapping[str, Any]) -> dict[str, Any]:
    """Construct the model-visible portion of one control/candidate pair.

    Pairing uses the runner's private step ids, but this returned object does
    not expose those ids, route names, family names, oracle fields, or hashes.
    """

    action = candidate_step.get("action_manifest") or {}
    safety = action.get("safety") or {}
    method = str(action.get("method", ""))
    if method not in _SAFE_METHODS:
        raise ValueError("only GET/POST are valid for the automatic proposal")
    encoding = tuple(str(x) for x in (action.get("encoding_chain") or []))
    if not encoding or len(encoding) > 4:
        raise ValueError("encoding chain is empty or unbounded")
    phase = _phase_from_probe_ref(action.get("probe_ref", ""))
    delta_tokens = _projection_delta(
        control_step.get("response_projection") or {},
        candidate_step.get("response_projection") or {},
    )
    return {
        "schema_version": "auto-goal-visible-pair-v1",
        "method": method,
        "encoding_class": "->".join(encoding),
        "phase": phase,
        "safe_probe": bool(safety.get("no_external_network"))
        and bool(safety.get("does_not_execute"))
        and bool(safety.get("no_database_write"))
        and bool(safety.get("no_credential_access")),
        "delta_tokens": list(delta_tokens),
        "delta_count": len(delta_tokens),
        "has_observed_change": bool(delta_tokens),
    }


def _prefix(token: str) -> str:
    parts = token.split("_")
    return parts[1] if len(parts) > 1 else "UNKNOWN"


def _entropy(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -(probability * math.log2(probability) + (1.0 - probability) * math.log2(1.0 - probability))


def _candidate_predicates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Synthesize generic predicates without looking at typed labels.

    A predicate is a bounded OR over a feature family (SURFACE, GEOMETRY,
    RESPONSE, ...).  The score rewards balanced coverage and independent
    context support, rather than oracle correlation.
    """

    contexts: dict[str, set[str]] = defaultdict(set)
    support: Counter[str] = Counter()
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        row_id = _bounded_id(row.get("row_id", "row"), "row_id")
        context = _bounded_id(row.get("context_group", "context"), "context_group")
        tokens = set(str(x) for x in row.get("delta_tokens", []))
        families = {_prefix(token) for token in tokens if token.startswith("DELTA_")}
        for family in families:
            support[family] += 1
            contexts[family].add(context)
            groups[family].add(row_id)

    total = max(1, len(rows))
    predicates: list[dict[str, Any]] = []
    for family, count in sorted(support.items()):
        p = count / total
        independent = len(contexts[family])
        independent_ratio = independent / max(1, len({str(r.get("context_group", "")) for r in rows}))
        score = _entropy(p) * independent_ratio / (1.0 + len(family) / 16.0)
        predicates.append(
            {
                "predicate_id": f"any_delta_{family.lower()}",
                "definition": {"any_delta_prefix": family},
                "support_count": count,
                "support_rate": round(p, 6),
                "independent_context_count": independent,
                "independent_context_rate": round(independent_ratio, 6),
                "unsupervised_score": round(score, 6),
            }
        )
    predicates.sort(key=lambda item: (-item["unsupervised_score"], item["predicate_id"]))
    return predicates


def propose_goal_and_labels(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Create a candidate goal and label ontology from visible rows only."""

    if not rows:
        raise ValueError("cannot propose a goal from an empty design set")
    for row in rows:
        leaked = _FORBIDDEN_INPUT_KEYS.intersection(row)
        if leaked:
            raise ValueError(f"evaluator/identity fields leaked into proposal: {sorted(leaked)}")
    predicates = _candidate_predicates(rows)
    if not predicates:
        selected = {
            "predicate_id": "any_delta_none",
            "definition": {"any_delta_prefix": "NONE"},
            "support_count": 0,
            "support_rate": 0.0,
            "independent_context_count": 0,
            "independent_context_rate": 0.0,
            "unsupervised_score": 0.0,
        }
    else:
        selected = predicates[0]
    prefix = str(selected["definition"]["any_delta_prefix"])
    return {
        "schema_version": SCHEMA_VERSION,
        "proposal_id": "pg96-auto-goal-label-v1",
        "proposal_inputs": {
            "row_count": len(rows),
            "oracle_visible": False,
            "family_visible": False,
            "raw_probe_visible": False,
            "raw_response_visible": False,
            "selection_objective": "balanced_support_x_independent_context_stability_x_simplicity",
        },
        "goal": {
            "goal_id": "auto_goal_stable_effect_discovery_v1",
            "intent": "discover a repeatable bounded observation change with safe probes",
            "success_predicate": selected["predicate_id"],
            "success_condition": [
                "candidate-vs-matched-control change is observed",
                "the change repeats on a second compatible probe",
                "no matched negative control produces the same success decision",
                "fresh reset and loopback safety remain true",
            ],
            "failure_condition": [
                "no observation change",
                "change is unsupported or unstable across contexts",
                "negative control has a matching success signal",
            ],
            "abstain_condition": [
                "feature family unseen during proposal design",
                "ambiguous or conflicting repeated observations",
            ],
            "budget": {"max_steps": 2, "max_candidate_probes": 2, "requires_fresh_reset": True},
        },
        "labels": [
            {
                "label_id": "AUTO_LABEL_NO_OBSERVED_CHANGE",
                "definition": {"predicate": "not_selected_prefix", "prefix": prefix},
                "decision": "reject",
            },
            {
                "label_id": "AUTO_LABEL_STABLE_EFFECT_CHANGE",
                "definition": {"predicate": "selected_prefix", "prefix": prefix},
                "decision": "confirm_candidate",
            },
            {
                "label_id": "AUTO_LABEL_UNSUPPORTED_OR_AMBIGUOUS",
                "definition": {"predicate": "unseen_prefix_or_conflict"},
                "decision": "abstain",
            },
        ],
        "candidate_predicates": predicates,
        "selected_predicate": selected,
        "audit": {
            "label_names_are_semantic_aliases_not_oracle_labels": True,
            "requires_independent_typed_oracle": True,
            "training_promotion_allowed": False,
            "memory_promotion_allowed": False,
        },
    }


def apply_proposal(row: Mapping[str, Any], proposal: Mapping[str, Any]) -> dict[str, Any]:
    """Decode a visible row using a proposal, with fail-closed abstention."""

    selected = proposal.get("selected_predicate") or {}
    prefix = str((selected.get("definition") or {}).get("any_delta_prefix", ""))
    tokens = [str(x) for x in row.get("delta_tokens", [])]
    observed_prefixes = {_prefix(token) for token in tokens if token.startswith("DELTA_")}
    known_prefixes = {
        str((item.get("definition") or {}).get("any_delta_prefix", ""))
        for item in proposal.get("candidate_predicates", [])
    }
    if observed_prefixes - known_prefixes:
        label = "AUTO_LABEL_UNSUPPORTED_OR_AMBIGUOUS"
        decision = "abstain"
    elif prefix and prefix in observed_prefixes:
        label = "AUTO_LABEL_STABLE_EFFECT_CHANGE"
        decision = "confirm_candidate"
    else:
        label = "AUTO_LABEL_NO_OBSERVED_CHANGE"
        decision = "reject"
    return {"label_id": label, "decision": decision, "observed_prefixes": sorted(observed_prefixes)}


def proposal_digest(proposal: Mapping[str, Any]) -> str:
    payload = json.dumps(proposal, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "SCHEMA_VERSION",
    "apply_proposal",
    "make_visible_pair",
    "proposal_digest",
    "propose_goal_and_labels",
]
