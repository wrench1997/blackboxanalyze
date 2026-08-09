"""Family-free multi-step belief state and fail-closed probe scheduler."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


GENERIC_STATES = ("effect", "input_only", "no_effect", "unknown")
SCHEMA_VERSION = "generic-causal-belief-v1"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize(values: Mapping[str, float]) -> dict[str, float]:
    clipped = {state: max(float(values.get(state, 0.0)), 1e-8) for state in GENERIC_STATES}
    total = sum(clipped.values())
    return {state: clipped[state] / total for state in GENERIC_STATES}


def entropy(values: Mapping[str, float]) -> float:
    posterior = _normalize(values)
    return -sum(value * math.log(value) for value in posterior.values() if value > 0)


def divergence(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    left_n = _normalize(left)
    right_n = _normalize(right)
    midpoint = {state: (left_n[state] + right_n[state]) / 2.0 for state in GENERIC_STATES}
    left_kl = sum(left_n[state] * math.log(left_n[state] / midpoint[state]) for state in GENERIC_STATES)
    right_kl = sum(right_n[state] * math.log(right_n[state] / midpoint[state]) for state in GENERIC_STATES)
    return max(0.0, 0.5 * (left_kl + right_kl))


class GenericBeliefState:
    """Update generic causal states from post-action model projections only."""

    def __init__(self, *, likelihood_power: float = 0.5, prior_mixing: float = 0.25) -> None:
        if not 0 < float(likelihood_power) <= 1 or not 0 < float(prior_mixing) <= 1:
            raise ValueError("belief powers must be in (0, 1]")
        self.likelihood_power = float(likelihood_power)
        self.prior_mixing = float(prior_mixing)
        self.posterior = {state: 1.0 / len(GENERIC_STATES) for state in GENERIC_STATES}
        self.steps: list[dict[str, Any]] = []
        self._seen_evidence: set[str] = set()

    def _posterior_after(self, likelihood: Mapping[str, float]) -> dict[str, float]:
        normalized = _normalize(likelihood)
        softened = _normalize({state: normalized[state] ** self.likelihood_power for state in GENERIC_STATES})
        return _normalize({
            state: (1.0 - self.prior_mixing) * self.posterior[state] + self.prior_mixing * softened[state]
            for state in GENERIC_STATES
        })

    def observe(self, action_id: str, likelihood: Mapping[str, float], *, evidence_hash: str) -> dict[str, Any]:
        evidence_hash = str(evidence_hash)
        if not evidence_hash or len(evidence_hash) < 16:
            raise ValueError("belief update requires a bounded evidence hash")
        before = dict(self.posterior)
        duplicate = evidence_hash in self._seen_evidence
        if duplicate:
            after = before
            gain = 0.0
        else:
            after = self._posterior_after(likelihood)
            gain = divergence(before, _normalize(likelihood))
            self.posterior = after
            self._seen_evidence.add(evidence_hash)
        step = {
            "schema_version": SCHEMA_VERSION,
            "step": len(self.steps) + 1,
            "action_id": str(action_id),
            "prior": before,
            "likelihood": _normalize(likelihood),
            "posterior": dict(after),
            "entropy_before": round(entropy(before), 6),
            "entropy_after": round(entropy(after), 6),
            "information_gain": round(gain, 6),
            "evidence_hash": evidence_hash,
            "accepted": not duplicate,
            "duplicate_evidence": duplicate,
        }
        self.steps.append(step)
        return copy.deepcopy(step)

    def information_gain(self, likelihood: Mapping[str, float]) -> float:
        return divergence(self.posterior, _normalize(likelihood))

    def choose_next(self, candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not candidates:
            raise ValueError("generic belief scheduler requires candidates")
        scored: list[tuple[float, int, Mapping[str, Any], float]] = []
        for index, candidate in enumerate(candidates):
            likelihood = candidate.get("likelihood") if isinstance(candidate.get("likelihood"), Mapping) else {}
            gain = self.information_gain(likelihood)
            score = 0.9 * gain + 0.1 * float(candidate.get("priority", 0.0) or 0.0)
            scored.append((score, -index, candidate, gain))
        _, _, selected, gain = max(scored, key=lambda item: (item[0], item[1]))
        result = copy.deepcopy(dict(selected))
        result["belief_information_gain"] = round(gain, 6)
        return result

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "posterior": dict(self.posterior),
            "entropy": round(entropy(self.posterior), 6),
            "unique_evidence_count": len(self._seen_evidence),
            "steps": copy.deepcopy(self.steps),
        }


def likelihood_from_projection(output: Mapping[str, Any]) -> dict[str, float]:
    """Map generic model output to a soft state likelihood, never a family."""

    composition = output.get("composition") if isinstance(output.get("composition"), Mapping) else {}
    atoms = {str(atom) for atom in composition.get("observed_atoms", [])}
    decision = str(output.get("decision", ""))
    if "candidate_without_surface_delta" in atoms:
        return {"effect": 0.05, "input_only": 0.80, "no_effect": 0.05, "unknown": 0.10}
    if "effect_present" in atoms and decision == "confirm_candidate":
        return {"effect": 0.80, "input_only": 0.05, "no_effect": 0.05, "unknown": 0.10}
    if decision == "reject":
        return {"effect": 0.05, "input_only": 0.05, "no_effect": 0.80, "unknown": 0.10}
    return {"effect": 0.10, "input_only": 0.20, "no_effect": 0.20, "unknown": 0.50}


def schedule_next_action(
    output: Mapping[str, Any],
    *,
    observed_methods: set[str],
    max_steps: int,
    step_count: int,
) -> str:
    """Return only a bounded next action; no action confirms a vulnerability."""

    if not bool(output.get("promotion_eligible") is False):
        return "abstain_invalid_promotion_contract"
    reason = str(output.get("reason", ""))
    composition = output.get("composition") if isinstance(output.get("composition"), Mapping) else {}
    atoms = {str(atom) for atom in composition.get("observed_atoms", [])}
    if reason == "invalid_probe_binding_attestation":
        return "abstain_invalid_binding"
    if "candidate_without_surface_delta" in atoms:
        if len(observed_methods) < 2:
            if step_count >= max_steps:
                return "abstain_budget_exhausted"
            return "repeat_matched_negative_other_method"
        return "await_typed_oracle_then_abstain"
    if "effect_present" in atoms:
        if len(observed_methods) < 2:
            if step_count >= max_steps:
                return "abstain_budget_exhausted"
            return "replay_other_method"
        return "await_typed_oracle"
    if step_count >= max_steps:
        return "abstain_budget_exhausted"
    if len(observed_methods) < 2:
        return "probe_other_method"
    return "abstain_no_repeated_effect"


__all__ = [
    "GENERIC_STATES",
    "GenericBeliefState",
    "SCHEMA_VERSION",
    "divergence",
    "entropy",
    "likelihood_from_projection",
    "schedule_next_action",
]
