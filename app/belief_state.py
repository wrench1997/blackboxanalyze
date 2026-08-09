"""Multi-step Bayesian-style belief state for shadow active probing."""

from __future__ import annotations

import copy
import math
from typing import Any

from .rule_ir_decoder import DECODER_FAMILIES


def _normalise(values: dict[str, float]) -> dict[str, float]:
    clipped = {family: max(float(values.get(family, 0.0)), 1e-8) for family in DECODER_FAMILIES}
    total = sum(clipped.values())
    return {family: value / total for family, value in clipped.items()}


def belief_entropy(belief: dict[str, float]) -> float:
    return -sum(value * math.log(value) for value in belief.values() if value > 0)


def jensen_shannon_divergence(left: dict[str, float], right: dict[str, float]) -> float:
    left = _normalise(left)
    right = _normalise(right)
    midpoint = {family: (left[family] + right[family]) / 2.0 for family in DECODER_FAMILIES}
    kl_left = sum(left[family] * math.log(left[family] / midpoint[family]) for family in DECODER_FAMILIES)
    kl_right = sum(right[family] * math.log(right[family] / midpoint[family]) for family in DECODER_FAMILIES)
    return max(0.0, 0.5 * (kl_left + kl_right))


class MultiStepBelief:
    """Updates a family posterior from visible discriminator likelihoods only."""

    def __init__(self, *, likelihood_power: float = 0.35, prior_mixing: float = 0.20):
        self.likelihood_power = likelihood_power
        self.prior_mixing = prior_mixing
        self.posterior = {family: 1.0 / len(DECODER_FAMILIES) for family in DECODER_FAMILIES}
        self.steps: list[dict[str, Any]] = []
        self._seen_evidence_hashes: set[str] = set()

    def posterior_after(self, probabilities: dict[str, float]) -> dict[str, float]:
        likelihood = _normalise(probabilities)
        softened = _normalise({family: likelihood[family] ** self.likelihood_power for family in DECODER_FAMILIES})
        # Fuse rather than repeatedly multiply posteriors.  This prevents a
        # single miscalibrated surface from collapsing all later beliefs.
        return _normalise({
            family: (1.0 - self.prior_mixing) * self.posterior[family] + self.prior_mixing * softened[family]
            for family in DECODER_FAMILIES
        })

    def information_gain(self, probabilities: dict[str, float]) -> float:
        return jensen_shannon_divergence(self.posterior, _normalise(probabilities))

    def choose_next_probe(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        if not candidates:
            raise ValueError("belief probe candidate list must not be empty")
        scored = []
        for index, row in enumerate(candidates):
            surface = dict(row.get("surface_discriminator") or {})
            probabilities = dict(surface.get("probabilities") or row.get("rule_ir_decoder", {}).get("probabilities") or {})
            gain = self.information_gain(probabilities)
            local_entropy = belief_entropy(_normalise(probabilities))
            score = 0.90 * gain + 0.05 * local_entropy + 0.05 * float(row.get("model_score", 0.0) or 0.0)
            scored.append((score, -index, row, gain, probabilities))
        score, _, row, gain, probabilities = max(scored, key=lambda item: (item[0], item[1]))
        row_copy = copy.deepcopy(row)
        row_copy["belief_probe_score"] = round(score, 6)
        row_copy["belief_information_gain"] = round(gain, 6)
        row_copy["belief_likelihood"] = _normalise(probabilities)
        return row_copy

    def observe(self, action_path: str, probabilities: dict[str, float], *, evidence_hash: str | None = None) -> dict[str, Any]:
        before = dict(self.posterior)
        duplicate_evidence = bool(evidence_hash and evidence_hash in self._seen_evidence_hashes)
        if duplicate_evidence:
            after = before
            gain = 0.0
        else:
            after = self.posterior_after(probabilities)
            gain = jensen_shannon_divergence(before, _normalise(probabilities))
            self.posterior = after
            if evidence_hash:
                self._seen_evidence_hashes.add(evidence_hash)
        step = {
            "step": len(self.steps) + 1,
            "action_path": action_path,
            "prior": before,
            "likelihood": _normalise(probabilities),
            "posterior": dict(after),
            "entropy_before": round(belief_entropy(before), 6),
            "entropy_after": round(belief_entropy(after), 6),
            "entropy_change": round(belief_entropy(before) - belief_entropy(after), 6),
            "information_gain": round(gain, 6),
            "evidence_hash": evidence_hash,
            "accepted": not duplicate_evidence,
            "duplicate_evidence": duplicate_evidence,
        }
        self.steps.append(step)
        return step

    def snapshot(self) -> dict[str, Any]:
        return {
            "posterior": dict(self.posterior),
            "entropy": round(belief_entropy(self.posterior), 6),
            "unique_evidence_count": len(self._seen_evidence_hashes),
            "steps": copy.deepcopy(self.steps),
        }
