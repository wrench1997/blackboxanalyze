"""Small, explicit Rule IR decision decoder used by PG-115.

PG-115 is intentionally a compact training experiment.  It is not a
vulnerability scanner and it does not emit executable payloads.  The decoder
only consumes the family/oracle-blind projections used by the replay bridge
and emits one of four abstract next-step decisions.  The evaluator label is
kept outside the model input and is used only for offline scoring.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Sequence

import torch
from torch import nn

from .rule_ir_decoder import validate_abstract_rule_ir


PG115_DECISIONS = (
    "confirmed_positive",
    "confirmed_negative",
    "candidate",
    "abstain",
)
FEATURE_DIM = 40
SCHEMA_VERSION = "pg115-small-rule-ir-decoder-v1"


def _bucket(value: str, size: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8", errors="replace"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % size


def _status_class(value: Any) -> str:
    if isinstance(value, str):
        return value.casefold()
    return "unknown"


def _shape_flags(shape_class: Any) -> list[float]:
    """Map a response-shape descriptor to bounded generic flags.

    These are shape words, not target/family names.  The raw descriptor is
    never persisted in the model feature vector.  The small semantic flags
    make the trial able to transfer from synthetic names to the PG-114 names
    without memorising a route, probe, family or target slot.
    """

    text = str(shape_class).casefold()
    tokens = (
        ("policy", "header", "transition"),
        ("shape", "decoy", "body"),
        ("opaque", "unknown", "untyped"),
        ("stable", "same", "neutral"),
    )
    return [float(any(token in text for token in group)) for group in tokens]


def model_input_feature_vector(
    model_input: dict[str, Any],
    *,
    prior_inputs: Iterable[dict[str, Any]] = (),
) -> list[float]:
    """Project one visible replay row into a fixed, oracle-blind vector.

    Identifiers, hashes, raw probe values, evaluator fields and target slots
    are deliberately ignored.  ``prior_inputs`` contains only earlier visible
    model inputs from the same episode, allowing the decoder to learn the
    minimum useful multi-step signal (a candidate replayed on another method).
    """

    action = model_input.get("action_manifest") or {}
    baseline = model_input.get("baseline_projection") or {}
    response = model_input.get("response_projection") or {}
    belief = model_input.get("belief_before") or {}
    bsp = response.get("bsp_core_projection") or {}
    values = [0.0] * FEATURE_DIM

    method = str(action.get("method", "")).upper()
    values[0] = float(method == "GET")
    values[1] = float(method == "POST")
    values[2] = float(bool(response.get("candidate_signal")))
    values[3] = float(bool(response.get("shape_changed")))
    values[4] = float(bool(response.get("policy_header_changed")))

    status = _status_class(response.get("status_class"))
    values[5] = float(status == "2xx")
    values[6] = float(status == "3xx")
    values[7] = float(status == "4xx")
    values[8] = float(status == "5xx")

    body_bucket = str(response.get("body_length_bucket", baseline.get("body_length_bucket", "")))
    values[9] = float(body_bucket == "0")
    values[10] = float("1-255" in body_bucket)
    values[11] = float("256-4095" in body_bucket)
    values[12] = float("4096" in body_bucket or "large" in body_bucket.casefold())

    values[13:17] = _shape_flags(response.get("shape_class", ""))
    values[17] = min(float(response.get("noise_bucket", 0) or 0) / 8.0, 1.0)
    values[18] = min(float(belief.get("effect", 0.0) or 0.0), 1.0)
    values[19] = min(float(belief.get("input_only", 0.0) or 0.0), 1.0)
    values[20] = min(float(belief.get("no_effect", 0.0) or 0.0), 1.0)
    values[21] = min(float(belief.get("unknown", 0.0) or 0.0), 1.0)
    values[22] = min(abs(float(bsp.get("leaf_mass_error", 0.0) or 0.0)), 1.0)
    values[23] = min(float(len(bsp.get("selected_leaf_ids") or [])) / 8.0, 1.0)
    values[24] = min(float(bsp.get("topology_version", 0) or 0) / 8.0, 1.0)
    values[25] = min(float(len(action.get("encoding_chain") or [])) / 4.0, 1.0)
    values[26] = float(action.get("placement") == "query")
    values[27] = float(action.get("placement") == "body")

    prior = list(prior_inputs)
    prior_responses = [row.get("response_projection") or {} for row in prior]
    prior_candidates = sum(bool(row.get("candidate_signal")) for row in prior_responses)
    values[28] = min(float(len(prior)) / 4.0, 1.0)
    values[29] = min(float(prior_candidates) / 2.0, 1.0)
    values[30] = float(any(str((row.get("action_manifest") or {}).get("method", "")).upper() != method for row in prior))
    values[31] = float(any(bool((row.get("response_projection") or {}).get("candidate_signal")) for row in prior))
    values[32] = float(any(bool((row.get("response_projection") or {}).get("policy_header_changed")) for row in prior))
    values[33] = float(any(bool((row.get("response_projection") or {}).get("shape_changed")) for row in prior))
    values[34] = float(bool(response.get("candidate_signal")) and bool(values[30]))
    values[35] = float(bool(response.get("candidate_signal")) and bool(values[31]) and bool(values[30]))

    # Four anonymous shape buckets provide a little capacity for varied
    # fixture descriptors while remaining independent of route and target IDs.
    shape = str(response.get("shape_class", ""))
    values[36 + _bucket(shape, 4)] = 1.0
    return values


RULE_IR_BY_DECISION: dict[str, dict[str, Any]] = {
    "confirmed_positive": {
        "op": "and",
        "args": [
            {"op": "policy_slot", "name": "effect_confirmed"},
            {"op": "policy_slot", "name": "matched_negative_control"},
        ],
    },
    "confirmed_negative": {"op": "policy_slot", "name": "effect_rejected"},
    "candidate": {"op": "policy_slot", "name": "candidate_needs_replay"},
    "abstain": {"op": "policy_slot", "name": "unknown_abstain"},
}

for _rule in RULE_IR_BY_DECISION.values():
    validate_abstract_rule_ir(_rule)


class SmallRuleIRDecisionDecoder(nn.Module):
    """A tiny GPU-friendly MLP for the PG-115 proof-of-training trial."""

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_dim: int = 48):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(hidden_dim, len(PG115_DECISIONS))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(features))

    @torch.inference_mode()
    def decode(self, features: torch.Tensor) -> list[dict[str, Any]]:
        probabilities = torch.softmax(self(features), dim=-1)
        values, indices = probabilities.max(dim=-1)
        outputs: list[dict[str, Any]] = []
        for confidence, index, row in zip(values.detach().cpu(), indices.detach().cpu(), probabilities.detach().cpu()):
            decision = PG115_DECISIONS[int(index)]
            outputs.append(
                {
                    "decision": decision,
                    "confidence": round(float(confidence), 6),
                    "rule_ir": RULE_IR_BY_DECISION[decision],
                    "probabilities": {
                        name: round(float(probability), 6)
                        for name, probability in zip(PG115_DECISIONS, row)
                    },
                }
            )
        return outputs


def decision_index(decision: str) -> int:
    if decision not in PG115_DECISIONS:
        raise ValueError(f"unknown PG-115 decision: {decision}")
    return PG115_DECISIONS.index(decision)


def canonical_model_input(model_input: dict[str, Any]) -> dict[str, Any]:
    """Return only the persisted, family/oracle-blind model-facing fields."""

    action = model_input.get("action_manifest") or {}
    baseline = model_input.get("baseline_projection") or {}
    response = model_input.get("response_projection") or {}
    belief = model_input.get("belief_before") or {}
    bsp = response.get("bsp_core_projection") or {}
    return {
        "action_manifest": {
            "method": action.get("method"),
            "placement": action.get("placement"),
            "encoding_chain": list(action.get("encoding_chain") or []),
            "safety": dict(action.get("safety") or {}),
        },
        "baseline_projection": {
            "body_length_bucket": baseline.get("body_length_bucket"),
            "status_class": baseline.get("status_class"),
        },
        "response_projection": {
            "candidate_signal": bool(response.get("candidate_signal")),
            "noise_bucket": int(response.get("noise_bucket", 0) or 0),
            "policy_header_changed": bool(response.get("policy_header_changed")),
            "shape_changed": bool(response.get("shape_changed")),
            "shape_class": response.get("shape_class"),
            "status_class": response.get("status_class"),
            "bsp_core_projection": {
                "leaf_mass_error": float(bsp.get("leaf_mass_error", 0.0) or 0.0),
                "selected_leaf_count": len(bsp.get("selected_leaf_ids") or []),
                "topology_version": int(bsp.get("topology_version", 0) or 0),
            },
        },
        "belief_before": {
            key: float(belief.get(key, 0.0) or 0.0)
            for key in ("effect", "input_only", "no_effect", "unknown")
        },
    }

