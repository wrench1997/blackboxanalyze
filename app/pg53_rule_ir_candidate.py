"""Quarantined PG-53 Rule IR candidate with the complete local family set.

The candidate is intentionally separate from the five-family Pikachu
checkpoint.  It consumes only the bounded visible projection, emits one of
eight non-executable abstract Rule IR templates, and treats the ordinary
response class as abstention.  Training/evaluation policy remains owned by
the PG-53 runner and capability gate.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn

from .rule_ir_decoder import FEATURE_DIM, validate_abstract_rule_ir


PG53_MODEL_FAMILIES = (
    "xss",
    "injection",
    "authentication",
    "access_control",
    "logic",
    "url_redirect",
    "input_validation",
    "command_injection",
    "ordinary_response",
)
PG53_TEMPLATE_FAMILIES = PG53_MODEL_FAMILIES[:-1]


def abstract_pg53_rule_ir(family: str) -> dict[str, Any] | None:
    templates: dict[str, dict[str, Any]] = {
        "xss": {"op": "not", "arg": {"op": "html_creates_nodes", "arg": {"op": "policy_slot", "name": "untrusted_text"}}},
        "injection": {"op": "policy_slot", "name": "untrusted_data_cannot_change_interpreter_structure"},
        "authentication": {"op": "and", "args": [{"op": "policy_slot", "name": "identity_proof_valid"}, {"op": "policy_slot", "name": "credential_policy_satisfied"}]},
        "access_control": {"op": "and", "args": [{"op": "policy_slot", "name": "subject_authenticated"}, {"op": "policy_slot", "name": "subject_authorized_for_resource"}]},
        "logic": {"op": "and", "args": [{"op": "policy_slot", "name": "invariant_holds"}, {"op": "policy_slot", "name": "state_replay_is_valid"}]},
        "url_redirect": {"op": "origin_eq", "left": {"op": "policy_slot", "name": "candidate_url"}, "right": {"op": "policy_slot", "name": "trusted_origin"}},
        "input_validation": {"op": "and", "args": [{"op": "policy_slot", "name": "representation_is_canonical"}, {"op": "policy_slot", "name": "value_is_in_declared_domain"}]},
        "command_injection": {"op": "not", "arg": {"op": "policy_slot", "name": "untrusted_data_reaches_command_interpreter"}},
    }
    if family == "ordinary_response":
        return None
    if family not in templates:
        raise KeyError(f"unsupported PG-53 family: {family}")
    result = copy.deepcopy(templates[family])
    validate_abstract_rule_ir(result)
    return result


class PG53RuleIRCandidate(nn.Module):
    """Small MLP used only as a quarantined research candidate."""

    def __init__(self, *, hidden_dim: int = 128, embedding_dim: int = 96, dropout: float = 0.08):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(embedding_dim, len(PG53_MODEL_FAMILIES))

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(features))

    @torch.inference_mode()
    def decode(self, features: torch.Tensor, *, abstain_threshold: float = 0.60, margin_threshold: float = 0.08) -> list[dict[str, Any]]:
        probabilities = torch.softmax(self(features), dim=-1)
        values, indices = probabilities.max(dim=-1)
        decoded: list[dict[str, Any]] = []
        for confidence, index, row in zip(values.cpu(), indices.cpu(), probabilities.cpu()):
            candidate = PG53_MODEL_FAMILIES[int(index)]
            sorted_values = torch.sort(row, descending=True).values
            margin = float(sorted_values[0] - sorted_values[1])
            accepted = candidate != "ordinary_response" and float(confidence) >= float(abstain_threshold) and margin >= float(margin_threshold)
            decoded.append({
                "candidate_family": candidate,
                "family": candidate if accepted else None,
                "confidence": round(float(confidence), 6),
                "margin": round(margin, 6),
                "abstained": not accepted,
                "rule_ir": abstract_pg53_rule_ir(candidate) if accepted else None,
                "probabilities": {name: round(float(value), 6) for name, value in zip(PG53_MODEL_FAMILIES, row)},
            })
        return decoded


__all__ = ["PG53_MODEL_FAMILIES", "PG53_TEMPLATE_FAMILIES", "PG53RuleIRCandidate", "abstract_pg53_rule_ir"]
