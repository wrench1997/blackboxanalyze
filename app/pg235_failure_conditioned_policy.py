"""PG-235 failure-conditioned action head and wire-row normalizer."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from .pg230_next_token_quality_funnel import (
    LANE_INDEX,
    LANES,
    REPAIR_ACTIONS,
    REPAIR_INDEX,
    _surface_class,
    digest,
    prepare_record,
)
from .pg233_cross_family_capacity import add_family_context, family_class


PG235_SCHEMA = "pg235-failure-conditioned-policy-v1"
ACTION_CLASSES = ("abstain", "send_candidate", "recheck_oracle")
ACTION_INDEX = {name: index for index, name in enumerate(ACTION_CLASSES)}


class FrozenXXLFailurePolicy(nn.Module):
    """Causal policy heads over a frozen sequence representation."""

    def __init__(self, *, d_model: int, hidden_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.context_projection = nn.Sequential(nn.LayerNorm(int(d_model)), nn.Linear(int(d_model), int(hidden_dim)), nn.GELU())
        self.token_head = nn.Linear(int(hidden_dim), int(vocab_size))
        self.lane_head = nn.Linear(int(hidden_dim), len(LANES))
        self.repair_head = nn.Linear(int(hidden_dim), len(REPAIR_ACTIONS))
        self.action_head = nn.Linear(int(hidden_dim), len(ACTION_CLASSES))

    def forward(self, context: torch.Tensor, *, classification_positions: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.context_projection(context)
        positions = classification_positions.to(device=hidden.device, dtype=torch.long).clamp(min=0, max=hidden.shape[1] - 1)
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]
        return {"token": self.token_head(hidden), "lane": self.lane_head(pooled), "repair": self.repair_head(pooled), "action": self.action_head(pooled)}


def prepare_wire_record(row: Mapping[str, Any]) -> dict[str, Any]:
    family = family_class(row.get("family"), row.get("route"))
    surface_role = {"sql": "sql_surface", "dom": "dom_surface", "redirect": "redirect_surface"}.get(family, "generic_surface")
    typed = bool(row.get("typed_effect_confirmed") or row.get("dom_surface_effect_confirmed"))
    result_verified = bool(row.get("result_fixture_verified"))
    payload_grounded = bool(family == "sql" and typed and result_verified and row.get("negative_clean", False))
    raw = {
        "source": "pg234_pikachu_wire_catalog",
        "seed": int(row.get("seed", 0) or 0),
        "surface_role": surface_role,
        "method": str(row.get("method", "GET")).upper(),
        "status_class": "2xx",
        "field_count": len(row.get("fields", [])),
        "history_len": 0,
        "fresh_reset_ok": bool(row.get("fresh_reset", False)),
        "reset_completed": bool(row.get("fresh_reset", False)),
        "reset_not_attempted": False,
        "candidate_sent": bool(row.get("candidate_sent", False)),
        "oracle_available": bool(typed or result_verified),
        "typed_effect_confirmed": typed,
        "typed_effect_observed": typed,
        "result_fixture_verified": result_verified,
        "candidate_reference_agreement": bool(row.get("candidate_reference_agreement", False)),
        "negative_clean": bool(row.get("negative_clean", False)),
        "binding_valid": True,
        "transport_error": False,
        "result_mismatch_observed": False,
        "next_step": "abstain" if not payload_grounded else "recheck_oracle",
        "previous_feedback": "none",
        "candidate_result_present": typed,
        "model_claimed_positive": False,
        "model_abstained": not payload_grounded,
        "evidence_hash": str(row.get("evidence_hash", "")),
        "payload_grounded_eligible": payload_grounded,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
    }
    if len(raw["evidence_hash"]) != 64:
        raw["evidence_hash"] = digest({"source": raw["source"], "seed": raw["seed"], "family": family, "method": raw["method"]})
    return add_family_context(prepare_record(raw), family=family, channel=raw["method"], source_role="evaluation_only")


def action_target(record: Mapping[str, Any]) -> str:
    if bool(record.get("payload_grounded_eligible", False)):
        return "send_candidate"
    return "abstain"


__all__ = ["ACTION_CLASSES", "ACTION_INDEX", "FrozenXXLFailurePolicy", "PG235_SCHEMA", "action_target", "prepare_wire_record"]
