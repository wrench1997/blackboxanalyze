"""PG-259 active-belief Rule-IR adapter.

The frozen PG-249 body remains the language/context representation.  This
adapter adds two small heads for the experiment's multi-step belief loop:
``belief`` summarizes the current evidence state and ``probe`` chooses the
next *abstract* observation to request.  Oracle outcomes are targets only;
they are never concatenated into the model input.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


PG259_SCHEMA = "pg259-active-belief-rule-ir-adapter-v1"
RULE_IR_CLASSES = (
    "sql_syntax",
    "sql_boolean",
    "sql_widebyte",
    "dom_marker",
    "oracle_gap",
    "other",
)
RULE_IR_INDEX = {name: index for index, name in enumerate(RULE_IR_CLASSES)}
FAMILY_CLASSES = ("sql", "dom", "other")
FAMILY_INDEX = {name: index for index, name in enumerate(FAMILY_CLASSES)}
BELIEF_CLASSES = ("confirmed_effect", "needs_reference", "oracle_gap", "repair_environment")
BELIEF_INDEX = {name: index for index, name in enumerate(BELIEF_CLASSES)}
PROBE_CLASSES = ("replay_confirm", "reference_probe", "negative_control", "reset_environment")
PROBE_INDEX = {name: index for index, name in enumerate(PROBE_CLASSES)}


class ActiveBeliefRuleIRAdapter(nn.Module):
    """Trainable multi-head adapter over a frozen sequence context."""

    def __init__(self, *, d_model: int, hidden_dim: int, token_vocab_size: int) -> None:
        super().__init__()
        self.context_projection = nn.Sequential(
            nn.LayerNorm(int(d_model)),
            nn.Linear(int(d_model), int(hidden_dim)),
            nn.GELU(),
        )
        self.classification_fusion = nn.Sequential(
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
            nn.GELU(),
        )
        self.token_head = nn.Linear(int(hidden_dim), int(token_vocab_size))
        self.rule_head = nn.Linear(int(hidden_dim), len(RULE_IR_CLASSES))
        self.family_head = nn.Linear(int(hidden_dim), len(FAMILY_CLASSES))
        self.belief_head = nn.Linear(int(hidden_dim), len(BELIEF_CLASSES))
        self.probe_head = nn.Linear(int(hidden_dim), len(PROBE_CLASSES))

    def forward(self, context: torch.Tensor, *, classification_positions: torch.Tensor, attention_mask: torch.Tensor | None = None, return_pooled: bool = False) -> dict[str, torch.Tensor]:
        hidden = self.context_projection(context)
        positions = classification_positions.to(device=hidden.device, dtype=torch.long).clamp(min=0, max=hidden.shape[1] - 1)
        local = hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]
        if attention_mask is None:
            pooled_context = hidden.mean(dim=1)
        else:
            mask = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
            pooled_context = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled = self.classification_fusion(torch.cat([local, pooled_context], dim=-1))
        output = {
            "token": self.token_head(hidden),
            "rule": self.rule_head(pooled),
            "family": self.family_head(pooled),
            "belief": self.belief_head(pooled),
            "probe": self.probe_head(pooled),
        }
        if return_pooled:
            output["pooled"] = pooled
        return output


def _target(mapping: dict[str, int], name: str, label: str) -> int:
    try:
        return mapping[str(name)]
    except KeyError as error:
        raise ValueError(f"unknown PG-259 {label}: {name!r}") from error


def rule_target(name: str) -> int:
    return _target(RULE_IR_INDEX, name, "Rule-IR class")


def family_target(name: str) -> int:
    return _target(FAMILY_INDEX, name, "family")


def belief_target(name: str) -> int:
    return _target(BELIEF_INDEX, name, "belief")


def probe_target(name: str) -> int:
    return _target(PROBE_INDEX, name, "probe")


def evaluate_active_adapter(
    model: ActiveBeliefRuleIRAdapter,
    context: torch.Tensor,
    token_targets: torch.Tensor,
    rule_targets: torch.Tensor,
    family_targets: torch.Tensor,
    belief_targets: torch.Tensor,
    probe_targets: torch.Tensor,
    positions: torch.Tensor,
) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        output = model(context, classification_positions=positions)
    valid = token_targets.ne(0)
    token_loss = nn.functional.cross_entropy(
        output["token"].reshape(-1, output["token"].shape[-1]),
        token_targets.reshape(-1),
        ignore_index=0,
    )
    token_pred = output["token"].argmax(-1)
    result: dict[str, Any] = {
        "token_loss": round(float(token_loss.detach().cpu()), 8),
        "perplexity": round(float(torch.exp(token_loss.detach().cpu().clamp(max=20.0))), 8),
        "next_token_accuracy": round(float(((token_pred == token_targets) & valid).sum().item() / max(int(valid.sum().item()), 1)), 8),
        "token_count": int(valid.sum().item()),
        "rule_accuracy": round(float((output["rule"].argmax(-1) == rule_targets).float().mean().item()), 8),
        "family_accuracy": round(float((output["family"].argmax(-1) == family_targets).float().mean().item()), 8),
        "belief_accuracy": round(float((output["belief"].argmax(-1) == belief_targets).float().mean().item()), 8),
        "probe_accuracy": round(float((output["probe"].argmax(-1) == probe_targets).float().mean().item()), 8),
        "row_count": int(rule_targets.shape[0]),
    }
    rule_pred = output["rule"].argmax(-1)
    for name, index in RULE_IR_INDEX.items():
        mask = rule_targets.eq(index)
        result[f"{name}_count"] = int(mask.sum().item())
        result[f"{name}_recall"] = round(float(((rule_pred == index) & mask).sum().item() / max(int(mask.sum().item()), 1)), 8)
    result["predicted_rule_classes"] = [RULE_IR_CLASSES[int(i)] for i in rule_pred.detach().cpu().tolist()]
    result["predicted_families"] = [FAMILY_CLASSES[int(i)] for i in output["family"].argmax(-1).detach().cpu().tolist()]
    result["predicted_beliefs"] = [BELIEF_CLASSES[int(i)] for i in output["belief"].argmax(-1).detach().cpu().tolist()]
    result["predicted_probes"] = [PROBE_CLASSES[int(i)] for i in output["probe"].argmax(-1).detach().cpu().tolist()]
    return result


__all__ = [
    "ActiveBeliefRuleIRAdapter",
    "BELIEF_CLASSES",
    "BELIEF_INDEX",
    "FAMILY_CLASSES",
    "FAMILY_INDEX",
    "PG259_SCHEMA",
    "PROBE_CLASSES",
    "PROBE_INDEX",
    "RULE_IR_CLASSES",
    "RULE_IR_INDEX",
    "belief_target",
    "evaluate_active_adapter",
    "family_target",
    "probe_target",
    "rule_target",
]
