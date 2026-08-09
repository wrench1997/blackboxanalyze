"""PG-258 multi-head Rule-IR adapter over the frozen PG-249 policy context.

The legacy send/abstain policy remains frozen.  PG-258 adds a separately
trained, higher-capacity representation head for abstract Rule-IR classes and
surface families.  It is trained from bounded process tokens from real local
SQL/XSS replays; oracle labels are supervision only and never enter the input.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


PG258_SCHEMA = "pg258-unified-rule-ir-adapter-v1"
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


class UnifiedRuleIRCapacityAdapter(nn.Module):
    """Trainable token/class heads over a frozen 4096-dim policy feature."""

    def __init__(self, *, d_model: int, hidden_dim: int, token_vocab_size: int) -> None:
        super().__init__()
        self.context_projection = nn.Sequential(
            nn.LayerNorm(int(d_model)),
            nn.Linear(int(d_model), int(hidden_dim)),
            nn.GELU(),
        )
        # The classification position is useful for a local decision, but a
        # Rule-IR class is often determined by a later failure/feedback token
        # in the same bounded trace.  Fuse it with a sequence summary so the
        # head can use the observed trajectory without receiving the oracle
        # label itself.
        self.classification_fusion = nn.Sequential(
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
            nn.GELU(),
        )
        self.token_head = nn.Linear(int(hidden_dim), int(token_vocab_size))
        self.rule_head = nn.Linear(int(hidden_dim), len(RULE_IR_CLASSES))
        self.family_head = nn.Linear(int(hidden_dim), len(FAMILY_CLASSES))

    def forward(self, context: torch.Tensor, *, classification_positions: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.context_projection(context)
        positions = classification_positions.to(device=hidden.device, dtype=torch.long).clamp(min=0, max=hidden.shape[1] - 1)
        local = hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]
        sequence_summary = hidden.mean(dim=1)
        pooled = self.classification_fusion(torch.cat([local, sequence_summary], dim=-1))
        return {
            "token": self.token_head(hidden),
            "rule": self.rule_head(pooled),
            "family": self.family_head(pooled),
        }


def rule_target(name: str) -> int:
    try:
        return RULE_IR_INDEX[str(name)]
    except KeyError as error:
        raise ValueError(f"unknown PG-258 Rule-IR class: {name!r}") from error


def family_target(name: str) -> int:
    try:
        return FAMILY_INDEX[str(name)]
    except KeyError as error:
        raise ValueError(f"unknown PG-258 family: {name!r}") from error


def evaluate_unified_adapter(
    model: UnifiedRuleIRCapacityAdapter,
    context: torch.Tensor,
    token_targets: torch.Tensor,
    rule_targets: torch.Tensor,
    family_targets: torch.Tensor,
    positions: torch.Tensor,
) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        output = model(context, classification_positions=positions)
    token_loss = nn.functional.cross_entropy(
        output["token"].reshape(-1, output["token"].shape[-1]),
        token_targets.reshape(-1),
        ignore_index=0,
    )
    token_pred = output["token"].argmax(-1)
    rule_pred = output["rule"].argmax(-1)
    family_pred = output["family"].argmax(-1)
    valid = token_targets.ne(0)
    result: dict[str, Any] = {
        "token_loss": round(float(token_loss.detach().cpu()), 8),
        "perplexity": round(float(torch.exp(token_loss.detach().cpu().clamp(max=20.0))), 8),
        "next_token_accuracy": round(float(((token_pred == token_targets) & valid).sum().item() / max(int(valid.sum().item()), 1)), 8),
        "token_count": int(valid.sum().item()),
        "rule_accuracy": round(float((rule_pred == rule_targets).float().mean().item()), 8),
        "family_accuracy": round(float((family_pred == family_targets).float().mean().item()), 8),
        "row_count": int(rule_targets.shape[0]),
        "predicted_rule_classes": [RULE_IR_CLASSES[int(index)] for index in rule_pred.detach().cpu().tolist()],
        "predicted_families": [FAMILY_CLASSES[int(index)] for index in family_pred.detach().cpu().tolist()],
    }
    for name, index in RULE_IR_INDEX.items():
        mask = rule_targets.eq(index)
        result[f"{name}_count"] = int(mask.sum().item())
        result[f"{name}_recall"] = round(float(((rule_pred == index) & mask).sum().item() / max(int(mask.sum().item()), 1)), 8)
    return result


__all__ = [
    "FAMILY_CLASSES",
    "FAMILY_INDEX",
    "PG258_SCHEMA",
    "RULE_IR_CLASSES",
    "RULE_IR_INDEX",
    "UnifiedRuleIRCapacityAdapter",
    "evaluate_unified_adapter",
    "family_target",
    "rule_target",
]
