"""PG-260 active-belief adapter with an explicit unknown-family abstain head."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from .pg259_active_belief_rule_ir_adapter import (
    ActiveBeliefRuleIRAdapter,
    BELIEF_CLASSES,
    FAMILY_CLASSES,
    PROBE_CLASSES,
    RULE_IR_CLASSES,
    belief_target,
    family_target,
    probe_target,
    rule_target,
)


ABSTAIN_CLASSES = ("continue_family", "unknown_family_abstain")


class PG260ActiveBeliefAdapter(ActiveBeliefRuleIRAdapter):
    """Reuse the frozen-context fusion and add one deliberately narrow head."""

    def __init__(self, *, d_model: int, hidden_dim: int, token_vocab_size: int) -> None:
        super().__init__(d_model=d_model, hidden_dim=hidden_dim, token_vocab_size=token_vocab_size)
        self.unknown_abstain = nn.Linear(hidden_dim, len(ABSTAIN_CLASSES))

    def forward(self, context: torch.Tensor, *, classification_positions: torch.Tensor, attention_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        output = super().forward(context, classification_positions=classification_positions, attention_mask=attention_mask, return_pooled=True)
        pooled = output["pooled"]
        output["unknown_abstain"] = self.unknown_abstain(pooled)
        return output


def unknown_abstain_target(row: Mapping[str, Any]) -> int:
    family = str(row.get("family_class", "other"))
    rule = str(row.get("rule_ir_class", "other"))
    return 1 if family == "other" or rule == "other" else 0


def evaluate_pg260_adapter(
    model: PG260ActiveBeliefAdapter,
    context: torch.Tensor,
    token_targets: torch.Tensor,
    rule_targets: torch.Tensor,
    family_targets: torch.Tensor,
    belief_targets: torch.Tensor,
    probe_targets: torch.Tensor,
    unknown_targets: torch.Tensor,
    positions: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        output = model(context, classification_positions=positions, attention_mask=attention_mask)
    token_pred = output["token"].argmax(-1)
    token_mask = token_targets.ne(0)
    token_correct = (token_pred.eq(token_targets) & token_mask).sum().item()
    token_total = token_mask.sum().item()

    def accuracy(name: str, targets: torch.Tensor) -> tuple[int, int, float]:
        pred = output[name].argmax(-1)
        correct = int(pred.eq(targets).sum().item())
        total = int(targets.numel())
        return correct, total, correct / total if total else 0.0

    rule_correct, rule_total, rule_accuracy = accuracy("rule", rule_targets)
    family_correct, family_total, family_accuracy = accuracy("family", family_targets)
    belief_correct, belief_total, belief_accuracy = accuracy("belief", belief_targets)
    probe_correct, probe_total, probe_accuracy = accuracy("probe", probe_targets)
    abstain_correct, abstain_total, abstain_accuracy = accuracy("unknown_abstain", unknown_targets)
    loss = nn.functional.cross_entropy(output["token"].reshape(-1, output["token"].shape[-1]), token_targets.reshape(-1), ignore_index=0)
    return {
        "token_accuracy": float(token_correct / token_total if token_total else 0.0),
        "token_loss": float(loss.detach().cpu()),
        "rule_accuracy": rule_accuracy,
        "family_accuracy": family_accuracy,
        "belief_accuracy": belief_accuracy,
        "probe_accuracy": probe_accuracy,
        "unknown_abstain_accuracy": abstain_accuracy,
        "token_correct": int(token_correct),
        "token_count": int(token_total),
        "rule_correct": rule_correct,
        "rule_count": rule_total,
        "family_correct": family_correct,
        "family_count": family_total,
        "belief_correct": belief_correct,
        "belief_count": belief_total,
        "probe_correct": probe_correct,
        "probe_count": probe_total,
        "unknown_abstain_correct": abstain_correct,
        "unknown_abstain_count": abstain_total,
    }


__all__ = [
    "ABSTAIN_CLASSES",
    "BELIEF_CLASSES",
    "FAMILY_CLASSES",
    "PROBE_CLASSES",
    "RULE_IR_CLASSES",
    "PG260ActiveBeliefAdapter",
    "belief_target",
    "evaluate_pg260_adapter",
    "family_target",
    "probe_target",
    "rule_target",
    "unknown_abstain_target",
]
