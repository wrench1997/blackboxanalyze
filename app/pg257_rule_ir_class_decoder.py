"""PG-257 Rule-IR class decoder over a frozen XXL context.

The decoder predicts the next abstract binding class (syntax, boolean, or
wide-byte boundary) from bounded process tokens.  Route names, raw payloads,
response bodies, and evaluator keys are intentionally outside the input.  A
separate evaluator supplies the target class during dataset construction; it
is not available to the decoder at inference time.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


PG257_SCHEMA = "pg257-rule-ir-class-decoder-v1"
RULE_CLASSES = ("syntax_boundary", "blind_boolean", "widebyte_escape_boundary")
RULE_CLASS_INDEX = {name: index for index, name in enumerate(RULE_CLASSES)}


class RuleIRClassDecoder(nn.Module):
    """Trainable token/lane-independent class head with an auxiliary LM head."""

    def __init__(self, *, d_model: int, hidden_dim: int, token_vocab_size: int) -> None:
        super().__init__()
        self.context_projection = nn.Sequential(nn.LayerNorm(int(d_model)), nn.Linear(int(d_model), int(hidden_dim)), nn.GELU())
        self.token_head = nn.Linear(int(hidden_dim), int(token_vocab_size))
        self.rule_head = nn.Linear(int(hidden_dim), len(RULE_CLASSES))

    def forward(self, context: torch.Tensor, *, classification_positions: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.context_projection(context)
        positions = classification_positions.to(device=hidden.device, dtype=torch.long).clamp(min=0, max=hidden.shape[1] - 1)
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]
        return {"token": self.token_head(hidden), "rule": self.rule_head(pooled)}


def class_target(name: str) -> int:
    try:
        return RULE_CLASS_INDEX[str(name)]
    except KeyError as error:
        raise ValueError(f"unknown PG-257 Rule-IR class: {name!r}") from error


def evaluate_decoder(model: RuleIRClassDecoder, context: torch.Tensor, token_targets: torch.Tensor, rule_targets: torch.Tensor, positions: torch.Tensor) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        output = model(context, classification_positions=positions)
    token_loss = nn.functional.cross_entropy(output["token"].reshape(-1, output["token"].shape[-1]), token_targets.reshape(-1), ignore_index=0)
    rule_pred = output["rule"].argmax(-1)
    token_pred = output["token"].argmax(-1)
    valid = token_targets.ne(0)
    result: dict[str, Any] = {
        "token_loss": round(float(token_loss.detach().cpu()), 8),
        "perplexity": round(float(torch.exp(token_loss.detach().cpu().clamp(max=20.0))), 8),
        "next_token_accuracy": round(float(((token_pred == token_targets) & valid).sum().item() / max(int(valid.sum().item()), 1)), 8),
        "token_count": int(valid.sum().item()),
        "rule_accuracy": round(float((rule_pred == rule_targets).float().mean().item()), 8),
        "row_count": int(rule_targets.shape[0]),
    }
    for name, index in RULE_CLASS_INDEX.items():
        mask = rule_targets.eq(index)
        result[f"{name}_count"] = int(mask.sum().item())
        result[f"{name}_recall"] = round(float(((rule_pred == index) & mask).sum().item() / max(int(mask.sum().item()), 1)), 8)
    result["predicted_classes"] = [RULE_CLASSES[int(index)] for index in rule_pred.detach().cpu().tolist()]
    return result


__all__ = ["PG257_SCHEMA", "RULE_CLASSES", "RULE_CLASS_INDEX", "RuleIRClassDecoder", "class_target", "evaluate_decoder"]
