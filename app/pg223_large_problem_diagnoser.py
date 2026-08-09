"""PG-223 frozen-XXL context adapter for the PG-222 problem diagnoser.

The 101M-parameter causal trace body is loaded from the existing Pikachu
surface-matrix checkpoint and kept frozen.  Only a small diagnostic adapter is
trained.  This isolates the question "does a larger process representation
help diagnosis?" from the separate question of whether the base model should
be changed.  Inputs are tokenized bounded observations; route names, raw
payloads and response bodies are not included.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .pg222_problem_diagnoser import (
    DIAGNOSIS_INDEX,
    DIAGNOSIS_NAMES,
    FEATURE_DIM,
    NEXT_STEP_INDEX,
    NEXT_STEP_NAMES,
    _metrics,
    _feature_view,
    diagnose_features,
)


PG223_SCHEMA = "pg223-frozen-xxl-problem-diagnoser-v1"


class LargeProblemDiagnoserAdapter(nn.Module):
    """Trainable diagnosis/next-step heads over frozen XXL hidden states."""

    def __init__(self, *, d_model: int = 1024, hidden_dim: int = 128) -> None:
        super().__init__()
        d_model = int(d_model)
        hidden_dim = int(hidden_dim)
        self.context_projection = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, hidden_dim), nn.GELU())
        self.structured_projection = nn.Sequential(nn.Linear(FEATURE_DIM, hidden_dim), nn.GELU())
        self.shared = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.diagnosis_head = nn.Linear(hidden_dim, len(DIAGNOSIS_NAMES))
        self.next_step_head = nn.Linear(hidden_dim, len(NEXT_STEP_NAMES))

    def forward(self, context: torch.Tensor, structured: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.shared(torch.cat([self.context_projection(context), self.structured_projection(structured)], dim=-1))
        return {"diagnosis": self.diagnosis_head(hidden), "next_step": self.next_step_head(hidden)}


def structured_tensor(rows: Sequence[Mapping[str, Any]], device: torch.device) -> torch.Tensor:
    return torch.tensor([diagnose_features(_feature_view(row)) for row in rows], dtype=torch.float32, device=device)


def target_tensors(rows: Sequence[Mapping[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    diagnosis = torch.tensor([DIAGNOSIS_INDEX[str(row["diagnosis"])] for row in rows], dtype=torch.long, device=device)
    next_step = torch.tensor([NEXT_STEP_INDEX[str(row["next_step"])] for row in rows], dtype=torch.long, device=device)
    return diagnosis, next_step


def train_large_adapter(
    model: LargeProblemDiagnoserAdapter,
    train_context: torch.Tensor,
    holdout_context: torch.Tensor,
    train_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
    *,
    epochs: int = 100,
    learning_rate: float = 1e-3,
) -> dict[str, Any]:
    if train_context.shape[0] != len(train_rows) or holdout_context.shape[0] != len(holdout_rows):
        raise ValueError("PG-223 context/row count mismatch")
    device = train_context.device
    train_structured = structured_tensor(train_rows, device)
    hold_structured = structured_tensor(holdout_rows, device)
    train_targets, train_steps = target_tensors(train_rows, device)
    hold_targets, hold_steps = target_tensors(holdout_rows, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=0.01)
    counts = torch.bincount(train_targets, minlength=len(DIAGNOSIS_NAMES)).float().clamp_min(1.0)
    weights = (counts.sum() / counts).to(device)
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        outputs = model(train_context, train_structured)
        loss = nn.functional.cross_entropy(outputs["diagnosis"], train_targets, weight=weights) + 0.35 * nn.functional.cross_entropy(outputs["next_step"], train_steps)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch in {1, int(epochs)} or epoch % 25 == 0:
            model.eval()
            with torch.inference_mode():
                hold_outputs = model(holdout_context, hold_structured)
            history.append({"epoch": epoch, "loss": round(float(loss.detach().cpu()), 8), "holdout": _metrics(hold_outputs, holdout_rows, device)})
    model.eval()
    with torch.inference_mode():
        train_outputs = model(train_context, train_structured)
        hold_outputs = model(holdout_context, hold_structured)
    return {
        "schema_version": PG223_SCHEMA,
        "d_model": int(train_context.shape[1]),
        "feature_dim": FEATURE_DIM,
        "train_rows": len(train_rows),
        "holdout_rows": len(holdout_rows),
        "epochs": int(epochs),
        "history": history,
        "train": _metrics(train_outputs, train_rows, device),
        "holdout": _metrics(hold_outputs, holdout_rows, device),
    }


__all__ = ["LargeProblemDiagnoserAdapter", "PG223_SCHEMA", "structured_tensor", "target_tensors", "train_large_adapter"]
