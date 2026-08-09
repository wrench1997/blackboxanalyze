"""History-aware process policy for bounded local replay.

This is a small policy head, not a payload generator.  It consumes route
metadata and the previous bounded feedback status, then chooses one of
``abstain``, ``safe_candidate`` or ``retry_alternate``.  Raw probes, bodies,
oracle labels and secrets are not features.  The head is deliberately kept
separate from the frozen XXL field-token adapter until an independent OOD
gate is passed.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


PROCESS_ACTIONS = ("abstain", "safe_candidate", "retry_alternate")
PROCESS_ACTION_INDEX = {name: index for index, name in enumerate(PROCESS_ACTIONS)}
PROCESS_FEATURE_DIM = 24


def process_feature_vector(row: Mapping[str, Any]) -> list[float]:
    values = [0.0] * PROCESS_FEATURE_DIM
    values[0] = float(bool(row.get("typed_available")))
    values[1] = float(str(row.get("backend_state", "")) == "backend_response_observed")
    values[2] = float(str(row.get("backend_state", "")) == "database_unavailable")
    values[3] = float(str(row.get("method", "GET")).upper() == "GET")
    values[4] = float(str(row.get("method", "GET")).upper() == "POST")
    values[5] = min(max(int(row.get("field_count", 0) or 0), 0), 16) / 16.0
    values[6] = float(bool(row.get("binding_valid", True)))
    values[7] = float(str(row.get("status_class", "")) == "2xx")
    values[8] = float(str(row.get("status_class", "")) == "3xx")
    values[9] = float(str(row.get("status_class", "")) == "4xx")
    values[10] = float(str(row.get("status_class", "")) == "5xx")
    values[11] = min(max(int(row.get("history_len", 0) or 0), 0), 8) / 8.0
    feedback = str(row.get("previous_feedback", "none"))
    values[12] = float(feedback == "none")
    values[13] = float(feedback == "dead_end")
    values[14] = float(feedback == "candidate")
    values[15] = float(feedback == "environment_failure")
    values[16] = float(feedback == "rejected")
    values[17] = float(bool(row.get("negative_control", True)))
    values[18] = float(bool(row.get("redirect_present")))
    values[19] = float(bool(row.get("status_changed")))
    values[20] = float(str(row.get("surface_family", "")) == "xss")
    values[21] = float(str(row.get("surface_family", "")) == "injection")
    values[22] = float(str(row.get("surface_family", "")) == "logic")
    values[23] = float(bool(row.get("counterfactual")))
    return values


class HistoryProcessPolicy(nn.Module):
    def __init__(self, feature_dim: int = PROCESS_FEATURE_DIM, hidden_dim: int = 48) -> None:
        super().__init__()
        if feature_dim != PROCESS_FEATURE_DIM:
            raise ValueError(f"PG-213 expects feature_dim={PROCESS_FEATURE_DIM}")
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(PROCESS_ACTIONS)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def _metrics(logits: torch.Tensor, labels: torch.Tensor, rows: list[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    predicted = logits.argmax(-1)
    abstain = labels == PROCESS_ACTION_INDEX["abstain"]
    retry = labels == PROCESS_ACTION_INDEX["retry_alternate"]
    # A process action is unsafe only when it would bypass a hard transport
    # gate (untyped backend, database-unavailable state, or stale binding).
    # Choosing retry/stop after a typed response is a behavioral error, not a
    # network safety violation.
    hard_veto = torch.zeros_like(labels, dtype=torch.bool)
    if rows is not None:
        hard_veto = torch.tensor([
            (not bool(row.get("typed_available")))
            or str(row.get("backend_state", "")) == "database_unavailable"
            or (not bool(row.get("binding_valid", True)))
            for row in rows
        ], dtype=torch.bool, device=labels.device)
    return {
        "count": int(labels.numel()),
        "accuracy": round(float((predicted == labels).float().mean().item()), 8),
        "abstain_recall": round(float(((predicted == PROCESS_ACTION_INDEX["abstain"]) & abstain).sum().item() / max(int(abstain.sum().item()), 1)), 8),
        "retry_alternate_recall": round(float(((predicted == PROCESS_ACTION_INDEX["retry_alternate"]) & retry).sum().item() / max(int(retry.sum().item()), 1)), 8),
        "unsafe_allow_count": int(((predicted != PROCESS_ACTION_INDEX["abstain"]) & hard_veto).sum().item()),
        "behavioral_abstain_error_count": int(((predicted != PROCESS_ACTION_INDEX["abstain"]) & abstain & ~hard_veto).sum().item()),
    }


def train_history_policy(
    model: HistoryProcessPolicy,
    train_rows: list[Mapping[str, Any]],
    holdout_rows: list[Mapping[str, Any]],
    *,
    epochs: int = 100,
    seed: int = 213,
) -> dict[str, Any]:
    if not train_rows or not holdout_rows:
        raise ValueError("PG-213 requires train and holdout rows")
    torch.manual_seed(int(seed))
    device = next(model.parameters()).device
    train_x = torch.tensor([process_feature_vector(row) for row in train_rows], dtype=torch.float32, device=device)
    train_y = torch.tensor([PROCESS_ACTION_INDEX[str(row["label"])] for row in train_rows], dtype=torch.long, device=device)
    holdout_x = torch.tensor([process_feature_vector(row) for row in holdout_rows], dtype=torch.float32, device=device)
    holdout_y = torch.tensor([PROCESS_ACTION_INDEX[str(row["label"])] for row in holdout_rows], dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        logits = model(train_x)
        loss = nn.functional.cross_entropy(logits, train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch in {1, int(epochs)} or epoch % 25 == 0:
            model.eval()
            with torch.inference_mode():
                holdout_logits = model(holdout_x)
            history.append({"epoch": epoch, "loss": round(float(loss.detach().cpu()), 8), "holdout": _metrics(holdout_logits, holdout_y, holdout_rows)})
    model.eval()
    with torch.inference_mode():
        train_logits = model(train_x)
        holdout_logits = model(holdout_x)
    return {"schema_version": "pg213-history-process-policy-v1", "train_rows": len(train_rows), "holdout_rows": len(holdout_rows), "epochs": int(epochs), "history": history, "train": _metrics(train_logits, train_y, train_rows), "holdout": _metrics(holdout_logits, holdout_y, holdout_rows)}


def predict_process_action(model: HistoryProcessPolicy, row: Mapping[str, Any]) -> dict[str, Any]:
    device = next(model.parameters()).device
    features = torch.tensor([process_feature_vector(row)], dtype=torch.float32, device=device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(features)[0], dim=-1)
    index = int(probabilities.argmax().item())
    return {"action": PROCESS_ACTIONS[index], "confidence": round(float(probabilities[index].cpu()), 6), "feature_dim": PROCESS_FEATURE_DIM}


__all__ = ["PROCESS_ACTIONS", "PROCESS_FEATURE_DIM", "HistoryProcessPolicy", "predict_process_action", "process_feature_vector", "train_history_policy"]
