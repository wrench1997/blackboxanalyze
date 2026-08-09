"""Multi-task grounding adapter for PG-201.

The 101M XXL body stays frozen while a shared adapter learns three coupled
tasks: next action, encoding/probe class, and abstract failure class.  This
keeps the large model in the loop without allowing route names, payload text,
or evaluator labels to leak into the policy features.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn

from .pg196_failure_action_decoder import ACTION_NAMES, FEATURE_DIM, encode_features


ENCODING_NAMES = ("identity", "dom_markup", "encoded_dom", "abstract_sql")
FAILURE_NAMES = ("no_effect", "status_changed", "redirect_shape", "validation_shape", "server_shape", "oracle_unknown")
TASK_SCHEMA = "pg201-multitask-grounding-adapter-v1"


class MultiTaskGroundingDecoder(nn.Module):
    def __init__(self, frozen_base: nn.Module, d_model: int = 1024, hidden_dim: int = 96) -> None:
        super().__init__()
        self.frozen_base = frozen_base
        self.context_projection = nn.Sequential(nn.Linear(d_model, 32), nn.Tanh())
        self.feature_projection = nn.Sequential(nn.Linear(FEATURE_DIM, 32), nn.GELU())
        self.shared = nn.Sequential(nn.Linear(64, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.action_head = nn.Linear(hidden_dim, len(ACTION_NAMES))
        self.encoding_head = nn.Linear(hidden_dim, len(ENCODING_NAMES))
        self.failure_head = nn.Linear(hidden_dim, len(FAILURE_NAMES))

    def forward(self, ids: torch.Tensor, mask: torch.Tensor, features: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            hidden = self.frozen_base.hidden(ids, mask)
        context = 0.05 * self.context_projection(hidden)
        structured = self.feature_projection(features)
        shared = self.shared(torch.cat([context, structured], dim=-1))
        return {
            "action": self.action_head(shared),
            "encoding": self.encoding_head(shared),
            "failure": self.failure_head(shared),
        }


def _tensor_rows(rows: list[dict[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.tensor([
        encode_features(**{key: row[key] for key in ("method", "redirect_hops", "status_class", "candidate_signal", "typed_available", "negative_control", "budget_remaining", "failure_kind")})
        for row in rows
    ], dtype=torch.float32, device=device)
    actions = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long, device=device)
    encodings = torch.tensor([int(row["encoding_label"]) for row in rows], dtype=torch.long, device=device)
    failures = torch.tensor([int(row["failure_label"]) for row in rows], dtype=torch.long, device=device)
    return features, actions, encodings, failures


def _metrics(outputs: dict[str, torch.Tensor], actions: torch.Tensor, encodings: torch.Tensor, failures: torch.Tensor) -> dict[str, Any]:
    action = outputs["action"].argmax(-1)
    encoding = outputs["encoding"].argmax(-1)
    failure = outputs["failure"].argmax(-1)
    candidate = actions == 2
    abstain = actions == 3
    return {
        "count": int(actions.numel()),
        "action_accuracy": round(float((action == actions).float().mean().item()), 8),
        "candidate_recall": round(float(((action == 2) & candidate).sum().item() / max(int(candidate.sum().item()), 1)), 8),
        "abstain_recall": round(float(((action == 3) & abstain).sum().item() / max(int(abstain.sum().item()), 1)), 8),
        "unsafe_allow_count": int(((action == 2) & ~candidate).sum().item()),
        "encoding_accuracy": round(float((encoding == encodings).float().mean().item()), 8),
        "failure_accuracy": round(float((failure == failures).float().mean().item()), 8),
    }


def train_multitask(
    model: MultiTaskGroundingDecoder,
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    ids: torch.Tensor,
    mask: torch.Tensor,
    *,
    epochs: int = 80,
) -> dict[str, Any]:
    if not train_rows or not holdout_rows:
        raise ValueError("PG-201 requires non-empty train and holdout rows")
    device = ids.device
    train_features, train_actions, train_encodings, train_failures = _tensor_rows(train_rows, device)
    holdout_features, holdout_actions, holdout_encodings, holdout_failures = _tensor_rows(holdout_rows, device)
    parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("frozen_base.")]
    optimizer = torch.optim.AdamW(parameters, lr=2e-3, weight_decay=0.01)
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        outputs = model(ids.expand(len(train_rows), -1), mask.expand(len(train_rows), -1), train_features)
        loss = (
            nn.functional.cross_entropy(outputs["action"], train_actions)
            + 0.4 * nn.functional.cross_entropy(outputs["encoding"], train_encodings)
            + 0.4 * nn.functional.cross_entropy(outputs["failure"], train_failures)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if epoch in {1, epochs} or epoch % 20 == 0:
            model.eval()
            with torch.inference_mode():
                outputs_holdout = model(ids.expand(len(holdout_rows), -1), mask.expand(len(holdout_rows), -1), holdout_features)
            history.append({"epoch": epoch, "loss": round(float(loss.detach().cpu()), 8), "holdout": _metrics(outputs_holdout, holdout_actions, holdout_encodings, holdout_failures)})
    model.eval()
    with torch.inference_mode():
        train_outputs = model(ids.expand(len(train_rows), -1), mask.expand(len(train_rows), -1), train_features)
        holdout_outputs = model(ids.expand(len(holdout_rows), -1), mask.expand(len(holdout_rows), -1), holdout_features)
    return {
        "schema_version": TASK_SCHEMA,
        "train_rows": len(train_rows),
        "holdout_rows": len(holdout_rows),
        "epochs": int(epochs),
        "history": history,
        "train": _metrics(train_outputs, train_actions, train_encodings, train_failures),
        "holdout": _metrics(holdout_outputs, holdout_actions, holdout_encodings, holdout_failures),
    }


def evaluate_multitask(model: MultiTaskGroundingDecoder, rows: list[dict[str, Any]], ids: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    if not rows:
        raise ValueError("PG-201 evaluation rows must not be empty")
    device = ids.device
    features, actions, encodings, failures = _tensor_rows(rows, device)
    model.eval()
    with torch.inference_mode():
        outputs = model(ids.expand(len(rows), -1), mask.expand(len(rows), -1), features)
    return _metrics(outputs, actions, encodings, failures)


def predict_multitask(model: MultiTaskGroundingDecoder, *, ids: torch.Tensor, mask: torch.Tensor, features: list[float]) -> dict[str, Any]:
    tensor = torch.tensor([features], dtype=torch.float32, device=ids.device)
    with torch.inference_mode():
        outputs = model(ids, mask, tensor)
        probabilities = {key: torch.softmax(value[0], dim=0) for key, value in outputs.items()}
    action_index = int(probabilities["action"].argmax().item())
    encoding_index = int(probabilities["encoding"].argmax().item())
    failure_index = int(probabilities["failure"].argmax().item())
    return {
        "action": ACTION_NAMES[action_index],
        "action_confidence": float(probabilities["action"][action_index].cpu()),
        "encoding": ENCODING_NAMES[encoding_index],
        "encoding_confidence": float(probabilities["encoding"][encoding_index].cpu()),
        "failure": FAILURE_NAMES[failure_index],
        "failure_confidence": float(probabilities["failure"][failure_index].cpu()),
    }


__all__ = [
    "ENCODING_NAMES", "FAILURE_NAMES", "MultiTaskGroundingDecoder", "TASK_SCHEMA",
    "evaluate_multitask", "predict_multitask", "train_multitask",
]
