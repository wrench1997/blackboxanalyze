"""PG-205 field-aware adapter for request/response token learning."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from .pg196_failure_action_decoder import ACTION_NAMES, FEATURE_DIM, encode_features
from .pg201_multitask_decoder import ENCODING_NAMES, FAILURE_NAMES
from .pg203_token_aware_decoder import TOKEN_FEATURE_DIM, token_features_for_row
from .pg205_request_response_tokens import FIELD_TOKEN_DIM, field_tokens_for_row


FIELD_AWARE_SCHEMA = "pg205-field-aware-grounding-adapter-v1"


class FieldTokenGroundingDecoder(nn.Module):
    """Frozen language body plus legacy and request/response structural slots."""

    def __init__(self, frozen_base: nn.Module, d_model: int | None = None, hidden_dim: int = 96) -> None:
        super().__init__()
        self.frozen_base = frozen_base
        if d_model is None:
            body = getattr(getattr(frozen_base, "base", frozen_base), "body", None)
            embedding = getattr(body, "token_embedding", None)
            d_model = int(getattr(embedding, "embedding_dim", 1024))
        self.context_projection = nn.Sequential(nn.Linear(d_model, 32), nn.Tanh())
        self.feature_projection = nn.Sequential(nn.Linear(FEATURE_DIM, 32), nn.GELU())
        self.token_projection = nn.Sequential(nn.Linear(TOKEN_FEATURE_DIM, 32), nn.GELU())
        self.field_projection = nn.Sequential(nn.Linear(FIELD_TOKEN_DIM, 32), nn.GELU())
        self.shared = nn.Sequential(nn.Linear(128, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.action_head = nn.Linear(hidden_dim, len(ACTION_NAMES))
        self.encoding_head = nn.Linear(hidden_dim, len(ENCODING_NAMES))
        self.failure_head = nn.Linear(hidden_dim, len(FAILURE_NAMES))

    def forward(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        features: torch.Tensor,
        token_features: torch.Tensor,
        field_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            hidden = self.frozen_base.hidden(ids, mask)
        context = 0.05 * self.context_projection(hidden)
        structured = self.feature_projection(features)
        legacy = self.token_projection(token_features)
        fields = self.field_projection(field_tokens)
        shared = self.shared(torch.cat([context, structured, legacy, fields], dim=-1))
        return {
            "action": self.action_head(shared),
            "encoding": self.encoding_head(shared),
            "failure": self.failure_head(shared),
        }


def warm_start_from_pg203(model: FieldTokenGroundingDecoder, state: Mapping[str, Any]) -> dict[str, Any]:
    """Copy compatible PG-203 heads and zero-pad its old 96-wide fusion input."""

    current = model.state_dict()
    copied: list[str] = []
    for name in (
        "context_projection.0.weight", "context_projection.0.bias",
        "feature_projection.0.weight", "feature_projection.0.bias",
        "token_projection.0.weight", "token_projection.0.bias",
        "action_head.weight", "action_head.bias",
        "encoding_head.weight", "encoding_head.bias",
        "failure_head.weight", "failure_head.bias",
        "shared.0.bias", "shared.2.weight", "shared.2.bias",
    ):
        if name in state and tuple(state[name].shape) == tuple(current[name].shape):
            current[name].copy_(state[name])
            copied.append(name)
    old_weight = state.get("shared.0.weight")
    if isinstance(old_weight, torch.Tensor) and tuple(old_weight.shape[:1]) == tuple(current["shared.0.weight"].shape[:1]):
        current["shared.0.weight"].zero_()
        width = min(int(old_weight.shape[1]), int(current["shared.0.weight"].shape[1]))
        current["shared.0.weight"][:, :width].copy_(old_weight[:, :width])
        copied.append("shared.0.weight_padded")
    model.load_state_dict(current)
    return {"source": "pg203_token_aware_adapter", "copied_keys": copied, "field_projection_initialized": True}


def _tensor_rows(rows: list[dict[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.tensor([
        encode_features(**{key: row[key] for key in ("method", "redirect_hops", "status_class", "candidate_signal", "typed_available", "negative_control", "budget_remaining", "failure_kind")})
        for row in rows
    ], dtype=torch.float32, device=device)
    legacy_tokens = torch.tensor([token_features_for_row(row) for row in rows], dtype=torch.float32, device=device)
    field_tokens = torch.tensor([field_tokens_for_row(row) for row in rows], dtype=torch.float32, device=device)
    actions = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long, device=device)
    encodings = torch.tensor([int(row["encoding_label"]) for row in rows], dtype=torch.long, device=device)
    failures = torch.tensor([int(row["failure_label"]) for row in rows], dtype=torch.long, device=device)
    return features, legacy_tokens, field_tokens, actions, encodings, failures


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


def train_field_aware(
    model: FieldTokenGroundingDecoder,
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    ids: torch.Tensor,
    mask: torch.Tensor,
    *,
    epochs: int = 80,
) -> dict[str, Any]:
    if not train_rows or not holdout_rows:
        raise ValueError("PG-205 requires non-empty train and holdout rows")
    device = ids.device
    train_features, train_legacy, train_fields, train_actions, train_encodings, train_failures = _tensor_rows(train_rows, device)
    holdout_features, holdout_legacy, holdout_fields, holdout_actions, holdout_encodings, holdout_failures = _tensor_rows(holdout_rows, device)
    parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("frozen_base.")]
    optimizer = torch.optim.AdamW(parameters, lr=2e-3, weight_decay=0.01)
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        outputs = model(
            ids.expand(len(train_rows), -1), mask.expand(len(train_rows), -1),
            train_features, train_legacy, train_fields,
        )
        loss = (
            nn.functional.cross_entropy(outputs["action"], train_actions)
            + 0.5 * nn.functional.cross_entropy(outputs["encoding"], train_encodings)
            + 0.5 * nn.functional.cross_entropy(outputs["failure"], train_failures)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if epoch in {1, epochs} or epoch % 20 == 0:
            model.eval()
            with torch.inference_mode():
                holdout_outputs = model(
                    ids.expand(len(holdout_rows), -1), mask.expand(len(holdout_rows), -1),
                    holdout_features, holdout_legacy, holdout_fields,
                )
            history.append({"epoch": epoch, "loss": round(float(loss.detach().cpu()), 8), "holdout": _metrics(holdout_outputs, holdout_actions, holdout_encodings, holdout_failures)})
    model.eval()
    with torch.inference_mode():
        train_outputs = model(ids.expand(len(train_rows), -1), mask.expand(len(train_rows), -1), train_features, train_legacy, train_fields)
        holdout_outputs = model(ids.expand(len(holdout_rows), -1), mask.expand(len(holdout_rows), -1), holdout_features, holdout_legacy, holdout_fields)
    return {
        "schema_version": FIELD_AWARE_SCHEMA,
        "train_rows": len(train_rows),
        "holdout_rows": len(holdout_rows),
        "epochs": int(epochs),
        "history": history,
        "train": _metrics(train_outputs, train_actions, train_encodings, train_failures),
        "holdout": _metrics(holdout_outputs, holdout_actions, holdout_encodings, holdout_failures),
    }


def evaluate_field_aware(model: FieldTokenGroundingDecoder, rows: list[dict[str, Any]], ids: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    if not rows:
        raise ValueError("PG-205 replay rows must not be empty")
    device = ids.device
    features, legacy, fields, actions, encodings, failures = _tensor_rows(rows, device)
    model.eval()
    with torch.inference_mode():
        outputs = model(ids.expand(len(rows), -1), mask.expand(len(rows), -1), features, legacy, fields)
    return _metrics(outputs, actions, encodings, failures)


def predict_field_aware(
    model: FieldTokenGroundingDecoder,
    *,
    ids: torch.Tensor,
    mask: torch.Tensor,
    features: list[float],
    token_features: list[float],
    field_tokens: list[float],
) -> dict[str, Any]:
    if len(field_tokens) != FIELD_TOKEN_DIM:
        raise ValueError("PG-205 field token feature dimension mismatch")
    feature_tensor = torch.tensor([features], dtype=torch.float32, device=ids.device)
    token_tensor = torch.tensor([token_features], dtype=torch.float32, device=ids.device)
    field_tensor = torch.tensor([field_tokens], dtype=torch.float32, device=ids.device)
    with torch.inference_mode():
        outputs = model(ids, mask, feature_tensor, token_tensor, field_tensor)
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
    "FIELD_AWARE_SCHEMA",
    "FieldTokenGroundingDecoder",
    "evaluate_field_aware",
    "predict_field_aware",
    "train_field_aware",
    "warm_start_from_pg203",
]
