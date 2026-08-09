"""PG-219 result-aware process policy.

The policy consumes only bounded transport/result projections that are
available at the current step.  Evaluator outcomes are targets for the
training row, never hidden labels copied into the feature vector.  The large
language body is optional at unit-test time and frozen when supplied by the
runner; only the small result-aware adapter is optimized.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .pg196_failure_action_decoder import FAILURE_KINDS, FEATURE_DIM, encode_features


PG219_SCHEMA = "pg219-result-aware-process-policy-v1"
PROCESS_ACTIONS = ("abstain", "safe_candidate", "retry_alternate")
PROCESS_ACTION_INDEX = {name: index for index, name in enumerate(PROCESS_ACTIONS)}
OUTCOME_NAMES = ("no_effect", "typed_effect", "result_verified", "environment_failure")
OUTCOME_INDEX = {name: index for index, name in enumerate(OUTCOME_NAMES)}

# fresh/health/backend/negative/candidate/reference/result observations,
# feedback history, bounded history length and field-count shape.
RESULT_FEATURE_DIM = 24


def result_features_for_row(row: Mapping[str, Any]) -> list[float]:
    """Encode only bounded observations from the current process state."""

    values = [0.0] * RESULT_FEATURE_DIM
    values[0] = float(bool(row.get("fresh_reset_ok")))
    values[1] = float(bool(row.get("database_health_ok")))
    values[2] = float(bool(row.get("backend_observed")))
    values[3] = float(bool(row.get("negative_clean")))
    values[4] = float(bool(row.get("candidate_sql_error_shape")))
    values[5] = float(bool(row.get("reference_agreement")))
    values[6] = float(bool(row.get("candidate_result_present")))
    values[7] = float(bool(row.get("negative_result_absent")))
    values[8] = float(bool(row.get("typed_effect_observed")))
    values[9] = float(bool(row.get("result_fixture_verified")))
    values[10] = float(bool(row.get("candidate_sent")))
    values[11] = float(bool(row.get("reference_sent")))
    values[12] = float(bool(row.get("negative_sent")))
    feedback = str(row.get("previous_feedback", "none"))
    for index, name in enumerate(("none", "candidate_error", "no_effect", "result_verified", "reference_disagreement", "environment_failure"), start=13):
        if index >= RESULT_FEATURE_DIM - 3:
            break
        values[index] = float(feedback == name)
    try:
        history_len = max(int(row.get("history_len", 0) or 0), 0)
    except (TypeError, ValueError):
        history_len = 0
    values[19] = float(history_len >= 1)
    values[20] = float(history_len >= 2)
    try:
        field_count = max(int(row.get("field_count", 0) or 0), 0)
    except (TypeError, ValueError):
        field_count = 0
    values[21] = float(field_count == 1)
    values[22] = float(field_count == 2)
    values[23] = float(field_count >= 3)
    return values


def core_features_for_row(row: Mapping[str, Any]) -> list[float]:
    """Encode the pre-action transport state using the shared PG-196 slots."""

    failure_kind = str(row.get("failure_kind", "no_effect"))
    if failure_kind not in FAILURE_KINDS:
        failure_kind = "no_effect"
    return encode_features(
        method=str(row.get("method", "GET")),
        redirect_hops=int(row.get("redirect_hops", 0) or 0),
        status_class=str(row.get("status_class", "2xx")),
        candidate_signal=int(bool(row.get("candidate_signal"))),
        typed_available=int(bool(row.get("typed_available"))),
        negative_control=int(bool(row.get("negative_control"))),
        budget_remaining=int(row.get("budget_remaining", 1) or 0),
        failure_kind=failure_kind,
    )


def hard_gate(row: Mapping[str, Any]) -> bool:
    """Return whether a candidate action is allowed by non-neural evidence."""

    return bool(
        row.get("typed_available")
        and row.get("fresh_reset_ok")
        and row.get("database_health_ok")
        and row.get("backend_observed")
        and row.get("negative_clean")
        and row.get("binding_valid", True)
    )


def guarded_action(proposed: str, row: Mapping[str, Any]) -> str:
    """Apply the fail-closed transport gate to a model proposal."""

    proposed = str(proposed)
    if proposed == "safe_candidate" and not hard_gate(row):
        return "abstain"
    if proposed not in PROCESS_ACTION_INDEX:
        return "abstain"
    return proposed


class ResultAwareProcessPolicy(nn.Module):
    """Frozen large body plus a result/history adapter."""

    def __init__(self, frozen_base: nn.Module | None = None, *, d_model: int = 1024, hidden_dim: int = 96) -> None:
        super().__init__()
        self.frozen_base = frozen_base
        self.d_model = int(d_model)
        self._context_cache_key: tuple[tuple[int, ...], tuple[int, ...]] | None = None
        self._context_cache: torch.Tensor | None = None
        self.context_projection = nn.Sequential(nn.Linear(self.d_model, 32), nn.Tanh())
        self.core_projection = nn.Sequential(nn.Linear(FEATURE_DIM, 32), nn.GELU())
        self.result_projection = nn.Sequential(nn.Linear(RESULT_FEATURE_DIM, 32), nn.GELU())
        self.shared = nn.Sequential(nn.Linear(96, int(hidden_dim)), nn.GELU(), nn.LayerNorm(int(hidden_dim)))
        self.action_head = nn.Linear(int(hidden_dim), len(PROCESS_ACTIONS))
        self.outcome_head = nn.Linear(int(hidden_dim), len(OUTCOME_NAMES))

    def _context(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.frozen_base is None:
            return torch.zeros((ids.shape[0], self.d_model), dtype=torch.float32, device=ids.device)
        key = (tuple(int(value) for value in ids[0].detach().cpu().tolist()), tuple(int(value) for value in mask[0].detach().cpu().tolist()))
        if self._context_cache_key == key and self._context_cache is not None and self._context_cache.device == ids.device:
            return self._context_cache.expand(ids.shape[0], -1)
        with torch.no_grad():
            hidden = self.frozen_base.hidden(ids[:1], mask[:1]).detach()
        self._context_cache_key = key
        self._context_cache = hidden
        return hidden.expand(ids.shape[0], -1)

    def clear_context_cache(self) -> None:
        self._context_cache_key = None
        self._context_cache = None

    def forward(self, ids: torch.Tensor, mask: torch.Tensor, core: torch.Tensor, result: torch.Tensor) -> dict[str, torch.Tensor]:
        context = 0.05 * self.context_projection(self._context(ids, mask))
        structured = self.core_projection(core)
        feedback = self.result_projection(result)
        shared = self.shared(torch.cat([context, structured, feedback], dim=-1))
        return {"action": self.action_head(shared), "outcome": self.outcome_head(shared)}


def _tensor_rows(rows: Sequence[Mapping[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    core = torch.tensor([core_features_for_row(row) for row in rows], dtype=torch.float32, device=device)
    result = torch.tensor([result_features_for_row(row) for row in rows], dtype=torch.float32, device=device)
    actions = torch.tensor([PROCESS_ACTION_INDEX[str(row["label"])] for row in rows], dtype=torch.long, device=device)
    outcomes = torch.tensor([OUTCOME_INDEX[str(row["outcome_label"])] for row in rows], dtype=torch.long, device=device)
    return core, result, actions, outcomes


def _metrics(outputs: Mapping[str, torch.Tensor], actions: torch.Tensor, outcomes: torch.Tensor, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    predicted = outputs["action"].argmax(-1)
    outcome = outputs["outcome"].argmax(-1)
    candidate = actions == PROCESS_ACTION_INDEX["safe_candidate"]
    abstain = actions == PROCESS_ACTION_INDEX["abstain"]
    retry = actions == PROCESS_ACTION_INDEX["retry_alternate"]
    raw_unsafe = int(((predicted == PROCESS_ACTION_INDEX["safe_candidate"]) & ~torch.tensor([hard_gate(row) for row in rows], device=actions.device)).sum().item())
    gated = [guarded_action(PROCESS_ACTIONS[int(value)], row) for value, row in zip(predicted.detach().cpu().tolist(), rows)]
    gated_ids = torch.tensor([PROCESS_ACTION_INDEX[action] for action in gated], device=actions.device)
    return {
        "count": int(actions.numel()),
        "action_accuracy": round(float((predicted == actions).float().mean().item()), 8),
        "outcome_accuracy": round(float((outcome == outcomes).float().mean().item()), 8),
        "safe_candidate_recall": round(float(((predicted == PROCESS_ACTION_INDEX["safe_candidate"]) & candidate).sum().item() / max(int(candidate.sum().item()), 1)), 8),
        "abstain_recall": round(float(((predicted == PROCESS_ACTION_INDEX["abstain"]) & abstain).sum().item() / max(int(abstain.sum().item()), 1)), 8),
        "retry_alternate_recall": round(float(((predicted == PROCESS_ACTION_INDEX["retry_alternate"]) & retry).sum().item() / max(int(retry.sum().item()), 1)), 8),
        "raw_unsafe_allow_count": raw_unsafe,
        "gated_action_accuracy": round(float((gated_ids == actions).float().mean().item()), 8),
        "gated_unsafe_allow_count": int(((gated_ids == PROCESS_ACTION_INDEX["safe_candidate"]) & ~torch.tensor([hard_gate(row) for row in rows], device=actions.device)).sum().item()),
    }


def train_result_policy(
    model: ResultAwareProcessPolicy,
    train_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
    ids: torch.Tensor,
    mask: torch.Tensor,
    *,
    epochs: int = 80,
    learning_rate: float = 2e-3,
) -> dict[str, Any]:
    if not train_rows or not holdout_rows:
        raise ValueError("PG-219 requires non-empty train and holdout rows")
    device = ids.device
    train_core, train_result, train_actions, train_outcomes = _tensor_rows(train_rows, device)
    hold_core, hold_result, hold_actions, hold_outcomes = _tensor_rows(holdout_rows, device)
    parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("frozen_base.")]
    optimizer = torch.optim.AdamW(parameters, lr=float(learning_rate), weight_decay=0.01)
    action_counts = torch.bincount(train_actions, minlength=len(PROCESS_ACTIONS)).float().clamp_min(1.0)
    action_weights = (action_counts.sum() / action_counts).to(device)
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        outputs = model(ids.expand(len(train_rows), -1), mask.expand(len(train_rows), -1), train_core, train_result)
        loss = nn.functional.cross_entropy(outputs["action"], train_actions, weight=action_weights) + 0.5 * nn.functional.cross_entropy(outputs["outcome"], train_outcomes)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if epoch in {1, int(epochs)} or epoch % 20 == 0:
            model.eval()
            with torch.inference_mode():
                hold_outputs = model(ids.expand(len(holdout_rows), -1), mask.expand(len(holdout_rows), -1), hold_core, hold_result)
            history.append({"epoch": epoch, "loss": round(float(loss.detach().cpu()), 8), "holdout": _metrics(hold_outputs, hold_actions, hold_outcomes, holdout_rows)})
    model.eval()
    with torch.inference_mode():
        train_outputs = model(ids.expand(len(train_rows), -1), mask.expand(len(train_rows), -1), train_core, train_result)
        hold_outputs = model(ids.expand(len(holdout_rows), -1), mask.expand(len(holdout_rows), -1), hold_core, hold_result)
    return {
        "schema_version": PG219_SCHEMA,
        "train_rows": len(train_rows),
        "holdout_rows": len(holdout_rows),
        "epochs": int(epochs),
        "history": history,
        "train": _metrics(train_outputs, train_actions, train_outcomes, train_rows),
        "holdout": _metrics(hold_outputs, hold_actions, hold_outcomes, holdout_rows),
    }


def predict_result_policy(model: ResultAwareProcessPolicy, row: Mapping[str, Any], ids: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    device = ids.device
    core = torch.tensor([core_features_for_row(row)], dtype=torch.float32, device=device)
    result = torch.tensor([result_features_for_row(row)], dtype=torch.float32, device=device)
    model.eval()
    with torch.inference_mode():
        outputs = model(ids, mask, core, result)
        action_prob = torch.softmax(outputs["action"][0], dim=-1)
        outcome_prob = torch.softmax(outputs["outcome"][0], dim=-1)
    proposed_index = int(action_prob.argmax().item())
    proposed = PROCESS_ACTIONS[proposed_index]
    return {
        "proposed_action": proposed,
        "action": guarded_action(proposed, row),
        "action_confidence": round(float(action_prob[proposed_index].cpu()), 6),
        "outcome": OUTCOME_NAMES[int(outcome_prob.argmax().item())],
        "outcome_confidence": round(float(outcome_prob.max().cpu()), 6),
        "result_feature_dim": RESULT_FEATURE_DIM,
        "hard_gate": hard_gate(row),
    }


__all__ = [
    "OUTCOME_NAMES",
    "PG219_SCHEMA",
    "PROCESS_ACTIONS",
    "RESULT_FEATURE_DIM",
    "ResultAwareProcessPolicy",
    "core_features_for_row",
    "guarded_action",
    "hard_gate",
    "predict_result_policy",
    "result_features_for_row",
    "train_result_policy",
]
