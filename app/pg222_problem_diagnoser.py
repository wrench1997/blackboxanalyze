"""PG-222 problem diagnoser.

This module is a small, structured diagnostic head, not a payload generator.
It consumes only bounded observations from the request/response process and
predicts why the current episode is stuck.  The evaluator label is never
accepted as an input feature.  A deterministic guard is applied after the
neural prediction so that a model cannot turn an incomplete observation into
``confirmed_local_effect``.

The intended use is to make the agent notice its own process mistakes early:
missing binding, a malformed candidate, an unavailable oracle, a broken
environment, a disagreeing reference, or a result that does not match the
claimed effect.  Raw payload strings and raw response bodies are deliberately
outside this representation.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn


PG222_SCHEMA = "pg222-problem-diagnoser-v1"

DIAGNOSIS_NAMES = (
    "environment_failure",
    "binding_failure",
    "oracle_unavailable",
    "candidate_no_effect",
    "reference_disagreement",
    "result_mismatch",
    "model_decision_error",
    "confirmed_local_effect",
    "inconclusive",
)
DIAGNOSIS_INDEX = {name: index for index, name in enumerate(DIAGNOSIS_NAMES)}

NEXT_STEP_NAMES = (
    "inspect_environment",
    "inspect_binding",
    "recheck_oracle",
    "retry_candidate",
    "compare_reference",
    "abstain",
)
NEXT_STEP_INDEX = {name: index for index, name in enumerate(NEXT_STEP_NAMES)}

# The first 27 positions describe the current observation.  The remaining
# slots encode bounded feedback/history/evidence state.  No route token,
# payload token, response body, or evaluator target is used.
FEATURE_NAMES = (
    "fresh_reset_ok",
    "database_health_ok",
    "backend_observed",
    "transport_error",
    "container_restart_used",
    "status_5xx",
    "reset_completed",
    "method_get",
    "method_post",
    "field_count_0",
    "field_count_1",
    "field_count_2",
    "field_count_3_plus",
    "binding_valid",
    "candidate_sent",
    "reference_sent",
    "negative_sent",
    "oracle_available",
    "typed_effect_observed",
    "result_fixture_verified",
    "boolean_differential",
    "candidate_reference_agreement",
    "negative_clean",
    "candidate_result_present",
    "negative_result_absent",
    "candidate_sql_error_shape",
    "result_mismatch_observed",
    "model_claimed_positive",
    "model_abstained",
    "feedback_none",
    "feedback_no_effect",
    "feedback_environment_failure",
    "feedback_reference_disagreement",
    "feedback_result_verified",
    "history_seen",
    "history_deep",
    "source_hash_present",
    "evidence_hash_present",
)
FEATURE_DIM = len(FEATURE_NAMES)

# These names are intentionally rejected by the feature encoder.  This makes
# accidental label leakage fail loudly when a new dataset adapter is added.
FORBIDDEN_FEATURE_KEYS = {
    "label",
    "diagnosis",
    "diagnosis_label",
    "outcome_label",
    "typed_effect_target",
    "result_fixture_target",
    "oracle_labels_as_features",
    "raw_payload",
    "payload",
    "raw_response",
    "response_body",
}


def _flag(value: Any) -> float:
    return float(bool(value))


def _status_5xx(value: Any) -> float:
    return float(str(value or "").startswith("5"))


def _field_bucket(value: Any) -> tuple[float, float, float, float]:
    try:
        count = max(int(value or 0), 0)
    except (TypeError, ValueError):
        count = 0
    return (float(count == 0), float(count == 1), float(count == 2), float(count >= 3))


def diagnose_features(row: Mapping[str, Any]) -> list[float]:
    """Encode an observation without reading hidden evaluator targets."""

    forbidden = FORBIDDEN_FEATURE_KEYS.intersection(row.keys())
    if forbidden:
        raise ValueError(f"PG-222 feature row contains forbidden target/raw keys: {sorted(forbidden)}")

    values = [0.0] * FEATURE_DIM
    values[0] = _flag(row.get("fresh_reset_ok"))
    values[1] = _flag(row.get("database_health_ok"))
    values[2] = _flag(row.get("backend_observed"))
    values[3] = _flag(row.get("transport_error"))
    values[4] = _flag(row.get("container_restart_used"))
    values[5] = _status_5xx(row.get("status_class"))
    values[6] = _flag(row.get("reset_completed"))
    method = str(row.get("method", "GET")).upper()
    values[7] = float(method == "GET")
    values[8] = float(method == "POST")
    values[9:13] = _field_bucket(row.get("field_count"))
    values[13] = _flag(row.get("binding_valid"))
    values[14] = _flag(row.get("candidate_sent"))
    values[15] = _flag(row.get("reference_sent"))
    values[16] = _flag(row.get("negative_sent"))
    values[17] = _flag(row.get("oracle_available"))
    values[18] = _flag(row.get("typed_effect_observed"))
    values[19] = _flag(row.get("result_fixture_verified"))
    values[20] = _flag(row.get("boolean_differential"))
    values[21] = _flag(row.get("candidate_reference_agreement"))
    values[22] = _flag(row.get("negative_clean"))
    values[23] = _flag(row.get("candidate_result_present"))
    values[24] = _flag(row.get("negative_result_absent"))
    values[25] = _flag(row.get("candidate_sql_error_shape"))
    values[26] = _flag(row.get("result_mismatch_observed"))
    values[27] = _flag(row.get("model_claimed_positive"))
    values[28] = _flag(row.get("model_abstained"))
    feedback = str(row.get("previous_feedback", "none"))
    for offset, name in enumerate(("none", "no_effect", "environment_failure", "reference_disagreement", "result_verified"), start=29):
        values[offset] = float(feedback == name)
    try:
        history = max(int(row.get("history_len", 0) or 0), 0)
    except (TypeError, ValueError):
        history = 0
    values[34] = float(history >= 1)
    values[35] = float(history >= 2)
    values[36] = float(len(str(row.get("source_hash", "") or "")) == 64)
    values[37] = float(len(str(row.get("evidence_hash", "") or "")) == 64)
    return values


def hard_diagnostic_gate(row: Mapping[str, Any]) -> bool:
    """Whether evidence is strong enough to allow a local-effect diagnosis."""

    return bool(
        row.get("fresh_reset_ok")
        and row.get("database_health_ok")
        and row.get("backend_observed")
        and row.get("reset_completed", True)
        and not row.get("transport_error")
        and not row.get("container_restart_used")
        and row.get("binding_valid", True)
        and row.get("candidate_sent")
        and row.get("reference_sent")
        and row.get("negative_sent")
        and row.get("oracle_available")
        and row.get("negative_clean")
        and row.get("candidate_reference_agreement")
        and (row.get("typed_effect_observed") or row.get("result_fixture_verified") or row.get("boolean_differential"))
        and not row.get("result_mismatch_observed")
    )


def guarded_diagnosis(proposed: str, row: Mapping[str, Any]) -> str:
    """Fail closed when the neural head proposes a positive diagnosis."""

    proposed = str(proposed)
    if proposed not in DIAGNOSIS_INDEX:
        return "inconclusive"
    # A positive claim made without the complete evidence gate is a process
    # error of the agent itself.  Surface that explicitly before attributing
    # the symptom to the environment or the target.
    if row.get("model_claimed_positive") and not hard_diagnostic_gate(row):
        return "model_decision_error"
    if proposed == "confirmed_local_effect" and not hard_diagnostic_gate(row):
        return "inconclusive"
    if proposed == "confirmed_local_effect":
        return "confirmed_local_effect"
    if not row.get("fresh_reset_ok") or not row.get("database_health_ok") or row.get("transport_error") or row.get("container_restart_used"):
        return "environment_failure"
    if not row.get("binding_valid", True):
        return "binding_failure"
    if row.get("result_mismatch_observed"):
        return "result_mismatch"
    if row.get("candidate_reference_agreement") is False:
        return "reference_disagreement"
    if proposed == "inconclusive":
        return "inconclusive"
    if row.get("oracle_available") is False and not row.get("typed_effect_observed") and not row.get("result_fixture_verified") and not row.get("boolean_differential"):
        return "oracle_unavailable"
    if row.get("candidate_sent") and row.get("oracle_available") and not row.get("typed_effect_observed") and not row.get("result_fixture_verified") and not row.get("boolean_differential") and row.get("negative_clean"):
        return "candidate_no_effect"
    return proposed


def expected_next_step(diagnosis: str) -> str:
    return {
        "environment_failure": "inspect_environment",
        "binding_failure": "inspect_binding",
        "oracle_unavailable": "recheck_oracle",
        "candidate_no_effect": "retry_candidate",
        "reference_disagreement": "compare_reference",
        "result_mismatch": "recheck_oracle",
        "model_decision_error": "abstain",
        "confirmed_local_effect": "abstain",
        "inconclusive": "abstain",
    }.get(str(diagnosis), "abstain")


class ProblemDiagnoser(nn.Module):
    """Compact diagnostic head trained on structured process observations."""

    def __init__(self, *, hidden_dim: int = 64) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.body = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.diagnosis_head = nn.Linear(hidden_dim, len(DIAGNOSIS_NAMES))
        self.next_step_head = nn.Linear(hidden_dim, len(NEXT_STEP_NAMES))

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.body(features)
        return {"diagnosis": self.diagnosis_head(hidden), "next_step": self.next_step_head(hidden)}


def _targets(rows: Sequence[Mapping[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    diagnosis = torch.tensor([DIAGNOSIS_INDEX[str(row["diagnosis"])] for row in rows], dtype=torch.long, device=device)
    next_steps = torch.tensor([NEXT_STEP_INDEX[expected_next_step(str(row["diagnosis"]))] for row in rows], dtype=torch.long, device=device)
    return diagnosis, next_steps


def _feature_view(row: Mapping[str, Any]) -> dict[str, Any]:
    """Remove supervision/retention metadata before encoding a training row."""

    return {key: value for key, value in row.items() if key not in FORBIDDEN_FEATURE_KEYS}


def _metrics(outputs: Mapping[str, torch.Tensor], rows: Sequence[Mapping[str, Any]], device: torch.device) -> dict[str, Any]:
    actual, next_steps = _targets(rows, device)
    predicted = outputs["diagnosis"].argmax(-1)
    predicted_steps = outputs["next_step"].argmax(-1)
    guarded = torch.tensor(
        [DIAGNOSIS_INDEX[guarded_diagnosis(DIAGNOSIS_NAMES[int(value)], row)] for value, row in zip(predicted.detach().cpu().tolist(), rows)],
        dtype=torch.long,
        device=device,
    )
    positive = actual == DIAGNOSIS_INDEX["confirmed_local_effect"]
    predicted_positive = predicted == DIAGNOSIS_INDEX["confirmed_local_effect"]
    guarded_positive = guarded == DIAGNOSIS_INDEX["confirmed_local_effect"]
    confusion: dict[str, dict[str, int]] = {}
    for truth, pred in zip(actual.detach().cpu().tolist(), guarded.detach().cpu().tolist()):
        truth_name, pred_name = DIAGNOSIS_NAMES[int(truth)], DIAGNOSIS_NAMES[int(pred)]
        confusion.setdefault(truth_name, {})[pred_name] = confusion.setdefault(truth_name, {}).get(pred_name, 0) + 1
    return {
        "count": int(actual.numel()),
        "diagnosis_accuracy": round(float((predicted == actual).float().mean().item()), 8),
        "guarded_diagnosis_accuracy": round(float((guarded == actual).float().mean().item()), 8),
        "next_step_accuracy": round(float((predicted_steps == next_steps).float().mean().item()), 8),
        "confirmed_local_effect_recall": round(float(((predicted_positive & positive).sum().item()) / max(int(positive.sum().item()), 1)), 8),
        "guarded_confirmed_local_effect_recall": round(float(((guarded_positive & positive).sum().item()) / max(int(positive.sum().item()), 1)), 8),
        "raw_positive_false_accept_count": int((predicted_positive & ~positive).sum().item()),
        "guarded_positive_false_accept_count": int((guarded_positive & ~positive).sum().item()),
        "confusion": confusion,
    }


def train_problem_diagnoser(
    model: ProblemDiagnoser,
    train_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
    *,
    epochs: int = 120,
    learning_rate: float = 2e-3,
    device: torch.device | None = None,
) -> dict[str, Any]:
    if not train_rows or not holdout_rows:
        raise ValueError("PG-222 requires non-empty train and holdout rows")
    device = device or torch.device("cpu")
    model.to(device)
    train_features = torch.tensor([diagnose_features(_feature_view(row)) for row in train_rows], dtype=torch.float32, device=device)
    hold_features = torch.tensor([diagnose_features(_feature_view(row)) for row in holdout_rows], dtype=torch.float32, device=device)
    train_targets, train_steps = _targets(train_rows, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=0.01)
    counts = torch.bincount(train_targets, minlength=len(DIAGNOSIS_NAMES)).float().clamp_min(1.0)
    weights = (counts.sum() / counts).to(device)
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        outputs = model(train_features)
        loss = nn.functional.cross_entropy(outputs["diagnosis"], train_targets, weight=weights) + 0.35 * nn.functional.cross_entropy(outputs["next_step"], train_steps)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch in {1, int(epochs)} or epoch % 30 == 0:
            model.eval()
            with torch.inference_mode():
                hold_outputs = model(hold_features)
            history.append({"epoch": epoch, "loss": round(float(loss.detach().cpu()), 8), "holdout": _metrics(hold_outputs, holdout_rows, device)})
    model.eval()
    with torch.inference_mode():
        train_outputs = model(train_features)
        hold_outputs = model(hold_features)
    return {
        "schema_version": PG222_SCHEMA,
        "feature_dim": FEATURE_DIM,
        "feature_names": list(FEATURE_NAMES),
        "diagnosis_names": list(DIAGNOSIS_NAMES),
        "next_step_names": list(NEXT_STEP_NAMES),
        "train_rows": len(train_rows),
        "holdout_rows": len(holdout_rows),
        "epochs": int(epochs),
        "history": history,
        "train": _metrics(train_outputs, train_rows, device),
        "holdout": _metrics(hold_outputs, holdout_rows, device),
    }


def predict_diagnosis(model: ProblemDiagnoser, row: Mapping[str, Any], *, device: torch.device | None = None) -> dict[str, Any]:
    device = device or torch.device("cpu")
    features = torch.tensor([diagnose_features(row)], dtype=torch.float32, device=device)
    model.eval()
    with torch.inference_mode():
        outputs = model(features)
        probabilities = torch.softmax(outputs["diagnosis"][0], dim=-1)
        step_probabilities = torch.softmax(outputs["next_step"][0], dim=-1)
    index = int(probabilities.argmax().item())
    step_index = int(step_probabilities.argmax().item())
    proposed = DIAGNOSIS_NAMES[index]
    guarded = guarded_diagnosis(proposed, row)
    return {
        "proposed_diagnosis": proposed,
        "diagnosis": guarded,
        "confidence": round(float(probabilities[index].cpu()), 6),
        "next_step": NEXT_STEP_NAMES[step_index],
        "next_step_confidence": round(float(step_probabilities[step_index].cpu()), 6),
        "hard_diagnostic_gate": hard_diagnostic_gate(row),
        "feature_dim": FEATURE_DIM,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
    }


__all__ = [
    "DIAGNOSIS_NAMES",
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "NEXT_STEP_NAMES",
    "PG222_SCHEMA",
    "ProblemDiagnoser",
    "diagnose_features",
    "expected_next_step",
    "guarded_diagnosis",
    "hard_diagnostic_gate",
    "predict_diagnosis",
    "train_problem_diagnoser",
]
