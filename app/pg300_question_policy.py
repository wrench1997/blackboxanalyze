"""PG-300 question-policy representation and audit helpers.

This module deliberately isolates the first causal decision in the loop:
which *observable* slot must be requested next.  It does not contain a
payload, response body, route, vulnerability family, or evaluator verdict.
The resulting records are still consumed by the decoder-only next-token
model; the separation simply makes identifiability measurable before action
and abstract transport assembly are added.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .pg293_failure_next_action import TARGET_BOS, TARGET_EOS, sha256_json


SCHEMA_VERSION = "pg300-question-policy-v1"
OBSERVATION_KEYS = ("typed_available", "feedback_state", "replay_ready", "evidence_present")
FORBIDDEN_KEYS = {
    "result_verified",
    "replay_expected",
    "typed_effect",
    "family",
    "lane",
    "route",
    "target",
    "url",
    "payload",
    "response_body",
}


def _parse(tokens: Sequence[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in tokens:
        token = str(raw)
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in OBSERVATION_KEYS or key.startswith("surface_") or key.startswith("distractor_"):
            values[key] = value
    return values


def _value(values: Mapping[str, str], key: str) -> str:
    value = str(values.get(key, "unknown"))
    return value if value else "unknown"


def canonical_question_context(tokens: Sequence[str]) -> list[str]:
    """Return a fixed-order, oracle-blind token sequence.

    The observation slots are kept as separate tokens rather than collapsed
    into a precomputed ``missing_mask``.  This prevents the experiment from
    hiding the compositional task in a lookup feature.  Surface tokens are
    bounded to coarse values and sorted so GET/POST or parameter ordering is
    not a shortcut.
    """

    values = _parse(tokens)
    context = ["[BOS]"]
    context.extend(f"{key}={_value(values, key)}" for key in OBSERVATION_KEYS)
    surface = sorted(
        token
        for key, value in values.items()
        if key.startswith("surface_")
        for token in (f"{key}={value}",)
    )
    context.extend(surface[:4])
    context.append("distractor=present")
    context.append("[EOS]")
    return context


def question_for_observation(tokens: Sequence[str]) -> str:
    """Reference policy for the identifiable question, never a final verdict."""

    values = _parse(tokens)
    typed = _value(values, "typed_available")
    if typed in {"unknown", "0"}:
        return "ask_typed_availability"
    if _value(values, "feedback_state") == "unknown":
        return "ask_feedback_state"
    if _value(values, "replay_ready") == "unknown":
        return "ask_replay_readiness"
    if _value(values, "evidence_present") == "unknown":
        return "ask_evidence_presence"
    return "none"


def question_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project a source trace into the question-only causal target."""

    context = canonical_question_context(list(row.get("context_tokens") or []))
    question = question_for_observation(context)
    target = [TARGET_BOS, f"question={question}", TARGET_EOS]
    projected = {
        "schema_version": SCHEMA_VERSION,
        "record_id": str(row.get("record_id") or sha256_json(context + target)[:16]),
        "split": str(row.get("split") or "train"),
        "training_eligible": bool(row.get("training_eligible", False)),
        "context_tokens": context,
        "target_tokens": target,
        "question": question,
        "safe_to_send": False,
        "source_group": str(row.get("source_group") or "synthetic_observation").split(":", 1)[0],
        "hard_negative": bool(row.get("hard_negative", False)),
        "record_sha256": "",
    }
    projected["record_sha256"] = sha256_json(projected)
    return projected


def audit_question_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit structural completeness and ensure no hidden answer leaks in."""

    forbidden = []
    missing_slots = []
    target_order = []
    for index, row in enumerate(records):
        context = [str(token) for token in row.get("context_tokens") or []]
        target = [str(token) for token in row.get("target_tokens") or []]
        token_keys = {token.split("=", 1)[0].lower() for token in context + target if "=" in token}
        if any(key.lower() in token_keys for key in FORBIDDEN_KEYS):
            forbidden.append(index)
        keys = {token.split("=", 1)[0] for token in context if "=" in token}
        if not all(key in keys for key in OBSERVATION_KEYS):
            missing_slots.append(index)
        target_order.append(bool(target and target[0] == TARGET_BOS and len(target) == 3 and target[1].startswith("question=")))
    splits = {str(row.get("split")) for row in records}
    checks = {
        "records_present": bool(records),
        "forbidden_answer_leaks_absent": not forbidden,
        "observation_slots_complete": not missing_slots,
        "question_target_shape": all(target_order),
        "train_present": "train" in splits,
        "holdout_present": "implementation_holdout" in splits,
        "hard_negative_present": "hard_negative_eval" in splits,
        "all_safe_disabled": all(row.get("safe_to_send") is False for row in records),
    }
    return {
        "schema_version": f"{SCHEMA_VERSION}-audit",
        "checks": checks,
        "forbidden_indices": forbidden,
        "missing_slot_indices": missing_slots,
        "status": "passed" if all(checks.values()) else "failed",
    }


__all__ = [
    "FORBIDDEN_KEYS",
    "OBSERVATION_KEYS",
    "SCHEMA_VERSION",
    "audit_question_records",
    "canonical_question_context",
    "question_for_observation",
    "question_record",
    "sha256_json",
]
