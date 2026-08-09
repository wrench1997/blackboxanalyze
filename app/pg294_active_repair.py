"""PG-294 oracle-blind typed-availability and active-repair projection.

This module deliberately separates *availability of a checking channel* from
the channel's verdict.  The model may see that a bounded evaluator is present,
that a replay is ready, or that an observable process failure occurred.  It
must never see ``typed_effect``, ``result_verified``, ``replay_expected`` or a
family/lane label as an input feature.

The records produced here are abstract Rule-IR action targets only.  They do
not contain raw payloads, response bodies, routes, or wire instructions.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .pg293_failure_next_action import (
    CONTEXT_BOS,
    CONTEXT_EOS,
    SAFE_FAILURES,
    TARGET_BOS,
    TARGET_EOS,
    NEXT_ACTIONS,
    REPAIR_ACTIONS,
    encode_batch,
    greedy_decode,
    sha256_json,
)


SCHEMA_VERSION = "pg294-active-repair-v1"

TYPED_FEEDBACK_STATES = (
    "unresolved",
    "transport_error",
    "observable_no_effect",
    "observable_progress",
    "unknown",
)
QUESTION_SLOTS = (
    "none",
    "ask_typed_availability",
    "ask_feedback_state",
    "ask_replay_readiness",
    "ask_evidence_presence",
)

_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.=:+/-]{1,80}$")
_ALLOWED_KEYS = frozenset(
    {
        "phase",
        "method",
        "status",
        "channel",
        "field_bucket",
        "history_bucket",
        "fresh_reset",
        "reset_completed",
        "source_attested",
        "reference_sent",
        "negative_sent",
        "candidate_sent",
        "candidate_error_shape",
        "backend_observed",
        "database_health",
        "binding_valid",
        "hard_gate",
        "transport_error",
        "result_mismatch",
        "model_abstained",
        "model_claimed_positive",
        "repair_attempted",
        "step_budget",
        "step",
        "candidate_present",
        "self_error",
        "failure_observed",
    }
)
_FORBIDDEN_KEYS = frozenset(
    {
        "family",
        "lane",
        "surface",
        "repair",
        "oracle",
        "replay_expected",
        "result_verified",
        "typed_effect",
        "feedback",
        "payload_grounded_eligible",
        "route",
    }
)
_FORBIDDEN_VALUES = (
    "typed_effect",
    "result_verified",
    "replay_expected",
    "oracle",
    "family",
    "lane",
    "payload",
    "response_body",
    "route",
)


def _safe_scalar(value: Any) -> str:
    value = str(value)
    if not _TOKEN_RE.fullmatch(value):
        return "unknown"
    lowered = value.casefold()
    if any(marker in lowered for marker in _FORBIDDEN_VALUES):
        return "unknown"
    return value


def _state_value(value: bool | str) -> str:
    if value is True:
        return "1"
    if value is False:
        return "0"
    return "unknown" if str(value) == "unknown" else "unknown"


def project_observable_context(tokens: Sequence[str]) -> list[str]:
    """Strip result/oracle shortcuts from a source token sequence."""

    if not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes)):
        raise ValueError("PG-294 context must be a token sequence")
    result: list[str] = [CONTEXT_BOS]
    for raw in tokens:
        token = str(raw)
        if token in {CONTEXT_BOS, CONTEXT_EOS, "[CTX_END]"}:
            continue
        if "=" not in token or not _TOKEN_RE.fullmatch(token):
            continue
        key, value = token.split("=", 1)
        if key in _FORBIDDEN_KEYS or key not in _ALLOWED_KEYS:
            continue
        value = _safe_scalar(value)
        if value == "unknown" and key not in {"status", "channel"}:
            # Unknown values carry no stable action information and are safer
            # as an omitted slot than as an accidental label.
            continue
        result.append(f"{key}={value}")
    result.append(CONTEXT_EOS)
    if len(result) > 80:
        result = result[:79] + [CONTEXT_EOS]
    return result


def _target(next_action: str, repair_action: str, question: str, safe_to_send: bool) -> list[str]:
    if next_action not in NEXT_ACTIONS:
        next_action = "abstain"
    if repair_action not in REPAIR_ACTIONS:
        repair_action = "none"
    if question not in QUESTION_SLOTS:
        question = "none"
    return [
        TARGET_BOS,
        f"next_action={next_action}",
        f"repair_action={repair_action}",
        f"question={question}",
        f"safe_to_send={int(bool(safe_to_send))}",
        TARGET_EOS,
    ]


def state_target(source_safe: bool, *, typed_available: bool | str, feedback_state: str, replay_ready: bool | str, evidence_present: bool | str) -> tuple[str, str, str, bool]:
    """Map availability/process feedback to a conservative abstract action."""

    if feedback_state not in TYPED_FEEDBACK_STATES:
        feedback_state = "unknown"
    if typed_available is not True:
        return "recheck_oracle", "recheck_oracle", "ask_typed_availability", False
    if replay_ready is not True:
        return "recheck_oracle", "recheck_oracle", "ask_replay_readiness", False
    if evidence_present is not True:
        return "recheck_oracle", "recheck_oracle", "ask_evidence_presence", False
    if feedback_state in {"unresolved", "unknown"}:
        return "recheck_oracle", "recheck_oracle", "ask_feedback_state", False
    if feedback_state == "transport_error":
        return "inspect_environment", "inspect_environment", "none", False
    if feedback_state == "observable_no_effect":
        return "repair_candidate", "retry_candidate", "none", False
    if feedback_state != "observable_progress":
        return "abstain", "none", "ask_feedback_state", False
    if source_safe:
        return "replay_confirmed", "none", "none", True
    return "abstain", "none", "none", False


def _record(base: Mapping[str, Any], *, split: str, state_id: str, typed_available: bool | str, feedback_state: str, replay_ready: bool | str, evidence_present: bool | str, source_safe: bool, hard_negative: bool = False) -> dict[str, Any]:
    core = project_observable_context(base.get("context_tokens") or base.get("tokens") or [])
    core = core[:-1] + [
        f"typed_available={_state_value(typed_available)}",
        f"feedback_state={feedback_state}",
        f"replay_ready={_state_value(replay_ready)}",
        f"evidence_present={_state_value(evidence_present)}",
        CONTEXT_EOS,
    ]
    next_action, repair_action, question, safe_to_send = state_target(
        source_safe,
        typed_available=typed_available,
        feedback_state=feedback_state,
        replay_ready=replay_ready,
        evidence_present=evidence_present,
    )
    target = _target(next_action, repair_action, question, safe_to_send)
    record = {
        "schema_version": SCHEMA_VERSION,
        "record_id": f"pg294:{split}:{state_id}:{sha256_json(core + target)[:16]}",
        "source_group": "abstract_state_projection",
        "split": split,
        "state_id": state_id,
        "context_tokens": core,
        "target_tokens": target,
        "next_action": next_action,
        "repair_action": repair_action,
        "question": question,
        "safe_to_send": bool(safe_to_send),
        "hard_negative": bool(hard_negative),
        "typed_availability_is_not_verdict": True,
        "oracle_label_in_context": False,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "route_identity_stored": False,
        "family_identity_stored": False,
        "training_eligible": not hard_negative,
        "memory_promotion_allowed": False,
        "source_evidence_hash": sha256_json({"state_id": state_id, "source_record": str(base.get("record_id", "")), "source_safe": bool(source_safe)}),
    }
    record["record_sha256"] = sha256_json(record)
    return record


def build_records(base_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Expand source traces into availability/feedback cells plus eval negatives."""

    records: list[dict[str, Any]] = []
    normal = [row for row in base_records if str(row.get("split")) != "hard_negative_eval"]
    states = (
        ("unavailable", False, "unresolved", False, False),
        ("available_unresolved", True, "unresolved", True, False),
        ("transport", True, "transport_error", True, True),
        ("observable", True, "observable_no_effect", True, True),
        ("progress", True, "observable_progress", True, True),
        ("missing_key", "unknown", "unknown", "unknown", "unknown"),
    )
    for base in normal:
        for state_id, typed_available, feedback_state, replay_ready, evidence_present in states:
            split = str(base.get("split") or "train")
            records.append(
                _record(
                    base,
                    split=split,
                    state_id=state_id,
                    typed_available=typed_available,
                    feedback_state=feedback_state,
                    replay_ready=replay_ready,
                    evidence_present=evidence_present,
                    source_safe=bool(base.get("safe_to_send", False)),
                )
            )
    # Same-context opposite targets are evaluation-only.  They expose whether
    # a decoder memorizes a positive state rather than composing evidence.
    for row in records:
        if not row["safe_to_send"]:
            continue
        if row["state_id"] != "progress":
            continue
        negative = dict(row)
        negative["record_id"] = f"pg294:hard-negative:{row['record_id']}"
        negative["split"] = "hard_negative_eval"
        negative["next_action"] = "abstain"
        negative["repair_action"] = "none"
        negative["safe_to_send"] = False
        negative["hard_negative"] = True
        negative["training_eligible"] = False
        negative["question"] = "none"
        negative["target_tokens"] = _target("abstain", "none", "none", False)
        negative["record_sha256"] = sha256_json(negative)
        records.append(negative)
    return records


def audit_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    forbidden = _FORBIDDEN_VALUES + ("feedback=", "replay_expected=", "result_verified=")
    for row in records:
        context = " ".join(str(token) for token in row.get("context_tokens", []))
        if any(marker in context.casefold() for marker in forbidden):
            failures.append(f"context_leak:{row.get('record_id')}")
        if row.get("oracle_label_in_context") is not False:
            failures.append(f"oracle_flag:{row.get('record_id')}")
        if row.get("raw_payload_strings_stored") is not False or row.get("raw_response_bodies_stored") is not False:
            failures.append(f"raw_retention:{row.get('record_id')}")
        if row.get("hard_negative") and row.get("training_eligible") is not False:
            failures.append(f"hard_negative_training:{row.get('record_id')}")
        if row.get("memory_promotion_allowed") is not False:
            failures.append(f"memory_promotion:{row.get('record_id')}")
    return {
        "status": "passed" if not failures else "failed",
        "record_count": len(records),
        "failure_count": len(failures),
        "failures": failures,
        "checks": {
            "oracle_blind_context": not any("context_leak:" in item for item in failures),
            "hard_negative_eval_only": all(not row.get("hard_negative") or row.get("training_eligible") is False for row in records),
            "raw_firewall": not any("raw_retention:" in item for item in failures),
            "memory_promotion_blocked": not any("memory_promotion:" in item for item in failures),
        },
    }


def evaluate_question_metrics(model: Any, records: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], device: Any) -> dict[str, Any]:
    """Measure proactive question selection without treating a final label as success."""

    if not records:
        return {"count": 0, "question_accuracy": None, "missing_question_recall": None, "unnecessary_question_rate": None}
    context, lengths, targets, _ = encode_batch(records, vocabulary, device)
    with torch.inference_mode():
        predicted_ids, _ = greedy_decode(model, context, lengths, vocabulary, max_length=targets.shape[1])
    reverse = {int(index): str(token) for token, index in vocabulary.items()}
    correct = 0
    expected_questions = []
    predicted_questions = []
    for index, row in enumerate(records):
        width = len(row.get("target_tokens") or []) - 1
        predicted = [reverse.get(int(item), "[UNK]") for item in predicted_ids[index, :width].detach().cpu().tolist()]
        expected = [str(item) for item in row.get("target_tokens", [])[1:]]
        expected_question = next((token.split("=", 1)[1] for token in expected if token.startswith("question=")), "none")
        predicted_question = next((token.split("=", 1)[1] for token in predicted if token.startswith("question=")), "none")
        expected_questions.append(expected_question)
        predicted_questions.append(predicted_question)
        correct += int(expected_question == predicted_question)
    missing_total = sum(int(value != "none") for value in expected_questions)
    missing_correct = sum(int(pred == expected and expected != "none") for pred, expected in zip(predicted_questions, expected_questions))
    unnecessary = sum(int(pred != "none" and expected == "none") for pred, expected in zip(predicted_questions, expected_questions))
    normal_total = sum(int(value == "none") for value in expected_questions)
    return {
        "count": len(records),
        "question_accuracy": round(correct / max(len(records), 1), 6),
        "missing_question_recall": round(missing_correct / max(missing_total, 1), 6) if missing_total else None,
        "unnecessary_question_rate": round(unnecessary / max(normal_total, 1), 6) if normal_total else None,
        "missing_count": missing_total,
    }


__all__ = [
    "SCHEMA_VERSION",
    "TYPED_FEEDBACK_STATES",
    "audit_records",
    "build_records",
    "project_observable_context",
    "state_target",
    "evaluate_question_metrics",
]
