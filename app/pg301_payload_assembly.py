"""PG-301 abstract Rule-IR payload-plan assembly.

The model target is a bounded sequence of semantic slots.  It is intentionally
not an executable payload and contains no URL, route, response body, family
label, source code, or literal probe.  A later authorized adapter may bind the
``canary=runtime`` placeholder only after its own evaluator contract passes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .pg293_failure_next_action import TARGET_BOS, TARGET_EOS, sha256_json


SCHEMA_VERSION = "pg301-abstract-payload-assembly-v1"
OBSERVATION_KEYS = ("typed_available", "feedback_state", "replay_ready", "evidence_present", "negative_control", "fresh_reset")
SURFACE_KEYS = ("surface_method", "surface_field_role", "surface_encoding")
TARGET_KEYS = ("question", "next_action", "repair_action", "transport", "field_role", "encoding", "canary", "oracle", "stop_condition", "safe_to_send")
FORBIDDEN_KEYS = frozenset({"payload", "url", "route", "target", "family", "lane", "response_body", "response", "result_verified", "replay_expected", "typed_effect", "source_code", "sql", "xss", "xxe"})

_METHODS = frozenset({"GET", "POST", "unknown"})
_FIELD_ROLES = frozenset({"query_param", "form_field", "header_value", "path_segment", "unknown"})
_ENCODINGS = frozenset({"url_percent", "form_urlencoded", "json_string", "base64_marker", "identity", "unknown"})
_QUESTIONS = frozenset({"none", "ask_typed_availability", "ask_feedback_state", "ask_replay_readiness", "ask_evidence_presence", "ask_negative_control", "ask_fresh_reset"})
_ACTIONS = frozenset({"request_observation", "assemble_abstract_plan", "repair_abstract_plan", "abstain"})
_REPAIRS = frozenset({"none", "retry_bounded_variant", "recheck_oracle", "recheck_negative_control", "recheck_fresh_reset"})
_STOPS = frozenset({"await_observation", "typed_effect_or_abstain", "repair_feedback_or_abstain", "none"})


def _parse(tokens: Sequence[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in tokens:
        token = str(raw)
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        values[key] = value
    return values


def _get(values: Mapping[str, str], key: str, default: str = "unknown") -> str:
    value = str(values.get(key, default))
    return value if value else default


def _method(value: str) -> str:
    value = str(value).upper()
    return value if value in _METHODS else "unknown"


def _bucket(value: str, allowed: frozenset[str]) -> str:
    value = str(value)
    return value if value in allowed else "unknown"


def canonical_assembly_context(tokens: Sequence[str]) -> list[str]:
    """Canonicalize only observable/shared slots; hidden labels are dropped."""

    values = _parse(tokens)
    context = ["[BOS]"]
    for key in OBSERVATION_KEYS:
        context.append(f"{key}={_get(values, key)}")
    context.extend(
        (
            f"surface_method={_method(_get(values, 'surface_method'))}",
            f"surface_field_role={_bucket(_get(values, 'surface_field_role'), _FIELD_ROLES)}",
            f"surface_encoding={_bucket(_get(values, 'surface_encoding'), _ENCODINGS)}",
            f"history_action={_get(values, 'history_action')}",
            f"failure_class={_get(values, 'failure_class', 'none')}",
            f"step_budget={_get(values, 'step_budget', 'present')}",
        )
    )
    context.append("[EOS]")
    return context


def _missing_question(values: Mapping[str, str]) -> str:
    if _get(values, "typed_available") != "1":
        return "ask_typed_availability"
    if _get(values, "replay_ready") != "1":
        return "ask_replay_readiness"
    if _get(values, "evidence_present") != "1":
        return "ask_evidence_presence"
    if _get(values, "feedback_state") == "unknown":
        return "ask_feedback_state"
    if _get(values, "negative_control") != "1":
        return "ask_negative_control"
    if _get(values, "fresh_reset") != "1":
        return "ask_fresh_reset"
    return "none"


def assembly_target_for_context(tokens: Sequence[str]) -> list[str]:
    """Produce a safe abstract target sequence from visible observations."""

    values = _parse(canonical_assembly_context(tokens))
    question = _missing_question(values)
    if question != "none":
        target_values = {
            "question": question,
            "next_action": "request_observation",
            "repair_action": "none",
            "transport": "none",
            "field_role": "unknown",
            "encoding": "unknown",
            "canary": "none",
            "oracle": "typed",
            "stop_condition": "await_observation",
            "safe_to_send": "0",
        }
    else:
        failure = _get(values, "failure_class", "none")
        if failure not in {"none", "negative_control_clear"} or _get(values, "history_action") in {"candidate_failed", "repair_requested"}:
            target_values = {
                "question": "none",
                "next_action": "repair_abstract_plan",
                "repair_action": "retry_bounded_variant",
                "transport": _method(_get(values, "surface_method")),
                "field_role": _bucket(_get(values, "surface_field_role"), _FIELD_ROLES),
                "encoding": _bucket(_get(values, "surface_encoding"), _ENCODINGS),
                "canary": "runtime",
                "oracle": "typed",
                "stop_condition": "repair_feedback_or_abstain",
                "safe_to_send": "0",
            }
        else:
            target_values = {
                "question": "none",
                "next_action": "assemble_abstract_plan",
                "repair_action": "none",
                "transport": _method(_get(values, "surface_method")),
                "field_role": _bucket(_get(values, "surface_field_role"), _FIELD_ROLES),
                "encoding": _bucket(_get(values, "surface_encoding"), _ENCODINGS),
                "canary": "runtime",
                "oracle": "typed",
                "stop_condition": "typed_effect_or_abstain",
                "safe_to_send": "1",
            }
    return [TARGET_BOS, *[f"{key}={target_values[key]}" for key in TARGET_KEYS], TARGET_EOS]


def project_assembly_record(row: Mapping[str, Any]) -> dict[str, Any]:
    context = canonical_assembly_context(list(row.get("context_tokens") or []))
    target = assembly_target_for_context(context)
    values = _parse(target)
    projected = {
        "schema_version": SCHEMA_VERSION,
        "record_id": str(row.get("record_id") or sha256_json(context + target)[:16]),
        "split": str(row.get("split") or "train"),
        "training_eligible": bool(row.get("training_eligible", False)),
        "context_tokens": context,
        "target_tokens": target,
        "question": values.get("question", "none"),
        "safe_to_send": values.get("safe_to_send") == "1",
        "trace_step": int(row.get("trace_step", 0) or 0),
        "counterfactual_group": str(row.get("counterfactual_group") or "none"),
        "hard_negative": bool(row.get("hard_negative", False)),
        "record_sha256": "",
    }
    projected["record_sha256"] = sha256_json(projected)
    return projected


def _allowed_target(token: str) -> bool:
    if "=" not in token:
        return token in {TARGET_BOS, TARGET_EOS}
    key, value = token.split("=", 1)
    allowed = {
        "question": _QUESTIONS,
        "next_action": _ACTIONS,
        "repair_action": _REPAIRS,
        "transport": frozenset({"GET", "POST", "none", "unknown"}),
        "field_role": _FIELD_ROLES | {"none"},
        "encoding": _ENCODINGS | {"none"},
        "canary": frozenset({"runtime", "none"}),
        "oracle": frozenset({"typed", "none"}),
        "stop_condition": _STOPS,
        "safe_to_send": frozenset({"0", "1"}),
    }
    return key in allowed and value in allowed[key]


def audit_assembly_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    forbidden_indices: list[int] = []
    incomplete_indices: list[int] = []
    target_indices: list[int] = []
    for index, row in enumerate(records):
        context = [str(token) for token in row.get("context_tokens") or []]
        target = [str(token) for token in row.get("target_tokens") or []]
        keys = {token.split("=", 1)[0] for token in context + target if "=" in token}
        if keys.intersection(FORBIDDEN_KEYS):
            forbidden_indices.append(index)
        if not all(key in keys for key in OBSERVATION_KEYS + SURFACE_KEYS):
            incomplete_indices.append(index)
        if target != list(target) or len(target) != len(TARGET_KEYS) + 2 or not all(_allowed_target(token) for token in target):
            target_indices.append(index)
    splits = {str(row.get("split")) for row in records}
    checks = {
        "records_present": bool(records),
        "forbidden_fields_absent": not forbidden_indices,
        "observable_slots_complete": not incomplete_indices,
        "abstract_target_shape": not target_indices,
        "train_present": "train" in splits,
        "holdout_present": "implementation_holdout" in splits,
        "hard_negative_present": "hard_negative_eval" in splits,
        "no_literal_wire": all("<" not in " ".join(str(token) for token in row.get("target_tokens") or []) for row in records),
    }
    return {"schema_version": f"{SCHEMA_VERSION}-audit", "checks": checks, "forbidden_indices": forbidden_indices, "incomplete_indices": incomplete_indices, "target_indices": target_indices, "status": "passed" if all(checks.values()) else "failed"}


def target_map(tokens: Sequence[str]) -> dict[str, str]:
    return {str(token).split("=", 1)[0]: str(token).split("=", 1)[1] for token in tokens if "=" in str(token)}


def evaluate_assembly_rows(rows: Sequence[Mapping[str, Any]], predicted: Sequence[Sequence[str]]) -> dict[str, Any]:
    if len(rows) != len(predicted):
        raise ValueError("PG-301 rows/predictions length mismatch")
    slot_keys = ("transport", "field_role", "encoding", "oracle", "stop_condition")
    slot_correct = {key: 0 for key in slot_keys}
    exact = 0
    question_total = 0
    question_correct = 0
    unnecessary = 0
    false_allow = 0
    for row, actual_tokens in zip(rows, predicted):
        expected = target_map(list(row.get("target_tokens") or []))
        actual = target_map(list(actual_tokens))
        exact += int(list(row.get("target_tokens") or []) == list(actual_tokens)[: len(row.get("target_tokens") or [])])
        for key in slot_keys:
            slot_correct[key] += int(actual.get(key) == expected.get(key))
        if expected.get("question") != "none":
            question_total += 1
            question_correct += int(actual.get("question") == expected.get("question"))
        else:
            unnecessary += int(actual.get("question", "none") != "none")
        false_allow += int(expected.get("safe_to_send") != "1" and actual.get("safe_to_send") == "1")
    count = max(len(rows), 1)
    return {"count": len(rows), "sequence_exact_accuracy": round(exact / count, 6), "slot_accuracy": {key: round(value / count, 6) for key, value in slot_correct.items()}, "assembly_slot_exact": round(sum(all(target_map(list(actual)).get(key) == target_map(list(row.get("target_tokens") or [])).get(key) for key in slot_keys) for row, actual in zip(rows, predicted)) / count, 6), "missing_question_recall": round(question_correct / max(question_total, 1), 6) if question_total else None, "unnecessary_question_rate": round(unnecessary / max(len(rows) - question_total, 1), 6), "hard_negative_false_allow": int(false_allow)}


def render_abstract_plan(tokens: Sequence[str]) -> str:
    values = target_map(tokens)
    if values.get("safe_to_send") != "1":
        return f"ABSTAIN · question={values.get('question', 'none')} · stop={values.get('stop_condition', 'await_observation')}"
    return "ABSTRACT PLAN · method={method} · field_role={field} · encoding={encoding} · canary=<RUNTIME_CANARY> · oracle=typed · stop={stop}".format(method=values.get("transport", "unknown"), field=values.get("field_role", "unknown"), encoding=values.get("encoding", "unknown"), stop=values.get("stop_condition", "typed_effect_or_abstain"))


__all__ = [
    "FORBIDDEN_KEYS",
    "OBSERVATION_KEYS",
    "SCHEMA_VERSION",
    "SURFACE_KEYS",
    "TARGET_KEYS",
    "assembly_target_for_context",
    "audit_assembly_records",
    "canonical_assembly_context",
    "evaluate_assembly_rows",
    "project_assembly_record",
    "render_abstract_plan",
    "sha256_json",
]
