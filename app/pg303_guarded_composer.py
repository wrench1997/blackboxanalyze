"""PG-303 fail-closed composer for symbolic Rule-IR proposals.

This is an auditable safety compiler, not another classifier.  It consumes a
neural symbolic proposal and visible observation slots, forces a question when
the visible state is non-identifiable, and binds only bounded abstract surface
values.  It never creates a literal wire payload.
"""

from __future__ import annotations

from collections.abc import Sequence

from .pg293_failure_next_action import TARGET_BOS, TARGET_EOS
from .pg301_payload_assembly import TARGET_KEYS, assembly_target_for_context, canonical_assembly_context, target_map


SCHEMA_VERSION = "pg303-guarded-composer-v1"
_ACTIONS = frozenset({"request_observation", "assemble_abstract_plan", "repair_abstract_plan", "abstain"})
_REPAIRS = frozenset({"none", "retry_bounded_variant", "recheck_oracle", "recheck_negative_control", "recheck_fresh_reset"})


def _target(values: dict[str, str]) -> list[str]:
    return [TARGET_BOS, *[f"{key}={values.get(key, 'unknown')}" for key in TARGET_KEYS], TARGET_EOS]


def compose_guarded_plan(model_tokens: Sequence[str], context_tokens: Sequence[str]) -> list[str]:
    """Compile a symbolic proposal under the visible-slot identifiability guard."""

    model = target_map(model_tokens)
    reference = target_map(assembly_target_for_context(context_tokens))
    context = target_map(canonical_assembly_context(context_tokens))
    missing_question = reference.get("question", "none")
    if missing_question != "none":
        # Two hidden worlds can share this context, so no model proposal can
        # safely choose a send/repair plan.  Ask the missing slot explicitly.
        return _target({
            "question": missing_question,
            "next_action": "request_observation",
            "repair_action": "none",
            "transport": "none",
            "field_role": "unknown",
            "encoding": "unknown",
            "canary": "none",
            "oracle": "typed",
            "stop_condition": "await_observation",
            "safe_to_send": "0",
        })

    action = model.get("next_action") if model.get("next_action") in _ACTIONS else "abstain"
    repair = model.get("repair_action") if model.get("repair_action") in _REPAIRS else "none"
    model_safe = model.get("safe_to_send") == "1"
    reference_safe = reference.get("safe_to_send") == "1"
    safe = model_safe and reference_safe
    method = context.get("surface_method") if context.get("surface_method") in {"GET", "POST"} else "none"
    field = context.get("surface_field_role") if context.get("surface_field_role") in {"query_param", "form_field", "header_value", "path_segment"} else "unknown"
    encoding = context.get("surface_encoding") if context.get("surface_encoding") in {"url_percent", "form_urlencoded", "json_string", "base64_marker", "identity"} else "unknown"
    if not safe:
        # A neural abstain/repair remains safe; a safe bit cannot override the
        # visible reset/evidence/negative-control contract.
        safe_value = "0"
    else:
        safe_value = "1"
    if action == "abstain":
        return _target({
            "question": "none",
            "next_action": "abstain",
            "repair_action": "none",
            "transport": "none",
            "field_role": "unknown",
            "encoding": "unknown",
            "canary": "none",
            "oracle": "typed",
            "stop_condition": "await_observation",
            "safe_to_send": "0",
        })
    return _target({
        "question": "none",
        "next_action": action,
        "repair_action": repair if action == "repair_abstract_plan" else "none",
        "transport": method,
        "field_role": field,
        "encoding": encoding,
        "canary": "runtime" if action != "request_observation" else "none",
        "oracle": "typed",
        "stop_condition": "typed_effect_or_abstain" if action == "assemble_abstract_plan" and safe else ("repair_feedback_or_abstain" if action == "repair_abstract_plan" else "await_observation"),
        "safe_to_send": safe_value,
    })


__all__ = ["SCHEMA_VERSION", "compose_guarded_plan"]
