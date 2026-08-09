"""Bind evaluator-known role/step state to abstract PG-343 context tokens.

The binding is intentionally separate from evaluator answers.  A role is
known before a probe is chosen (candidate/reference/negative/replay), and a
step is a process state (preflight/baseline/failure/repair/replay).  These
two bounded symbols may be visible to the model; raw payloads, response
bodies, route names, oracle answers, and evaluator literals may not.

This module never infers a role from target tokens.  Callers must provide an
explicit source/evaluator-side role-step attestation.  Missing or conflicting
attestation is a hard error for PG-343 data collection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

ALLOWED_ROLES = frozenset({"candidate", "reference", "negative", "replay"})
ALLOWED_STEPS = frozenset({"preflight", "baseline", "failure", "repair", "replay"})
ROLE_TOKEN = "belief_probe_role"
STEP_TOKEN = "belief_process_step"
FORBIDDEN = (
    "raw_",
    "payload",
    "response_body=",
    "response_body_text=",
    "family=",
    "route_literal=",
    "route_name=",
    "oracle=",
    "evaluator=",
    "typed_effect=",
    "expected_answer=",
)
TARGET_KEY_PREFIXES = (
    "probe_variant_ref=",
    "next_action=",
    "question=",
    "repair_action=",
    "safe_to_send=",
    "transport_ref=",
    "field_role_ref=",
    "encoding_ref=",
    "[target_",
)


def _symbol(value: Any, *, allowed: frozenset[str], field: str) -> str:
    text = str(value).strip().casefold().replace("-", "_")
    if text not in allowed:
        raise ValueError(f"PG-343 {field} is not allow-listed")
    return text


def _token_value(tokens: Sequence[Any], key: str) -> str | None:
    prefix = f"{key}="
    values = [str(token)[len(prefix) :] for token in tokens if str(token).startswith(prefix)]
    if not values:
        return None
    if len(set(values)) != 1:
        raise ValueError(f"PG-343 {key} has conflicting context tokens")
    return values[0]


def bind_context_tokens(context_tokens: Sequence[Any], *, role: Any, step: Any) -> list[str]:
    """Return context tokens with explicit role/step symbols.

    Existing matching bindings are idempotent.  Existing conflicting or
    forbidden tokens fail closed.  The function returns a new list and never
    mutates the caller's token sequence.
    """

    if not isinstance(context_tokens, Sequence) or isinstance(context_tokens, (str, bytes)):
        raise TypeError("PG-343 context_tokens must be a token sequence")
    normalized_role = _symbol(role, allowed=ALLOWED_ROLES, field="role")
    normalized_step = _symbol(step, allowed=ALLOWED_STEPS, field="step")
    result = [str(token) for token in context_tokens]
    for token in result:
        folded = token.casefold()
        if any(marker in folded for marker in FORBIDDEN) or any(folded.startswith(prefix) for prefix in TARGET_KEY_PREFIXES):
            raise ValueError("PG-343 context firewall rejected role-step context")
    existing_role = _token_value(result, ROLE_TOKEN)
    existing_step = _token_value(result, STEP_TOKEN)
    if existing_role is not None and existing_role != normalized_role:
        raise ValueError("PG-343 role binding conflicts with existing context")
    if existing_step is not None and existing_step != normalized_step:
        raise ValueError("PG-343 step binding conflicts with existing context")
    if existing_role is None:
        result.append(f"{ROLE_TOKEN}={normalized_role}")
    if existing_step is None:
        result.append(f"{STEP_TOKEN}={normalized_step}")
    return result


def bind_observation(observation: Mapping[str, Any], *, role: Any, step: Any) -> dict[str, Any]:
    """Add explicit role/step projections to a tokenizer observation.

    The values live under the already-declared belief/replay surface and are
    emitted only as bounded abstract tokens by the PG-331 tokenizer.  The
    input is deep-copied so evaluator-side state is not modified.
    """

    if not isinstance(observation, Mapping):
        raise TypeError("PG-343 observation must be a mapping")
    normalized_role = _symbol(role, allowed=ALLOWED_ROLES, field="role")
    normalized_step = _symbol(step, allowed=ALLOWED_STEPS, field="step")
    result = deepcopy(dict(observation))
    belief = result.get("belief_and_replay")
    if not isinstance(belief, Mapping):
        belief = {}
    else:
        belief = dict(belief)
    existing_role = belief.get("probe_role")
    existing_step = belief.get("process_step")
    if existing_role is not None and _symbol(existing_role, allowed=ALLOWED_ROLES, field="role") != normalized_role:
        raise ValueError("PG-343 observation role conflicts with attestation")
    if existing_step is not None and _symbol(existing_step, allowed=ALLOWED_STEPS, field="step") != normalized_step:
        raise ValueError("PG-343 observation step conflicts with attestation")
    belief["probe_role"] = normalized_role
    belief["process_step"] = normalized_step
    result["belief_and_replay"] = belief
    return result


def binding_present(context_tokens: Sequence[Any]) -> bool:
    """Return true only when exactly one role and one step token are present."""

    try:
        role = _token_value(context_tokens, ROLE_TOKEN)
        step = _token_value(context_tokens, STEP_TOKEN)
    except ValueError:
        return False
    return role in ALLOWED_ROLES and step in ALLOWED_STEPS
