"""PG-292-live orchestration boundary for an authorized typed evaluator.

This module joins three already-independent gates without turning their
metrics into a scanner:

* the learned key/value gate contributes only a bounded probability and
  threshold;
* the Rule-IR verifier accepts only an abstract target-token sequence; and
* PG-284 validates bounded GET/POST projections, reset evidence and a typed
  evaluator hash.

No request value, response body, credential, or literal payload is accepted
or constructed here.  A successful result means that an already performed
authorized replay has a consistent typed effect.  It is not a general
vulnerability claim and it is not permission to emit a wire request.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .pg284_evaluator_contract import evaluate_typed_replay
from .pg288_rule_ir_verifier import verify_plan_tokens


SCHEMA_VERSION = "pg292-live-typed-evaluator-v1"
MAX_CONTEXT_TOKENS = 192
MAX_TOKEN_LENGTH = 128
FORBIDDEN_CONTEXT_MARKERS = (
    "payload=",
    "raw_payload=",
    "probe_value=",
    "request_body=",
    "response_body=",
    "body_text=",
    "<script",
    "javascript:",
)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _contains_forbidden(value: Any, key: str = "") -> bool:
    lowered = key.casefold()
    if lowered in {
        "payload",
        "raw_payload",
        "payload_value",
        "probe_value",
        "request_body",
        "response_body",
        "raw_response",
        "body_text",
        "html",
        "query_value",
        "form_value",
        "credential",
    }:
        return True
    if isinstance(value, Mapping):
        return any(_contains_forbidden(child, str(child_key)) for child_key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden(child, key) for child in value)
    return False


def validate_context_tokens(tokens: Sequence[str]) -> list[str]:
    """Validate model-visible context without accepting raw material."""

    if not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes)):
        raise ValueError("PG-292-live context_tokens must be a sequence")
    values = [str(token) for token in tokens]
    if not values or len(values) > MAX_CONTEXT_TOKENS:
        raise ValueError("PG-292-live context_tokens length is outside the bound")
    for token in values:
        if not token or len(token) > MAX_TOKEN_LENGTH:
            raise ValueError("PG-292-live context token is empty or too long")
        lowered = token.casefold()
        if any(marker in lowered for marker in FORBIDDEN_CONTEXT_MARKERS):
            raise ValueError("PG-292-live context contains literal probe material")
    if "ir_family_agnostic=1" not in values:
        raise ValueError("PG-292-live context must declare family-agnostic Rule-IR")
    return values


def _bounded_probability(value: Any, name: str) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"PG-292-live {name} must be numeric") from error
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"PG-292-live {name} must be in [0, 1]")
    return probability


def evaluate_pg292_live(
    *,
    context_tokens: Sequence[str],
    plan_tokens: Sequence[str],
    gate_probability: float,
    gate_threshold: float,
    surface: Mapping[str, Any],
    reset: Mapping[str, Any],
    reference: Mapping[str, Any],
    negative: Mapping[str, Any],
    candidate: Mapping[str, Any],
    replay: Mapping[str, Any],
    typed_evidence: Mapping[str, Any],
    remote_probe: Mapping[str, Any],
    hard_negative: bool = False,
    operator_reviewed: bool = False,
    independent_audit_pass: bool = False,
    cross_seed_reviewed: bool = False,
) -> dict[str, Any]:
    """Join PG-292, PG-288 and PG-284 with fail-closed promotion semantics."""

    context = validate_context_tokens(context_tokens)
    if _contains_forbidden({"surface": surface, "reset": reset, "reference": reference, "negative": negative, "candidate": candidate, "replay": replay, "typed_evidence": typed_evidence}):
        raise ValueError("PG-292-live evaluator input contains raw request/response material")
    probability = _bounded_probability(gate_probability, "gate_probability")
    threshold = _bounded_probability(gate_threshold, "gate_threshold")
    remote = dict(remote_probe or {})

    # First evaluate the external typed evidence.  The Rule-IR verifier is
    # rerun with this result so a replay_confirmed token cannot self-approve.
    typed = evaluate_typed_replay(
        surface=surface,
        reset=reset,
        reference=reference,
        negative=negative,
        candidate=candidate,
        replay=replay,
        typed_evidence=typed_evidence,
        remote_probe=remote,
        hard_negative=hard_negative,
    )
    typed_confirmed = bool(typed.get("typed_effect_confirmed"))
    structure = verify_plan_tokens(plan_tokens, typed_oracle_confirmed=typed_confirmed)
    gate_allowed = probability >= threshold
    model_safe_bit = structure.get("fields", {}).get("safe_to_send") == "1"
    evaluator_ready = bool(
        gate_allowed
        and model_safe_bit
        and structure.get("eligible_for_send")
        and typed.get("status") == "confirmed_effect"
        and remote.get("status") == "available"
        and not hard_negative
    )

    promotion_ready = bool(
        evaluator_ready
        and operator_reviewed
        and independent_audit_pass
        and cross_seed_reviewed
    )
    reasons: list[str] = []
    if not gate_allowed:
        reasons.append("feature_gate_below_threshold")
    if not structure.get("valid_structure"):
        reasons.append("rule_ir_structure_invalid")
    if not structure.get("renderable"):
        reasons.append("rule_ir_not_renderable")
    if not typed_confirmed:
        reasons.append("typed_replay_not_confirmed")
    if remote.get("status") != "available":
        reasons.append("authorized_remote_docker_unavailable")
    if hard_negative:
        reasons.append("family_or_route_hard_negative")
    if evaluator_ready and not operator_reviewed:
        reasons.append("operator_review_required")
    if evaluator_ready and not independent_audit_pass:
        reasons.append("independent_audit_required")
    if evaluator_ready and not cross_seed_reviewed:
        reasons.append("cross_seed_review_required")
    reasons = list(dict.fromkeys(reasons))

    if promotion_ready:
        status = "typed_replay_candidate_for_training_review"
        decision = "hold_for_explicit_promotion"
    elif evaluator_ready:
        status = "typed_replay_confirmed"
        decision = "do_not_promote"
    else:
        status = "blocked"
        decision = "abstain"

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "decision": decision,
        "context_tokens": context,
        "gate": {
            "probability": probability,
            "threshold": threshold,
            "allowed": gate_allowed,
        },
        "rule_ir": structure,
        "typed_replay": typed,
        "checks": {
            "feature_gate": gate_allowed,
            "rule_ir_renderable": bool(structure.get("renderable")),
            "typed_effect": typed_confirmed,
            "remote_docker_available": remote.get("status") == "available",
            "hard_negative_rejected": not hard_negative,
            "operator_reviewed": bool(operator_reviewed),
            "independent_audit_pass": bool(independent_audit_pass),
            "cross_seed_reviewed": bool(cross_seed_reviewed),
        },
        "reasons": reasons,
        "wire_emission_allowed": False,
        "literal_payload_stored": False,
        "raw_response_stored": False,
        "confirmed_positive": False,
        "vulnerability_claim_allowed": False,
        "training_eligible": promotion_ready,
        "memory_promotion_allowed": False,
    }
    unsigned = dict(result)
    result["evidence_sha256"] = sha256_json(unsigned)
    return result


__all__ = ["SCHEMA_VERSION", "evaluate_pg292_live", "sha256_json", "validate_context_tokens"]
