"""PG-282 evaluator-only binding for an abstract probe plan.

PG-281 deliberately predicts only a small Rule-IR plan.  PG-282 is the
boundary between that plan and an authorized target-side evaluator: it may
bind method/channel/encoding to an already observed surface, but it never
constructs a literal payload and never treats a reflection or status change
as a vulnerability result.

The binding is fail-closed.  A ``confirmed_positive`` evaluator effect needs
all of: an available authorized remote target, a typed effect, a clean
matched negative, a fresh reset, reference agreement, a replay match, and a
verifiable evidence hash.  The result is still not a vulnerability claim;
that final decision remains outside this generic adapter.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping
from urllib.parse import urlsplit


SCHEMA_VERSION = "pg282-evaluator-only-binding-v1"
PLAN_CLASSES = frozenset({"sql", "xss", "redirect", "logic", "file", "other"})
PLAN_CHANNELS = frozenset({"query", "form", "unknown"})
PLAN_ENCODINGS = frozenset({"plain", "url_percent", "unknown"})
PLAN_ACTIONS = frozenset({"replay_confirmed", "abstain"})
METHODS = frozenset({"GET", "POST"})
EVIDENCE_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_LITERAL_KEYS = frozenset({
    "payload", "raw_payload", "payload_value", "probe_value", "request_body",
    "response_body", "raw_response", "query_value", "form_value", "literal",
})


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _bool(value: Any) -> bool:
    return value is True


def _contains_forbidden_literal(value: Any, *, key: str = "") -> bool:
    """Reject persisted literal values while allowing abstract ``probe_class``."""

    if key.casefold() in FORBIDDEN_LITERAL_KEYS:
        return True
    if isinstance(value, Mapping):
        return any(_contains_forbidden_literal(child, key=str(child_key)) for child_key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_literal(child, key=key) for child in value)
    return False


def validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the model output without accepting an evaluator label."""

    if _contains_forbidden_literal(plan):
        raise ValueError("PG-282 plan contains a literal payload/response field")
    required = ("probe_class", "channel", "encoding", "final_action", "safe_to_send", "oracle_required")
    missing = [name for name in required if name not in plan]
    if missing:
        raise ValueError(f"PG-282 plan is missing fields: {', '.join(missing)}")
    probe_class = str(plan["probe_class"])
    channel = str(plan["channel"])
    encoding = str(plan["encoding"])
    action = str(plan["final_action"])
    if probe_class not in PLAN_CLASSES or channel not in PLAN_CHANNELS or encoding not in PLAN_ENCODINGS or action not in PLAN_ACTIONS:
        raise ValueError("PG-282 plan contains an unsupported abstract value")
    safe_to_send = plan["safe_to_send"]
    if not isinstance(safe_to_send, bool):
        raise ValueError("PG-282 safe_to_send must be boolean")
    if not _bool(plan["oracle_required"]):
        raise ValueError("PG-282 requires an evaluator-only oracle")
    if action == "replay_confirmed" and not safe_to_send:
        raise ValueError("replay_confirmed requires safe_to_send=true")
    if action == "abstain" and safe_to_send:
        raise ValueError("abstain requires safe_to_send=false")
    return {
        "probe_class": probe_class,
        "channel": channel,
        "encoding": encoding,
        "final_action": action,
        "safe_to_send": safe_to_send,
        "oracle_required": True,
    }


def _validate_path(path: str) -> str:
    if not isinstance(path, str) or not path.startswith("/") or path.startswith("//") or len(path) > 2048:
        raise ValueError("PG-282 surface path must be origin-relative")
    parsed = urlsplit(path)
    if parsed.scheme or parsed.netloc or parsed.fragment or ".." in parsed.path.split("/"):
        raise ValueError("PG-282 surface path must not contain an origin or traversal")
    return path


def validate_surface(surface: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an observed route manifest, not arbitrary caller input."""

    if _contains_forbidden_literal(surface):
        raise ValueError("PG-282 surface contains a literal request/response value")
    surface_id = str(surface.get("surface_id", ""))
    if not surface_id or len(surface_id) > 128:
        raise ValueError("PG-282 surface_id is required and bounded")
    method = str(surface.get("method", "")).upper()
    if method not in METHODS:
        raise ValueError("PG-282 surface method must be GET or POST")
    channel = str(surface.get("channel", ""))
    expected_channel = "query" if method == "GET" else "form"
    if channel != expected_channel:
        raise ValueError("PG-282 surface channel/method mismatch")
    path = _validate_path(str(surface.get("path", "/")))
    try:
        field_count = int(surface.get("field_count", 0))
    except (TypeError, ValueError) as error:
        raise ValueError("PG-282 field_count must be an integer") from error
    if not 1 <= field_count <= 32:
        raise ValueError("PG-282 field_count must be between 1 and 32")
    authorization = str(surface.get("authorization", ""))
    if authorization != "operator_allowlisted_remote_docker":
        raise ValueError("PG-282 surface is not in the authorized remote Docker scope")
    return {
        "surface_id": surface_id,
        "path": path,
        "method": method,
        "channel": channel,
        "field_count": field_count,
        "authorization": authorization,
        "typed_evaluator": str(surface.get("typed_evaluator", "unknown")),
        "fresh_reset_contract": bool(surface.get("fresh_reset_contract", False)),
        "reference_contract": bool(surface.get("reference_contract", False)),
        "negative_contract": bool(surface.get("negative_contract", False)),
    }


def _evidence_check(evidence: Mapping[str, Any]) -> tuple[bool, str]:
    digest = str(evidence.get("evidence_sha256", ""))
    if not EVIDENCE_DIGEST_RE.fullmatch(digest):
        return False, "evidence_hash_missing_or_invalid"
    unsigned = {str(key): value for key, value in evidence.items() if str(key) != "evidence_sha256"}
    if sha256_json(unsigned) != digest:
        return False, "evidence_hash_mismatch"
    return True, "ok"


def bind_abstract_plan(
    plan: Mapping[str, Any],
    surface: Mapping[str, Any],
    *,
    remote_probe: Mapping[str, Any] | None = None,
    evaluator_evidence: Mapping[str, Any] | None = None,
    hard_negative: bool = False,
) -> dict[str, Any]:
    """Bind a plan to a route and return only an abstract wire/evidence view."""

    normalized_plan = validate_plan(plan)
    normalized_surface = validate_surface(surface)
    probe = dict(remote_probe or {})
    evidence = dict(evaluator_evidence or {})
    if _contains_forbidden_literal(evidence):
        raise ValueError("PG-282 evaluator evidence contains a literal payload/response field")
    hash_ok, hash_reason = _evidence_check(evidence) if evidence else (False, "evaluator_evidence_missing")
    checks = {
        "plan_valid": True,
        "surface_valid": True,
        "authorized_remote_available": probe.get("status") == "available",
        "typed_effect": _bool(evidence.get("typed_effect_confirmed")),
        "negative_control_clean": _bool(evidence.get("negative_control_clean")),
        "fresh_reset": _bool(evidence.get("fresh_reset_attested")) and normalized_surface["fresh_reset_contract"],
        "reference_agreement": _bool(evidence.get("reference_agreement")) and normalized_surface["reference_contract"],
        "replay_consistent": _bool(evidence.get("replay_consistent")),
        "evidence_hash": hash_ok,
        "non_destructive": evidence.get("non_destructive") is True,
    }
    reasons: list[str] = []
    if hard_negative:
        reasons.append("family_or_route_hard_negative")
    if not normalized_plan["safe_to_send"] or normalized_plan["final_action"] == "abstain":
        reasons.append("model_abstain")
    if not checks["authorized_remote_available"]:
        reasons.append("authorized_remote_docker_unavailable")
    for name, passed in checks.items():
        if not passed and name not in {"authorized_remote_available"}:
            reasons.append(name if name != "evidence_hash" else hash_reason)
    reasons = list(dict.fromkeys(reasons))
    all_confirmation_checks = all(checks.values()) and not hard_negative
    if all_confirmation_checks and normalized_plan["safe_to_send"] and normalized_plan["final_action"] == "replay_confirmed":
        status = "confirmed_positive"
        decision = "evaluator_confirmed"
    elif normalized_plan["safe_to_send"] and not hard_negative and checks["authorized_remote_available"]:
        status = "candidate_ready"
        decision = "await_typed_evaluator"
    elif normalized_plan["safe_to_send"] and not hard_negative:
        status = "await_evaluator"
        decision = "blocked_before_send"
    else:
        status = "abstain"
        decision = "do_not_send"
    wire_shape = {
        "method": normalized_surface["method"],
        "path": normalized_surface["path"],
        "channel": normalized_surface["channel"],
        "encoding": normalized_plan["encoding"],
        "field_count": normalized_surface["field_count"],
        "surface_id_sha256": sha256_json(normalized_surface["surface_id"]),
        "literal_values_present": False,
    }
    evidence_projection = {
        "typed_effect_confirmed": bool(evidence.get("typed_effect_confirmed", False)),
        "negative_control_clean": bool(evidence.get("negative_control_clean", False)),
        "fresh_reset_attested": bool(evidence.get("fresh_reset_attested", False)),
        "reference_agreement": bool(evidence.get("reference_agreement", False)),
        "replay_consistent": bool(evidence.get("replay_consistent", False)),
        "evidence_sha256": str(evidence.get("evidence_sha256", "")),
        "evidence_hash_valid": hash_ok,
        "non_destructive": evidence.get("non_destructive") is True,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "decision": decision,
        "plan": normalized_plan,
        "surface": normalized_surface,
        "wire_shape": wire_shape,
        "checks": checks,
        "reasons": reasons,
        "evaluator_evidence": evidence_projection,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "literal_payload_stored": False,
        "raw_response_stored": False,
        "hard_negative": bool(hard_negative),
    }
    result["binding_evidence_sha256"] = sha256_json({
        "status": status,
        "decision": decision,
        "plan": normalized_plan,
        "surface": wire_shape,
        "checks": checks,
        "reasons": reasons,
        "evaluator_evidence": evidence_projection,
    })
    return result


__all__ = [
    "SCHEMA_VERSION",
    "bind_abstract_plan",
    "canonical",
    "sha256_json",
    "validate_plan",
    "validate_surface",
]
