"""PG-284 typed evaluator contract for authorized GET/POST replay.

This module is the target-side acceptance boundary for the next-action head.
It consumes only bounded response projections and an evaluator attestation;
it never accepts or stores a literal payload, response body, credential or
arbitrary URL.  A positive here means ``typed_effect_confirmed`` for the
declared local surface, not a general vulnerability claim.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from typing import Any, Mapping
from urllib.parse import urlsplit


SCHEMA_VERSION = "pg284-typed-evaluator-contract-v1"
EFFECT_TYPES = frozenset({"dom_effect", "sql_ast_shape", "redirect_hop", "logic_transition", "result_shape"})
METHODS = frozenset({"GET", "POST"})
RAW_KEYS = frozenset({
    "payload", "raw_payload", "payload_value", "probe_value", "request_body",
    "response_body", "raw_response", "body_text", "html", "query_value",
    "form_value", "cookie", "authorization_header", "credential",
})
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9_.:/-]{1,160}$")


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _contains_raw(value: Any, key: str = "") -> bool:
    if key.casefold() in RAW_KEYS:
        return True
    if isinstance(value, Mapping):
        return any(_contains_raw(child, str(child_key)) for child_key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_raw(child, key) for child in value)
    return False


def _bounded_hash(value: Any, name: str) -> str:
    text = str(value or "")
    if not HASH_RE.fullmatch(text):
        raise ValueError(f"PG-284 {name} must be a SHA-256 digest")
    return text


def validate_surface(surface: Mapping[str, Any]) -> dict[str, Any]:
    if _contains_raw(surface):
        raise ValueError("PG-284 surface contains literal request/response material")
    surface_id = str(surface.get("surface_id", ""))
    if not ID_RE.fullmatch(surface_id):
        raise ValueError("PG-284 surface_id is invalid")
    method = str(surface.get("method", "")).upper()
    if method not in METHODS:
        raise ValueError("PG-284 surface method must be GET or POST")
    path = str(surface.get("path", ""))
    parsed = urlsplit(path)
    if not path.startswith("/") or path.startswith("//") or parsed.scheme or parsed.netloc or parsed.fragment or ".." in parsed.path.split("/"):
        raise ValueError("PG-284 surface path must be origin-relative")
    channel = str(surface.get("channel", ""))
    if channel != ("query" if method == "GET" else "form"):
        raise ValueError("PG-284 surface channel/method mismatch")
    authorization = str(surface.get("authorization", ""))
    if authorization != "operator_allowlisted_remote_docker":
        raise ValueError("PG-284 surface is outside the remote Docker authorization scope")
    source_attestation = _bounded_hash(surface.get("source_attestation_sha256"), "source_attestation_sha256")
    return {
        "surface_id": surface_id,
        "method": method,
        "path": path,
        "channel": channel,
        "field_count": max(1, min(int(surface.get("field_count", 1)), 32)),
        "authorization": authorization,
        "source_attestation_sha256": source_attestation,
        "evaluator_kind": str(surface.get("evaluator_kind", "unknown")),
    }


def validate_projection(projection: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Accept only the bounded shape returned by a target evaluator."""

    if _contains_raw(projection):
        raise ValueError(f"PG-284 {name} contains raw request/response material")
    status_class = str(projection.get("status_class", ""))
    if status_class not in {"1xx", "2xx", "3xx", "4xx", "5xx", "unknown"}:
        raise ValueError(f"PG-284 {name}.status_class is invalid")
    shape_hash = _bounded_hash(projection.get("shape_sha256"), f"{name}.shape_sha256")
    return {
        "status_class": status_class,
        "shape_sha256": shape_hash,
        "redirect_hops": max(0, min(int(projection.get("redirect_hops", 0) or 0), 8)),
        "backend_observed": bool(projection.get("backend_observed", False)),
        "effect_marker": str(projection.get("effect_marker", "none"))[:80],
    }


def validate_reset(reset: Mapping[str, Any]) -> dict[str, Any]:
    if _contains_raw(reset):
        raise ValueError("PG-284 reset attestation contains raw material")
    reset_id = str(reset.get("reset_id", ""))
    if not ID_RE.fullmatch(reset_id):
        raise ValueError("PG-284 reset_id is invalid")
    try:
        volume_mount_count = int(reset.get("volume_mount_count", -1))
    except (TypeError, ValueError) as error:
        raise ValueError("PG-284 volume_mount_count is invalid") from error
    return {
        "reset_id": reset_id,
        "fresh_target": bool(reset.get("fresh_target", False)),
        "container_recreated": bool(reset.get("container_recreated", False)),
        "container_restart_used": bool(reset.get("container_restart_used", False)),
        "volume_mount_count": volume_mount_count,
        "database_health_gate": str(reset.get("database_health_gate", "unknown")),
        "state_change_allowed": bool(reset.get("state_change_allowed", False)),
    }


def _validate_typed_evidence(evidence: Mapping[str, Any]) -> tuple[dict[str, Any], bool, str]:
    if _contains_raw(evidence):
        raise ValueError("PG-284 typed evidence contains raw material")
    effect_type = str(evidence.get("effect_type", ""))
    if effect_type not in EFFECT_TYPES:
        raise ValueError("PG-284 effect_type is not allow-listed")
    unsigned = {str(key): value for key, value in evidence.items() if str(key) != "evidence_sha256"}
    digest = str(evidence.get("evidence_sha256", ""))
    digest_valid = bool(HASH_RE.fullmatch(digest)) and sha256_json(unsigned) == digest
    normalized = {
        "effect_type": effect_type,
        "typed_effect_confirmed": bool(evidence.get("typed_effect_confirmed", False)),
        "negative_control_clean": bool(evidence.get("negative_control_clean", False)),
        "reference_agreement": bool(evidence.get("reference_agreement", False)),
        "replay_consistent": bool(evidence.get("replay_consistent", False)),
        "non_destructive": bool(evidence.get("non_destructive", False)),
        "evaluator_id": str(evidence.get("evaluator_id", "unknown")),
        "evidence_sha256": digest,
        "evidence_hash_valid": digest_valid,
    }
    return normalized, digest_valid, "ok" if digest_valid else "evidence_hash_mismatch"


def evaluate_typed_replay(
    *,
    surface: Mapping[str, Any],
    reset: Mapping[str, Any],
    reference: Mapping[str, Any],
    negative: Mapping[str, Any],
    candidate: Mapping[str, Any],
    replay: Mapping[str, Any],
    typed_evidence: Mapping[str, Any],
    remote_probe: Mapping[str, Any],
    hard_negative: bool = False,
) -> dict[str, Any]:
    """Return a fail-closed evaluator result without constructing a request."""

    normalized_surface = validate_surface(surface)
    normalized_reset = validate_reset(reset)
    projections = {
        "reference": validate_projection(reference, "reference"),
        "negative": validate_projection(negative, "negative"),
        "candidate": validate_projection(candidate, "candidate"),
        "replay": validate_projection(replay, "replay"),
    }
    normalized_evidence, evidence_hash_valid, hash_reason = _validate_typed_evidence(typed_evidence)
    reset_ok = bool(
        normalized_reset["fresh_target"]
        and normalized_reset["container_recreated"]
        and not normalized_reset["container_restart_used"]
        and normalized_reset["volume_mount_count"] == 0
        and normalized_reset["state_change_allowed"] is False
        and normalized_reset["database_health_gate"] == "healthy"
    )
    candidate_replay_same = projections["candidate"]["shape_sha256"] == projections["replay"]["shape_sha256"]
    negative_clean = normalized_evidence["negative_control_clean"] and not bool(projections["negative"].get("backend_observed") and projections["negative"]["effect_marker"] != "none")
    reference_agreement = normalized_evidence["reference_agreement"]
    typed_effect = normalized_evidence["typed_effect_confirmed"]
    checks = {
        "remote_docker_available": remote_probe.get("status") == "available",
        "authorized_surface": normalized_surface["authorization"] == "operator_allowlisted_remote_docker",
        "fresh_reset": reset_ok,
        "negative_control_clean": negative_clean,
        "typed_effect": typed_effect,
        "reference_agreement": reference_agreement,
        "replay_consistent": normalized_evidence["replay_consistent"] and candidate_replay_same,
        "evidence_hash": evidence_hash_valid,
        "non_destructive": normalized_evidence["non_destructive"],
    }
    reasons: list[str] = []
    if hard_negative:
        reasons.append("hard_negative")
    for name, passed in checks.items():
        if not passed:
            reasons.append(hash_reason if name == "evidence_hash" else name)
    reasons = list(dict.fromkeys(reasons))
    confirmed = all(checks.values()) and not hard_negative
    if confirmed:
        status = "confirmed_effect"
        decision = "typed_replay_accept"
    elif checks["remote_docker_available"] and not hard_negative:
        status = "await_evaluator"
        decision = "do_not_promote"
    else:
        status = "blocked"
        decision = "do_not_send_or_promote"
    evidence_projection = {
        "surface_id": normalized_surface["surface_id"],
        "reset_id": normalized_reset["reset_id"],
        "effect_type": normalized_evidence["effect_type"],
        "checks": checks,
        "reference_shape_sha256": projections["reference"]["shape_sha256"],
        "negative_shape_sha256": projections["negative"]["shape_sha256"],
        "candidate_shape_sha256": projections["candidate"]["shape_sha256"],
        "replay_shape_sha256": projections["replay"]["shape_sha256"],
        "typed_evidence_sha256": normalized_evidence["evidence_sha256"],
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "decision": decision,
        "surface": normalized_surface,
        "reset": normalized_reset,
        "projections": projections,
        "typed_evidence": normalized_evidence,
        "checks": checks,
        "reasons": reasons,
        "typed_effect_confirmed": confirmed,
        "confirmed_positive": False,
        "vulnerability_claim_allowed": False,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "literal_payload_stored": False,
        "raw_response_stored": False,
        "hard_negative": bool(hard_negative),
        "evidence_projection_sha256": sha256_json(evidence_projection),
    }
    return result


__all__ = [
    "EFFECT_TYPES",
    "METHODS",
    "SCHEMA_VERSION",
    "evaluate_typed_replay",
    "sha256_json",
    "validate_projection",
    "validate_reset",
    "validate_surface",
]
