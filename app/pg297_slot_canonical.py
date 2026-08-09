"""PG-297 canonical slot/key-value token projection for causal assembly."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .pg293_failure_next_action import CONTEXT_BOS, CONTEXT_EOS, sha256_json


SCHEMA_VERSION = "pg297-slot-canonical-v1"
REQUIRED_KEYS = ("method", "channel", "status", "field_bucket", "typed_available", "feedback_state", "replay_ready", "evidence_present")
KEY_ORDER = ("phase", "method", "channel", "status", "field_bucket", "history_bucket", "fresh_reset", "reset_completed", "source_attested", "reference_sent", "negative_sent", "candidate_sent", "candidate_error_shape", "backend_observed", "database_health", "binding_valid", "hard_gate", "transport_error", "result_mismatch", "model_abstained", "model_claimed_positive", "repair_attempted", "step_budget", "step", "candidate_present", "self_error", "typed_available", "feedback_state", "replay_ready", "evidence_present")
_ORDER = {key: index for index, key in enumerate(KEY_ORDER)}
_BOOL_KEYS = frozenset({"fresh_reset", "reset_completed", "source_attested", "reference_sent", "negative_sent", "candidate_sent", "candidate_error_shape", "backend_observed", "database_health", "binding_valid", "hard_gate", "transport_error", "result_mismatch", "model_abstained", "model_claimed_positive", "repair_attempted", "candidate_present", "self_error"})
_FORBIDDEN_KEYS = frozenset({"family", "lane", "surface", "repair", "oracle", "route", "replay_expected", "result_verified", "typed_effect", "feedback", "payload_grounded_eligible"})
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.=:+/-]{1,80}$")


def _bucket(value: str) -> str:
    if value in {"0", "1"}:
        return value
    if value.casefold() in {"unknown", "none", "missing"}:
        return "unknown"
    return "present"


def canonical_value(key: str, value: str) -> str:
    value = str(value)
    lowered = value.casefold()
    if any(term in lowered for term in ("typed_effect", "result_verified", "replay_expected", "oracle", "payload", "response_body", "family", "lane", "route")):
        return "unknown"
    if key == "method":
        return value.upper() if value.upper() in {"GET", "POST"} else "unknown"
    if key == "channel":
        return value if value in {"query", "form", "json", "unknown"} else "unknown"
    if key == "status":
        if value in {"2xx", "3xx", "4xx", "5xx", "unknown"}:
            return value
        if value.isdigit() and len(value) == 3:
            return f"{value[0]}xx"
        return "unknown"
    if key in _BOOL_KEYS:
        return value if value in {"0", "1"} else "unknown"
    if key in {"typed_available", "replay_ready", "evidence_present"}:
        return value if value in {"0", "1", "unknown"} else "unknown"
    if key == "feedback_state":
        return value if value in {"unresolved", "transport_error", "observable_no_effect", "observable_progress", "unknown"} else "unknown"
    if key in {"field_bucket", "history_bucket", "step_budget", "step"}:
        return _bucket(value)
    if key == "phase":
        return value if value in {"observe", "diagnose", "repair", "replay", "unknown"} else "unknown"
    return "present" if _TOKEN_RE.fullmatch(value) else "unknown"


def canonicalize_context(tokens: Sequence[str]) -> list[str]:
    values: dict[str, str] = {}
    for raw in tokens:
        token = str(raw)
        if "=" not in token or not _TOKEN_RE.fullmatch(token):
            continue
        key, value = token.split("=", 1)
        if key in _FORBIDDEN_KEYS or key not in _ORDER:
            continue
        values.setdefault(key, canonical_value(key, value))
    for key in REQUIRED_KEYS:
        values.setdefault(key, "unknown")
    ordered = [f"{key}={values[key]}" for key in sorted(values, key=lambda item: _ORDER[item])]
    return [CONTEXT_BOS, *ordered, CONTEXT_EOS]


def canonicalize_record(row: Mapping[str, Any]) -> dict[str, Any]:
    clone = dict(row)
    clone["schema_version"] = SCHEMA_VERSION
    clone["context_tokens"] = canonicalize_context(row.get("context_tokens") or row.get("tokens") or [])
    clone["source_group"] = "canonical_slot_projection"
    clone["record_id"] = f"pg297:{row.get('split', 'unknown')}:{sha256_json(clone['context_tokens'] + list(row.get('target_tokens') or []))[:16]}"
    clone["oracle_label_in_context"] = False
    clone["raw_payload_strings_stored"] = False
    clone["raw_response_bodies_stored"] = False
    clone["route_identity_stored"] = False
    clone["family_identity_stored"] = False
    clone["memory_promotion_allowed"] = False
    clone["record_sha256"] = sha256_json(clone)
    return clone


def audit_canonical_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    forbidden = ("typed_effect", "result_verified", "replay_expected", "family=", "lane=", "route=", "payload", "response_body")
    failures: list[str] = []
    for row in records:
        context = " ".join(str(token) for token in row.get("context_tokens", []))
        if any(term in context.casefold() for term in forbidden):
            failures.append(f"context_leak:{row.get('record_id')}")
        if row.get("hard_negative") and row.get("training_eligible") is not False:
            failures.append(f"hard_negative_training:{row.get('record_id')}")
    return {"status": "passed" if not failures else "failed", "record_count": len(records), "failures": failures, "checks": {"oracle_blind": not failures, "required_slots_present": all(all(any(str(token).startswith(key + "=") for token in row.get("context_tokens", [])) for key in REQUIRED_KEYS) for row in records)}}


__all__ = ["SCHEMA_VERSION", "REQUIRED_KEYS", "audit_canonical_records", "canonicalize_context", "canonicalize_record", "canonical_value"]
