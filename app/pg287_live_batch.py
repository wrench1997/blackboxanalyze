"""Fail-closed batch gate for PG-287-live records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "pg287-live-identifiability-batch-v1"
FORBIDDEN_CONTEXT = ("family=", "oracle=", "typed_effect=", "positive=", "payload=", "literal=", "<script", "javascript:")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _hash_ok(record: Mapping[str, Any]) -> bool:
    digest = str(record.get("record_sha256", ""))
    if len(digest) != 64:
        return False
    unsigned = {str(key): value for key, value in record.items() if str(key) != "record_sha256"}
    return sha256_json(unsigned) == digest


def _context_safe(record: Mapping[str, Any]) -> bool:
    tokens = [str(token) for token in list(record.get("context_tokens") or [])]
    return (
        "ir_family_agnostic=1" in tokens
        and "oracle_label_in_context=0" in tokens
        and "literal_probe_in_context=0" in tokens
        and not bool(record.get("raw_payload_strings_stored"))
        and not bool(record.get("raw_response_bodies_stored"))
        and not any(any(bad.casefold() in token.casefold() for bad in FORBIDDEN_CONTEXT) for token in tokens)
    )


def audit_pg287_live_batch(
    records: Sequence[Mapping[str, Any]],
    *,
    hard_negative_records: Sequence[Mapping[str, Any]] = (),
    independent_audit_pass: bool = False,
    remote_docker_status: str = "unavailable",
    min_fresh_resets: int = 3,
    min_per_method: int = 3,
    min_resolved_family_rows: int = 3,
    min_typed_modalities: int = 2,
) -> dict[str, Any]:
    """Audit source-heldout live rows before allowing remote training.

    The external splitter must assign ``split=family_holdout`` without putting
    a family name in model context.  A family split containing only ambiguous
    rows is reported as a coverage failure rather than an accuracy score.
    """

    rows = [dict(row) for row in records if isinstance(row, Mapping)]
    hard = [dict(row) for row in hard_negative_records if isinstance(row, Mapping)]
    eligible = [
        row for row in rows
        if row.get("training_eligible") is True
        and row.get("variant") in {"ambiguous", "resolved"}
        and row.get("hard_negative") is False
        and row.get("quality", {}).get("fresh_reset") is True
        and row.get("quality", {}).get("source_authorized") is True
    ]
    reset_ids = {str(row.get("source_record_id", "")) + ":" + str(row.get("quality", {}).get("reset_id", row.get("record_id", ""))) for row in eligible}
    # PG-286 keeps the typed modality name in a bounded field, not in context.
    modalities = {str(row.get("quality", {}).get("typed_effect", "")) for row in eligible if row.get("quality", {}).get("typed_effect")}
    get_count = sum(str(row.get("method", "")).upper() == "GET" for row in eligible)
    post_count = sum(str(row.get("method", "")).upper() == "POST" for row in eligible)
    family_resolved = sum(row.get("split") == "family_holdout" and row.get("variant") == "resolved" for row in eligible)
    ambiguous = sum(row.get("variant") == "ambiguous" for row in eligible)
    checks = {
        "record_integrity": all(_hash_ok(row) for row in [*rows, *hard]),
        "context_family_agnostic": all(_context_safe(row) for row in [*rows, *hard]),
        "no_single_row_training": all(row.get("training_eligible") is False for row in hard),
        "hard_negative_quarantined": all(row.get("hard_negative") is True and row.get("training_eligible") is False for row in hard),
        "remote_docker_available": remote_docker_status == "available",
        "independent_audit": bool(independent_audit_pass),
        "fresh_reset_coverage": len(reset_ids) >= int(min_fresh_resets),
        "get_coverage": get_count >= int(min_per_method),
        "post_coverage": post_count >= int(min_per_method),
        "family_resolved_coverage": family_resolved >= int(min_resolved_family_rows),
        "typed_modality_coverage": len(modalities) >= int(min_typed_modalities),
        "hard_negative_present": bool(hard),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready_for_remote_a800_training" if all(checks.values()) else "blocked",
        "record_count": len(rows),
        "eligible_record_count": len(eligible) if all(checks.values()) else 0,
        "hard_negative_count": len(hard),
        "ambiguous_count": ambiguous,
        "family_resolved_count": family_resolved,
        "fresh_reset_count": len(reset_ids),
        "get_count": get_count,
        "post_count": post_count,
        "typed_modalities": sorted(modalities),
        "checks": checks,
        "blocking_reasons": [name for name, passed in checks.items() if not passed],
        "remote_docker_status": remote_docker_status,
        "independent_audit_pass": bool(independent_audit_pass),
        "training_eligible_rows": len(eligible) if all(checks.values()) else 0,
        "memory_promotion_allowed_rows": 0,
        "vulnerability_claim_allowed": False,
        "other_gpus_untouched": True,
        "interpretation": "只有真实 evaluator、fresh reset、GET/POST、resolved family coverage、typed modality、hard-negative 和独立审计同时通过，才允许远程 A800 训练；coverage 缺失不是 0% 能力分数。",
    }
    result["audit_sha256"] = sha256_json(result)
    return result


__all__ = ["SCHEMA_VERSION", "audit_pg287_live_batch", "sha256_json"]
