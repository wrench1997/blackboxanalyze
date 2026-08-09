"""Batch-level promotion gate for PG-286 live observation records.

The collector is intentionally conservative: a single confirmed evaluator
record is useful for collection, but it is not enough to train a payload-plan
decoder.  This module checks cross-reset/method/modality coverage and keeps
family labels outside the token context.  It returns a report only; it does not
mutate model weights or promote memory.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "pg286-live-batch-promotion-v1"
FORBIDDEN_CONTEXT = ("family=", "oracle=", "typed_effect", "positive", "payload=", "<script", "javascript:", "union select", "drop table")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _record_hash_ok(record: Mapping[str, Any]) -> bool:
    digest = str(record.get("record_sha256", ""))
    if len(digest) != 64:
        return False
    unsigned = {str(key): value for key, value in record.items() if str(key) != "record_sha256"}
    return sha256_json(unsigned) == digest


def _context_safe(record: Mapping[str, Any]) -> bool:
    tokens = [str(token) for token in list(record.get("context_tokens") or [])]
    return (
        "ir_family_agnostic=1" in tokens
        and not bool(record.get("raw_payload_stored"))
        and not bool(record.get("raw_response_body_stored"))
        and not any(any(bad.casefold() in token.casefold() for bad in FORBIDDEN_CONTEXT) for token in tokens)
    )


def audit_pg286_live_batch(
    records: Sequence[Mapping[str, Any]],
    *,
    hard_negative_records: Sequence[Mapping[str, Any]] = (),
    independent_audit_pass: bool = False,
    remote_docker_status: str = "unavailable",
    min_fresh_resets: int = 3,
    min_per_method: int = 3,
    min_typed_modalities: int = 2,
) -> dict[str, Any]:
    """Audit a candidate batch and return a fail-closed promotion report."""

    rows = [dict(row) for row in records if isinstance(row, Mapping)]
    hard = [dict(row) for row in hard_negative_records if isinstance(row, Mapping)]
    eligible = [
        row
        for row in rows
        if row.get("decision") == "eligible_for_cross_seed_review"
        and row.get("evaluator_status") == "confirmed_effect"
        and row.get("operator_reviewed") is True
        and row.get("hard_negative") is False
    ]
    reset_ids = {str(row.get("reset", {}).get("reset_id", "")) for row in eligible if row.get("reset", {}).get("reset_id")}
    methods = {str(row.get("surface", {}).get("method", "")).upper() for row in eligible}
    modalities = {str(row.get("typed_effect_type", "")) for row in eligible if row.get("typed_effect_type")}
    get_count = sum(str(row.get("surface", {}).get("method", "")).upper() == "GET" for row in eligible)
    post_count = sum(str(row.get("surface", {}).get("method", "")).upper() == "POST" for row in eligible)
    checks = {
        "record_integrity": all(_record_hash_ok(row) for row in [*rows, *hard]),
        "context_family_agnostic": all(_context_safe(row) for row in [*rows, *hard]),
        "no_single_row_training": all(row.get("training_eligible") is False for row in [*rows, *hard]),
        "hard_negative_quarantined": all(row.get("hard_negative") is True and row.get("decision") != "eligible_for_cross_seed_review" and row.get("training_eligible") is False for row in hard),
        "remote_docker_available": remote_docker_status == "available",
        "independent_audit": bool(independent_audit_pass),
        "fresh_reset_coverage": len(reset_ids) >= int(min_fresh_resets),
        "get_coverage": get_count >= int(min_per_method),
        "post_coverage": post_count >= int(min_per_method),
        "typed_modality_coverage": len(modalities) >= int(min_typed_modalities),
        "hard_negative_present": bool(hard),
    }
    reasons = [name for name, passed in checks.items() if not passed]
    ready = all(checks.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready_for_remote_a800_training" if ready else "blocked",
        "record_count": len(rows),
        "eligible_record_count": len(eligible) if ready else 0,
        "hard_negative_count": len(hard),
        "fresh_reset_count": len(reset_ids),
        "get_count": get_count,
        "post_count": post_count,
        "typed_modality_count": len(modalities),
        "typed_modalities": sorted(modalities),
        "checks": checks,
        "blocking_reasons": reasons,
        "remote_docker_status": remote_docker_status,
        "independent_audit_pass": bool(independent_audit_pass),
        "training_eligible_rows": len(eligible) if ready else 0,
        "memory_promotion_allowed_rows": 0,
        "vulnerability_claim_allowed": False,
        "other_gpus_untouched": True,
        "interpretation": "只有跨 reset/method/typed modality 的 live record 批次通过，才允许在远程 A800 GPU0 做后续训练；当前缺少真实 Docker 时保持 blocked。",
    }
    result["audit_sha256"] = sha256_json(result)
    return result


__all__ = ["SCHEMA_VERSION", "audit_pg286_live_batch", "sha256_json"]
