"""PG-256 evaluator-only oracle for Pikachu's wide-byte SQL route.

PG-255 could see a syntax-shaped response but could not tell whether the
escape/connection boundary changed the selected rows.  This module projects a
bounded row count from the local read-only response and evaluates a matched
baseline, candidate, reference, and negative quartet.  It never retains the
response body or treats the result as a general vulnerability claim.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .maze_engine import sha256_json


PG256_SCHEMA = "pg256-pikachu-widebyte-result-oracle-v1"
ROUTE = "/vul/sqli/sqli_widebyte.php"
METHOD = "POST"
_ROW_MARKER = "your uid:"
_MAX_ROWS = 16


def _length_bucket(length: int) -> str:
    if length <= 0:
        return "0"
    if length <= 255:
        return "1-255"
    if length <= 4095:
        return "256-4095"
    if length <= 65535:
        return "4096-65535"
    return "65536+"


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def project_widebyte_response(response: Any, *, label: str) -> dict[str, Any]:
    """Project only bounded row/error/status evidence from an HTTP response."""

    body = str(getattr(response, "text", "") or "").casefold()
    content = bytes(getattr(response, "content", b"") or b"")
    status_code = int(getattr(response, "status_code", 0) or 0)
    row_count = min(body.count(_ROW_MARKER), _MAX_ROWS)
    error_shape = any(token in body for token in ("sql syntax", "warning:", "fatal error", "mysqli"))
    projection = {
        "status_code": status_code,
        "status_class": f"{status_code // 100}xx",
        "body_length_bucket": _length_bucket(len(content)),
        "row_marker": _ROW_MARKER,
        "row_count_capped": row_count,
        "result_shape": "record_present" if row_count else "record_absent",
        "sql_error_shape": bool(error_shape),
        "body_sha256": hashlib.sha256(content).hexdigest(),
        "label": str(label),
    }
    projection["projection_sha256"] = sha256_json(projection)
    return {
        "schema_version": PG256_SCHEMA,
        "response_projection": projection,
        "raw_response_retained": False,
    }


def _reset_ok(reset: Mapping[str, Any]) -> bool:
    return bool(
        reset.get("fresh_target")
        and reset.get("completed")
        and reset.get("container_recreated")
        and not reset.get("container_restart_used")
        and int(reset.get("volume_mount_count", -1)) == 0
        and reset.get("database_health_gate") == "mysqli_root_pikachu_ok"
        and reset.get("state_change_allowed") is False
        and reset.get("external_network") is False
    )


def evaluate_widebyte_effect(
    *,
    route: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    negative: Mapping[str, Any],
    reset: Mapping[str, Any],
    source_hash: str,
    candidate_class: str,
    reference_class: str,
) -> dict[str, Any]:
    """Apply the independent PG-256 wide-byte row-differential contract."""

    reasons: list[str] = []
    if str(route.get("path")) != ROUTE or str(route.get("method", "")).upper() != METHOD:
        reasons.append("route_contract_mismatch")
    if not _reset_ok(reset):
        reasons.append("fresh_database_reset_attestation_missing")
    source_hash = str(source_hash).casefold()
    if len(source_hash) != 64 or any(char not in "0123456789abcdef" for char in source_hash):
        reasons.append("source_hash_invalid")

    bp = dict(baseline.get("response_projection") or {})
    cp = dict(candidate.get("response_projection") or {})
    rp = dict(reference.get("response_projection") or {})
    np = dict(negative.get("response_projection") or {})
    baseline_rows = int(bp.get("row_count_capped", 0) or 0)
    candidate_rows = int(cp.get("row_count_capped", 0) or 0)
    reference_rows = int(rp.get("row_count_capped", 0) or 0)
    negative_rows = int(np.get("row_count_capped", 0) or 0)
    if baseline_rows < 1:
        reasons.append("known_fixture_baseline_missing")
    if candidate_rows <= baseline_rows:
        reasons.append("candidate_row_count_not_above_baseline")
    if reference_rows <= baseline_rows:
        reasons.append("reference_row_count_not_above_baseline")
    if negative_rows != 0:
        reasons.append("negative_control_returned_rows")
    if candidate_rows != reference_rows:
        reasons.append("candidate_reference_row_count_disagreement")
    if bool(cp.get("sql_error_shape")) or bool(rp.get("sql_error_shape")):
        reasons.append("candidate_or_reference_sql_error_shape")
    if str(candidate_class) != "widebyte_escape_boundary":
        reasons.append("candidate_class_not_widebyte_escape_boundary")
    if str(reference_class) != "widebyte_escape_boundary":
        reasons.append("reference_class_not_widebyte_escape_boundary")

    candidate_effect = candidate_rows > baseline_rows and negative_rows == 0
    reference_effect = reference_rows > baseline_rows and negative_rows == 0
    agreement = candidate_rows == reference_rows
    evidence = {
        "schema_version": PG256_SCHEMA,
        "route": ROUTE,
        "method": METHOD,
        "source_sha256": source_hash,
        "baseline_row_count_capped": baseline_rows,
        "candidate_row_count_capped": candidate_rows,
        "reference_row_count_capped": reference_rows,
        "negative_row_count_capped": negative_rows,
        "candidate_class": str(candidate_class),
        "reference_class": str(reference_class),
        "candidate_effect": candidate_effect,
        "reference_effect": reference_effect,
        "candidate_reference_agreement": agreement,
        "fresh_reset": True,
        "database_write": False,
        "time_delay_used": False,
        "external_network": False,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
    }
    evidence["evidence_hash"] = sha256_json(evidence)
    confirmed = not reasons
    return {
        "schema_version": PG256_SCHEMA,
        "route": ROUTE,
        "contract": {
            "kind": "widebyte_row_count_differential",
            "row_count_cap": _MAX_ROWS,
            "read_only": True,
            "vulnerability_claim_allowed": False,
        },
        "typed_effect_confirmed": confirmed,
        "confirmed_positive": confirmed,
        "vulnerability_claim_allowed": False,
        "reasons": reasons,
        "evidence": evidence,
        "evidence_hash": evidence["evidence_hash"],
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
    }


__all__ = ["PG256_SCHEMA", "ROUTE", "METHOD", "project_widebyte_response", "evaluate_widebyte_effect"]
