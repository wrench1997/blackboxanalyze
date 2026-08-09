"""Fail-closed typed oracle for PG-PK-10 logic/access pairs."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .maze_engine import sha256_json


LOGIC_ACCESS_REVALIDATION_SCHEMA = "sift-logic-access-oracle-revalidation-v1"


def _valid_hash(record: dict[str, Any]) -> bool:
    evidence = record.get("evidence")
    if not isinstance(evidence, dict):
        return False
    declared = str(evidence.get("evidence_hash", ""))
    if len(declared) != 64:
        return False
    body = dict(evidence)
    body.pop("evidence_hash", None)
    return declared == sha256_json(body)


def _projection_bound_to_evidence(record: dict[str, Any]) -> bool:
    evidence = record.get("evidence")
    projection = record.get("oracle_projection")
    if not isinstance(evidence, dict) or not isinstance(projection, dict):
        return False
    embedded = evidence.get("oracle_projection")
    return isinstance(embedded, dict) and sha256_json(embedded) == sha256_json(projection)


def _typed_positive(projection: dict[str, Any], expected_family: str, expected_signal: str) -> bool:
    if not bool(projection.get("positive")):
        return False
    if str(projection.get("oracle_signal", "")) != expected_signal:
        return False
    typed = projection.get("typed") or {}
    if expected_family == "access_control":
        return all(bool(typed.get(key)) for key in ("protected_resource", "non_admin_subject", "quota_nonzero", "unexpected_grant"))
    if expected_signal == "business_boundary_mismatch":
        return all(bool(typed.get(key)) for key in ("member", "boundary_hit", "expected_issued", "invariant_violation")) and not bool(typed.get("observed_issued"))
    if expected_signal == "history_binding_mismatch":
        return all(bool(typed.get(key)) for key in ("commit_action", "previous_verified", "unexpected_replay_accept")) and not bool(typed.get("challenge_matches"))
    return False


def revalidate_logic_access_pair(
    records: Iterable[dict[str, Any]],
    *,
    authorized_source_hash: str,
    expected_family: str,
    oracle_name: str,
    expected_signal: str,
    required_variants: Iterable[str] = ("plain", "url_percent"),
) -> dict[str, Any]:
    """Accept only a complete multi-encoding pair with typed evidence.

    The function is intentionally stricter than the model head: a high
    confidence classification, a 200 response, or a generic body delta never
    counts as a logic/access exit on its own.
    """

    rows = [row for row in records if isinstance(row, dict)]
    required = tuple(dict.fromkeys(str(item) for item in required_variants))
    reasons: list[str] = []
    if not rows:
        return {"schema_version": LOGIC_ACCESS_REVALIDATION_SCHEMA, "accepted": False, "reasons": ["empty_pair"], "record_count": 0}
    pair_meta = [row.get("pair") or {} for row in rows]
    pair_ids = {str(pair.get("pair_id", "")) for pair in pair_meta}
    variants = [str(pair.get("variant", "")) for pair in pair_meta]
    if len(pair_ids) != 1 or "" in pair_ids:
        reasons.append("pair_id_disagreement")
    counts = Counter(variants)
    if set(counts) != set(required) or any(counts[item] != 1 for item in required):
        reasons.append("required_encoding_pair_missing_or_duplicated")
    candidates = {str(row.get("candidate_family", "")) for row in rows}
    if candidates != {str(expected_family)}:
        reasons.append("model_family_disagreement")
    observed_signals: set[str] = set()
    oracle_names: set[str] = set()
    for row in rows:
        semantic = row.get("semantic") or {}
        projection = row.get("oracle_projection") or {}
        oracle_names.add(str(semantic.get("expected_oracle", "")))
        observed_signals.add(str(projection.get("oracle_signal", "")))
        if str(semantic.get("expected_oracle", "")) != str(oracle_name):
            reasons.append("oracle_contract_mismatch")
        if not _valid_hash(row):
            reasons.append("invalid_evidence_hash")
        if not _projection_bound_to_evidence(row):
            reasons.append("oracle_projection_not_bound_to_evidence")
        evidence = row.get("evidence") or {}
        reset = evidence.get("reset") or {}
        if str(reset.get("fixture_source_sha256", "")) != str(authorized_source_hash):
            reasons.append("source_attestation_mismatch")
        if not bool(reset.get("fresh_target")):
            reasons.append("fresh_target_attestation_missing")
        if not bool(reset.get("state_change_allowed") is False):
            reasons.append("state_change_policy_missing")
        if not _typed_positive(projection, expected_family, expected_signal):
            reasons.append("typed_oracle_not_satisfied")
        if not bool(row.get("rule_ir_result")):
            reasons.append("positive_rule_ir_result_missing")
        safety = row.get("safety") or {}
        if any(bool(projection.get(key)) for key in ("state_mutated", "database_touched", "external_network")):
            reasons.append("unsafe_oracle_side_effect")
        if bool(safety.get("state_mutated")) or bool(safety.get("credentials_stored")):
            reasons.append("unsafe_record_side_effect")
    unique = list(dict.fromkeys(reasons))
    return {
        "schema_version": LOGIC_ACCESS_REVALIDATION_SCHEMA,
        "accepted": not unique,
        "reasons": unique,
        "record_count": len(rows),
        "pair_id": next(iter(pair_ids), None),
        "expected_family": expected_family,
        "expected_signal": expected_signal,
        "oracle_names": sorted(oracle_names),
        "observed_signals": sorted(observed_signals),
        "variants": sorted(variants),
        "candidate_families": sorted(candidates),
        "evidence_hashes": sorted(str((row.get("evidence") or {}).get("evidence_hash", "")) for row in rows),
    }


__all__ = ["LOGIC_ACCESS_REVALIDATION_SCHEMA", "revalidate_logic_access_pair"]
