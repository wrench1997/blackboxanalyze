"""Fail-closed pair revalidation for abstract SQL-channel evidence."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .maze_engine import sha256_json


SQL_REVALIDATION_SCHEMA = "sift-sql-oracle-revalidation-v1"


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


def revalidate_sql_pair(
    records: Iterable[dict[str, Any]],
    *,
    authorized_source_hash: str,
    oracle_name: str,
    expected_family: str = "injection",
    required_variants: Iterable[str] = ("plain", "url_percent"),
) -> dict[str, Any]:
    rows = [row for row in records if isinstance(row, dict)]
    required = tuple(dict.fromkeys(str(item) for item in required_variants))
    reasons: list[str] = []
    if not rows:
        return {"schema_version": SQL_REVALIDATION_SCHEMA, "accepted": False, "reasons": ["empty_pair"], "record_count": 0}
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
    modalities: set[str] = set()
    for row in rows:
        semantic = row.get("semantic") or {}
        projection = row.get("oracle_projection") or {}
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
        if not bool(projection.get("controlled_differential")):
            reasons.append("controlled_differential_missing")
        if not bool(projection.get("interpreter_boundary")):
            reasons.append("interpreter_boundary_missing")
        if not bool(row.get("rule_ir_result")):
            reasons.append("positive_sql_oracle_not_satisfied")
        modalities.add(str(projection.get("modality", "")))
        if bool(projection.get("database_touched")) or bool(projection.get("real_sleep_performed")) or bool(projection.get("external_network")):
            reasons.append("unsafe_sql_oracle_side_effect")
    unique = list(dict.fromkeys(reasons))
    return {
        "schema_version": SQL_REVALIDATION_SCHEMA,
        "accepted": not unique,
        "reasons": unique,
        "record_count": len(rows),
        "pair_id": next(iter(pair_ids), None),
        "variants": sorted(variants),
        "modalities": sorted(modalities),
        "candidate_families": sorted(candidates),
        "evidence_hashes": sorted(str((row.get("evidence") or {}).get("evidence_hash", "")) for row in rows),
    }


__all__ = ["SQL_REVALIDATION_SCHEMA", "revalidate_sql_pair"]
