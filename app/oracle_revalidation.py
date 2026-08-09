"""Fail-closed positive-oracle revalidation for cross-application transfer.

The decoder is allowed to propose a family, but it is never allowed to turn a
proposal into a positive result by itself.  This module is the small, explicit
boundary between a model candidate and a pinned local oracle.  It checks the
pair structure, source attestation, bounded oracle projection, and evidence
hashes before accepting a pair.  A malformed or incomplete record returns a
quarantine decision rather than raising a positive result.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .maze_engine import sha256_json


ORACLE_REVALIDATION_SCHEMA = "sift-oracle-revalidation-v1"


def _evidence_hash_valid(record: dict[str, Any]) -> bool:
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


def _pair_metadata(record: dict[str, Any]) -> tuple[str, str, str] | None:
    pair = record.get("pair")
    if not isinstance(pair, dict):
        return None
    pair_id = str(pair.get("pair_id", ""))
    variant = str(pair.get("variant", ""))
    surface_role = str(pair.get("surface_role", ""))
    if not pair_id or not variant or not surface_role:
        return None
    return pair_id, variant, surface_role


def revalidate_positive_pair(
    records: Iterable[dict[str, Any]],
    *,
    expected_family: str,
    oracle_name: str,
    authorized_source_hash: str,
    required_variants: Iterable[str] = ("plain", "url_percent"),
    required_surface_role: str | None = None,
    required_sink_kind: str | None = None,
) -> dict[str, Any]:
    """Revalidate one pair against a pinned, family-specific positive oracle.

    ``records`` are evaluator-side objects and must not be included in the
    decoder input.  The function deliberately requires exactly one record per
    required encoding variant.  It returns a structured reason list so a
    caller can report abstention without guessing why the gate failed.
    """

    rows = [row for row in records if isinstance(row, dict)]
    required = tuple(dict.fromkeys(str(value) for value in required_variants))
    reasons: list[str] = []
    if not rows:
        return {
            "schema_version": ORACLE_REVALIDATION_SCHEMA,
            "accepted": False,
            "reasons": ["empty_pair"],
            "record_count": 0,
        }
    metadata = [_pair_metadata(row) for row in rows]
    if any(item is None for item in metadata):
        reasons.append("missing_pair_metadata")
    valid_metadata = [item for item in metadata if item is not None]
    pair_ids = {item[0] for item in valid_metadata}
    roles = {item[2] for item in valid_metadata}
    variants = [item[1] for item in valid_metadata]
    if len(pair_ids) != 1:
        reasons.append("pair_id_disagreement")
    if len(roles) != 1:
        reasons.append("surface_role_disagreement")
    if required_surface_role is not None and roles != {str(required_surface_role)}:
        reasons.append("unexpected_surface_role")
    counts = Counter(variants)
    if set(counts) != set(required) or any(counts[variant] != 1 for variant in required):
        reasons.append("required_encoding_pair_missing_or_duplicated")

    candidates = {str(row.get("candidate_family", "")) for row in rows}
    if candidates != {str(expected_family)}:
        reasons.append("model_family_disagreement")

    for row in rows:
        semantic = row.get("semantic") or {}
        if str(semantic.get("expected_oracle", "")) != str(oracle_name):
            reasons.append("oracle_contract_mismatch")
        if not _evidence_hash_valid(row):
            reasons.append("invalid_evidence_hash")
        if not _projection_bound_to_evidence(row):
            reasons.append("oracle_projection_not_bound_to_evidence")
        evidence = row.get("evidence") or {}
        reset = evidence.get("reset") or {}
        if str(reset.get("fixture_source_sha256", "")) != str(authorized_source_hash):
            reasons.append("source_attestation_mismatch")
        if not bool(reset.get("fresh_target")):
            reasons.append("fresh_target_attestation_missing")
        if not bool(row.get("rule_ir_result")):
            reasons.append("positive_oracle_not_satisfied")
        projection = row.get("oracle_projection") or {}
        if required_sink_kind is not None and str(projection.get("sink_kind", "")) != str(required_sink_kind):
            reasons.append("sink_binding_mismatch")
        if not bool(projection.get("marker_in_attribute")):
            reasons.append("attribute_oracle_signal_missing")
        if bool(projection.get("marker_in_script_source")):
            reasons.append("unexpected_script_signal")

    unique_reasons = list(dict.fromkeys(reasons))
    return {
        "schema_version": ORACLE_REVALIDATION_SCHEMA,
        "accepted": not unique_reasons,
        "reasons": unique_reasons,
        "record_count": len(rows),
        "pair_id": next(iter(pair_ids), None),
        "surface_role": next(iter(roles), None),
        "variants": sorted(variants),
        "candidate_families": sorted(candidates),
        "evidence_hashes": sorted(
            str((row.get("evidence") or {}).get("evidence_hash", "")) for row in rows
        ),
    }


__all__ = ["ORACLE_REVALIDATION_SCHEMA", "revalidate_positive_pair"]
