"""Evaluator-side acceptance gate for PG-25D payload observations.

The gate separates transport/reflection signals from a target-authoritative
success claim.  It never executes a probe and never promotes a row merely
because a marker is reflected, an error appears, or a status/latency changes.
"""

from __future__ import annotations

import copy
from typing import Any

from app.cross_lab_safe_catalog import validate_sample


ACCEPTANCE_PROTOCOL_SCHEMA = "pg-pk-25d-payload-acceptance-v1"
WEAK_MODALITIES = frozenset({"reflection", "syntax_error", "bounded_timing", "status_change", "transport_error"})


def _acceptance_reasons(row: dict[str, Any]) -> list[str]:
    oracle = dict(row.get("oracle_projection") or {})
    decision = dict(row.get("decision") or {})
    reasons: list[str] = []
    if decision.get("evidence_status") != "confirmed_positive":
        reasons.append("not_confirmed_positive")
    if not bool(oracle.get("positive")):
        reasons.append("oracle_not_positive")
    if not bool(oracle.get("positive_authority")):
        reasons.append("oracle_not_authoritative")
    if str(oracle.get("modality")) in WEAK_MODALITIES:
        reasons.append("weak_oracle_modality")
    if str(oracle.get("confirmed_effect", "none")) == "none":
        reasons.append("no_typed_effect")
    regex_evidence = dict((oracle.get("signals") or {}).get("regex_evidence") or {})
    if not regex_evidence:
        reasons.append("regex_evidence_missing")
    elif not bool(regex_evidence.get("matched")):
        reasons.append("regex_evidence_not_matched")
    if not bool(decision.get("oracle_revalidated")):
        reasons.append("oracle_not_revalidated")
    control = row.get("negative_control") or {}
    if not control:
        reasons.append("negative_control_missing")
    if decision.get("training_action") != "accept":
        reasons.append("training_action_not_accept")
    return sorted(set(reasons))


def evaluate_catalog(catalog: dict[str, Any]) -> dict[str, Any]:
    """Validate and summarize a bounded catalog without changing it."""

    if not isinstance(catalog, dict) or not str(catalog.get("schema_version", "")).startswith("sift-cross-lab-safe-catalog-"):
        raise ValueError("unsupported safe catalog for PG-25D acceptance")
    source = dict(catalog.get("source") or {})
    rows = [validate_sample(dict(row), source) for row in catalog.get("samples") or []]
    if not rows:
        raise ValueError("PG-25D acceptance requires at least one validated sample")
    positive_rows = [row for row in rows if row["decision"]["evidence_status"] == "confirmed_positive"]
    accepted_rows = [row for row in rows if not _acceptance_reasons(row)]
    rejection_reasons: dict[str, int] = {}
    for row in rows:
        for reason in _acceptance_reasons(row):
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
    status = "accepted_positive_present" if accepted_rows else "preflight_only_or_abstain"
    return {
        "schema_version": ACCEPTANCE_PROTOCOL_SCHEMA,
        "catalog_id": str(catalog.get("catalog_id", "")),
        "source_id": str(source.get("source_id", "")),
        "source_sha256": str(source.get("source_sha256", "")),
        "sample_count": len(rows),
        "confirmed_positive_count": len(positive_rows),
        "accepted_positive_count": len(accepted_rows),
        "confirmed_negative_count": sum(row["decision"]["evidence_status"] == "confirmed_negative" for row in rows),
        "candidate_count": sum(row["decision"]["evidence_status"] == "candidate" for row in rows),
        "abstain_count": sum(row["decision"]["evidence_status"] == "abstain" for row in rows),
        "training_eligible": bool(catalog.get("training_eligible")) and bool(accepted_rows),
        "status": status,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "raw_bodies_retained": False,
        "raw_probes_retained": False,
        "weak_signals_are_not_success": True,
        "validated_sample_ids": [str(row["sample_id"]) for row in rows],
    }


__all__ = ["ACCEPTANCE_PROTOCOL_SCHEMA", "evaluate_catalog"]
