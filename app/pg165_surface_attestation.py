"""Independent attestation for bounded PG-51 Docker surface effects.

This verifier deliberately attests only a non-executing canary effect (for
example, a marker was reflected in an HTML projection).  It never claims XSS,
SQL injection, redirect, authentication bypass or any other vulnerability.
The collector's oracle fields and decision fields are not trusted; the
attestor recomputes the bounded result from the response projection and pair
metadata.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Iterable

from .maze_engine import sha256_json


SCHEMA_VERSION = "pg165-independent-surface-attestation-v1"
ATTESTATION_CONTRACT = "pg165-safe-surface-effect-v1"
ATTESTATION_CONTRACT_SHA256 = hashlib.sha256(ATTESTATION_CONTRACT.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _verify_evidence(row: dict[str, Any]) -> tuple[bool, str]:
    evidence = row.get("evidence")
    if not isinstance(evidence, dict):
        return False, "missing_evidence"
    declared = evidence.get("evidence_hash")
    if not _is_sha256(declared):
        return False, "invalid_evidence_hash"
    body = dict(evidence)
    body.pop("evidence_hash", None)
    body.pop("evidence_hash_algorithm", None)
    if sha256_json(body) != declared:
        return False, "evidence_hash_mismatch"
    reset = row.get("reset")
    if not isinstance(reset, dict) or not bool(reset.get("fresh_target")) or reset.get("state_change_allowed") is not False:
        return False, "fresh_reset_gate_failed"
    if reset.get("external_network") is not False or reset.get("evaluator_state_hidden") is not True:
        return False, "reset_safety_gate_failed"
    if not bool(row.get("target_instance_id")):
        return False, "missing_target_instance"
    return True, "ok"


def _projection(row: dict[str, Any]) -> dict[str, Any] | None:
    response = row.get("response_projection")
    if not isinstance(response, dict):
        return None
    marker = response.get("marker")
    if not isinstance(marker, dict):
        marker = {}
    shape = response.get("shape")
    if not isinstance(shape, dict):
        shape = {}
    # Keep only stable, bounded observables.  No body text, headers values,
    # route names, payloads or hashes enter the model-facing projection.
    return {
        "method": str(row.get("method", "")).upper(),
        "placement": str((row.get("payload_manifest") or {}).get("placement", "unknown")),
        "encoding_depth": min(max(int((row.get("payload_manifest") or {}).get("encoding_depth", 0) or 0), 0), 4),
        "status_class": str(response.get("status_class", "unknown")),
        "content_type_class": str(response.get("content_type_class", "unknown")),
        "body_length_bucket": str(response.get("body_length_bucket", "unknown")),
        "marker_reflected": bool(marker.get("reflected")),
        "marker_count": min(max(int(marker.get("count", 0) or 0), 0), 8),
        "marker_location": str(marker.get("location", "none")),
        "state_changed": bool(response.get("state_changed")),
        "status_changed": bool(response.get("status_changed")),
        "location_origin_changed": bool(response.get("location_origin_changed")),
        "transport_error": bool(response.get("transport_error")),
        "shape_kind": str(shape.get("kind", "unknown")),
        "shape_field_count": min(max(int(shape.get("field_count", 0) or 0), 0), 512),
        "shape_scalar_count": min(max(int(shape.get("scalar_count", 0) or 0), 0), 512),
    }


def _pair_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("target_instance_id", "")), str(row.get("surface_id", "")), str(row.get("method", "")).upper())


def attest_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = [dict(row) for row in rows]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    checks: list[dict[str, Any]] = []
    for row in materialized:
        valid, reason = _verify_evidence(row)
        projection = _projection(row)
        row["_attestation_projection"] = projection
        row["_attestation_gate"] = {"evidence_valid": valid, "reason": reason}
        checks.append({"sample_id": row.get("sample_id"), "evidence_valid": valid, "reason": reason})
        if valid and projection is not None:
            groups[_pair_key(row)].append(row)

    attestations: list[dict[str, Any]] = []
    row_attestations: dict[str, dict[str, Any]] = {}
    for key, pair in sorted(groups.items(), key=lambda item: item[0]):
        controls = [row for row in pair if row.get("pair_role") == "control"]
        candidates = [row for row in pair if row.get("pair_role") == "candidate"]
        if len(controls) != 1 or len(candidates) != 1:
            status = "abstain_unmatched_pair"
            reason = "exactly_one_control_and_candidate_required"
        else:
            control, candidate = controls[0], candidates[0]
            cp = control["_attestation_projection"]
            xp = candidate["_attestation_projection"]
            matched_negative = not bool(cp["marker_reflected"]) and not bool(cp["state_changed"])
            if xp["transport_error"] or cp["transport_error"]:
                status = "abstain_environment_failure"
                reason = "transport_error_in_pair"
            elif matched_negative and xp["marker_reflected"] and xp["marker_count"] > 0 and not xp["state_changed"]:
                # This is a safe reflection observation, not code execution.
                status = "confirmed_safe_reflection_effect"
                reason = "candidate_marker_reflection_with_matched_negative"
            elif matched_negative and not xp["marker_reflected"] and not xp["state_changed"]:
                status = "confirmed_safe_no_observable_effect"
                reason = "candidate_and_control_no_bounded_effect"
            else:
                status = "abstain_ambiguous_pair"
                reason = "pair_differential_not_in_safe_contract"
        evidence_hashes = [str(row.get("evidence", {}).get("evidence_hash", "")) for row in pair]
        attestation_body = {
            "schema_version": SCHEMA_VERSION,
            "contract": ATTESTATION_CONTRACT,
            "contract_sha256": ATTESTATION_CONTRACT_SHA256,
            "target_instance_id": key[0],
            "surface_id": key[1],
            "method": key[2],
            "status": status,
            "reason": reason,
            "matched_negative": status.startswith("confirmed_safe_"),
            "vulnerability_claim_allowed": False,
            "evidence_hashes": sorted(evidence_hashes),
        }
        attestation = {**attestation_body, "attestation_sha256": sha256_json(attestation_body)}
        attestations.append(attestation)
        for row in pair:
            row_attestations[str(row.get("sample_id"))] = attestation

    output_rows: list[dict[str, Any]] = []
    for row in materialized:
        sample_id = str(row.get("sample_id"))
        attestation = row_attestations.get(sample_id)
        projection = row.pop("_attestation_projection", None)
        row.pop("_attestation_gate", None)
        row["attestation"] = attestation or {"status": "abstain_unverified_row", "vulnerability_claim_allowed": False}
        row["model_projection"] = projection or {}
        row["training_label"] = {
            "confirmed_safe_reflection_effect": "surface_effect",
            "confirmed_safe_no_observable_effect": "surface_no_effect",
        }.get(row["attestation"].get("status"), "abstain")
        row["training_eligible"] = row["training_label"] != "abstain" and bool(row["attestation"].get("matched_negative"))
        row["vulnerability_label"] = False
        output_rows.append(row)
    eligible = [row for row in output_rows if row["training_eligible"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": ATTESTATION_CONTRACT,
        "contract_sha256": ATTESTATION_CONTRACT_SHA256,
        "rows": output_rows,
        "attestations": attestations,
        "checks": checks,
        "confirmed_safe_effect_count": sum(att["status"] == "confirmed_safe_reflection_effect" for att in attestations),
        "confirmed_safe_no_effect_count": sum(att["status"] == "confirmed_safe_no_observable_effect" for att in attestations),
        "training_eligible_row_count": len(eligible),
        "vulnerability_claim_allowed": False,
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "memory_promotion_allowed": False,
    }


__all__ = ["ATTESTATION_CONTRACT", "ATTESTATION_CONTRACT_SHA256", "SCHEMA_VERSION", "attest_rows"]
