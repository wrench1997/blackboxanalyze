"""Bounded result-shape oracle for local Pikachu read-only fixtures."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .maze_engine import sha256_json


PG218_SCHEMA = "pg218-pikachu-result-oracle-v1"
_ROW_MARKERS = ("your uid", "uid:", "email is")


def fixture_values(route: Mapping[str, Any]) -> tuple[dict[str, str], str]:
    """Return a known non-exploit record selector, represented by a kind."""

    fields = {str(item) for item in list(route.get("fields") or [])}
    values: dict[str, str] = {}
    for field in sorted(fields):
        if field.casefold() == "submit":
            values[field] = "submit"
        elif field.casefold() == "id":
            values[field] = "1"
        else:
            values[field] = "kobe"
    return values, "known_record_id_1_or_user_fixture"


def negative_fixture_values(route: Mapping[str, Any], marker: str) -> dict[str, str]:
    fields = {str(item) for item in list(route.get("fields") or [])}
    values: dict[str, str] = {}
    for field in sorted(fields):
        if field.casefold() == "submit":
            values[field] = "submit"
        elif field.casefold() == "id":
            values[field] = "999999"
        else:
            values[field] = marker
    return values


def project_result_response(response: Any, *, route: Mapping[str, Any], fixture_kind: str) -> dict[str, Any]:
    body = str(getattr(response, "text", "") or "").casefold()
    content = bytes(getattr(response, "content", b"") or b"")
    marker_count = min(sum(body.count(marker) for marker in _ROW_MARKERS), 3)
    projection = {
        "status_code": int(getattr(response, "status_code", 0) or 0),
        "status_class": f"{int(getattr(response, 'status_code', 0) or 0) // 100}xx",
        "body_length_bucket": "0" if not content else "1-255" if len(content) <= 255 else "256-4095" if len(content) <= 4095 else "4096-65535" if len(content) <= 65535 else "65536+",
        "row_marker_count": marker_count,
        "result_shape": "record_present" if marker_count else "record_absent",
        "body_sha256": hashlib.sha256(content).hexdigest(),
        "fixture_kind": str(fixture_kind),
        "route": str(route.get("path", "")),
    }
    projection["projection_sha256"] = sha256_json(projection)
    return {"schema_version": PG218_SCHEMA, "response_projection": projection, "raw_response_retained": False}


def evaluate_result_fixture(*, route: Mapping[str, Any], positive: Mapping[str, Any], negative: Mapping[str, Any], typed_effect: Mapping[str, Any], reset: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    positive_projection = dict(positive.get("response_projection") or {})
    negative_projection = dict(negative.get("response_projection") or {})
    if not reset.get("fresh_target") or not reset.get("container_recreated") or reset.get("container_restart_used") or int(reset.get("volume_mount_count", -1)) != 0 or reset.get("database_health_gate") != "mysqli_root_pikachu_ok":
        reasons.append("fresh_database_reset_attestation_missing")
    if int(positive_projection.get("row_marker_count", 0)) < 1:
        reasons.append("known_positive_fixture_record_missing")
    if int(negative_projection.get("row_marker_count", 0)) != 0:
        reasons.append("negative_fixture_has_record")
    if not bool(typed_effect.get("typed_effect_confirmed")):
        reasons.append("typed_input_effect_not_confirmed")
    verified = not reasons
    evidence = {
        "route": str(route.get("path", "")),
        "positive_result_shape": positive_projection.get("result_shape"),
        "negative_result_shape": negative_projection.get("result_shape"),
        "positive_row_marker_count": int(positive_projection.get("row_marker_count", 0)),
        "negative_row_marker_count": int(negative_projection.get("row_marker_count", 0)),
        "typed_effect_confirmed": bool(typed_effect.get("typed_effect_confirmed")),
        "database_write": False,
        "time_delay_used": False,
        "external_network": False,
        "raw_response_bodies_stored": False,
    }
    evidence["evidence_hash"] = sha256_json(evidence)
    return {"schema_version": PG218_SCHEMA, "result_fixture_verified": verified, "reasons": reasons, "evidence": evidence, "evidence_hash": evidence["evidence_hash"], "raw_response_bodies_stored": False, "vulnerability_claim_allowed": False}


__all__ = ["PG218_SCHEMA", "fixture_values", "negative_fixture_values", "project_result_response", "evaluate_result_fixture"]
