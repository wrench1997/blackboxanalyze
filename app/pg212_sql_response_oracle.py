"""Evaluator-only SQL response-shape helpers for local Pikachu replay.

This module deliberately does not implement a SQL engine or a time-based
oracle.  It classifies only bounded transport/response shapes and detects the
known Pikachu configuration-failure page in memory.  Runtime probe values are
created for one loopback request and are never returned by the projection.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import httpx

from .pg179b_iterative_probe import _summary, validate_marker


PG212_SCHEMA = "pg212-pikachu-sql-response-shape-oracle-v1"
_BACKEND_FAILURE_MARKERS = (
    "database connection failed",
    "could not connect to database",
    "数据库连接失败",
    "数据库连接错误",
)
_SAFE_CLASSES = frozenset({"syntax_shape", "encoded_syntax_shape"})
_FORBIDDEN_CLASSES = frozenset({"time_delay", "local_side_channel", "write", "destructive"})


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


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


def _backend_state(body_text: str) -> str:
    lowered = str(body_text).casefold()
    if any(marker in lowered for marker in _BACKEND_FAILURE_MARKERS):
        return "database_unavailable"
    return "backend_response_observed"


def project_sql_response(
    response: httpx.Response,
    *,
    marker: str,
    baseline_status: int | None = None,
) -> dict[str, Any]:
    """Project a SQL response without retaining its body text."""

    marker = validate_marker(marker)
    projection, signal, _ = _summary(response, marker=marker, baseline_status=baseline_status)
    body_text = response.text
    backend_state = _backend_state(body_text)
    # ``_summary`` already bounds marker/error signals.  Only a state label and
    # a structural digest leave this function; the body itself is discarded.
    projection = dict(projection)
    projection["backend_state"] = backend_state
    projection["body_length_bucket"] = _length_bucket(len(response.content))
    projection["projection_sha256"] = _digest(projection)
    oracle = {
        "oracle_id": "pg212-sql-response-shape-v1",
        "typed_available": backend_state == "backend_response_observed",
        "backend_state": backend_state,
        "response_shape_only": True,
        "sql_ast_observed": False,
        "database_execution_confirmed": False,
        "time_delay_used": False,
        "database_write": False,
        "vulnerability_claim_allowed": False,
        "abstain_reason": "pikachu_database_unavailable" if backend_state == "database_unavailable" else "no_backend_ast_or_result_oracle",
        "signal": {
            "status_changed": bool(signal.get("status_changed")),
            "sql_error_shape": bool(signal.get("sql_error_shape")),
            "marker_reflected": bool(signal.get("marker_reflected")),
            "body_length_bucket": projection.get("body_length_bucket"),
            "shape": dict(projection.get("shape") or {}),
        },
    }
    oracle["oracle_sha256"] = _digest(oracle)
    return {
        "schema_version": PG212_SCHEMA,
        "response_projection": projection,
        "oracle": oracle,
        "signal": {
            "status_changed": bool(signal.get("status_changed")),
            "sql_error_shape": bool(signal.get("sql_error_shape")),
            "marker_reflected": bool(signal.get("marker_reflected")),
        },
        "raw_response_retained": False,
    }


def compare_sql_shapes(control: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Compare two bounded projections; never call the result a vulnerability."""

    left = dict(control.get("response_projection") or {})
    right = dict(candidate.get("response_projection") or {})
    fields = {
        "status_class_changed": left.get("status_class") != right.get("status_class"),
        "body_length_bucket_changed": left.get("body_length_bucket") != right.get("body_length_bucket"),
        "shape_changed": (left.get("shape") or {}) != (right.get("shape") or {}),
        "backend_state_changed": left.get("backend_state") != right.get("backend_state"),
        "sql_error_shape_changed": bool((control.get("signal") or {}).get("sql_error_shape")) != bool((candidate.get("signal") or {}).get("sql_error_shape")),
    }
    differential = any(bool(value) for value in fields.values())
    return {
        "response_shape_differential": differential,
        "fields": fields,
        "positive": False,
        "vulnerability_claim_allowed": False,
        "evidence_sha256": _digest(fields),
    }


def build_sql_probe_values(*, field_names: list[str], marker: str, probe_class: str) -> dict[str, str]:
    """Create a short-lived, non-time-based SQL syntax probe."""

    marker = validate_marker(marker)
    probe_class = str(probe_class)
    if probe_class in _FORBIDDEN_CLASSES or probe_class not in _SAFE_CLASSES and probe_class != "control":
        raise ValueError("PG-212 probe class is not allow-listed")
    values: dict[str, str] = {}
    for field in sorted({str(item) for item in field_names if str(item)}):
        lowered = field.casefold()
        if lowered == "submit":
            values[field] = "submit"
        elif lowered == "id":
            values[field] = "1" if probe_class == "control" else "1'"
        elif probe_class == "control":
            values[field] = marker
        elif probe_class == "encoded_syntax_shape":
            values[field] = marker + "%27"
        else:
            # A single quote is a bounded syntax-shape probe, not a query,
            # comment, delay, write, or external callback.
            values[field] = marker + "'"
    return values


__all__ = [
    "PG212_SCHEMA",
    "build_sql_probe_values",
    "compare_sql_shapes",
    "project_sql_response",
]
