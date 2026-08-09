"""Synthetic SQL AST differential oracle with no database execution.

The oracle compares a parameterized query shape with a deliberately
concatenated *abstract* query.  Fragment classes are inert markers rather than
SQL attack payloads.  The result is suitable for the rule-maze evidence layer
and cannot reach a database.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


FRAGMENT_CLASSES = frozenset({
    "plain",
    "quoted_value",
    "operator_like",
    "comment_like",
    "subquery_like",
    "blind_boolean",
    "row_shape",
    "syntax_error",
    "time_delay",
    "local_side_channel",
})


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SqlAstResult:
    fragment_class: str
    parameterized_ast: dict[str, Any]
    concatenated_ast: dict[str, Any]
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fragment_class": self.fragment_class,
            "parameterized_ast": self.parameterized_ast,
            "concatenated_ast": self.concatenated_ast,
            "evidence": dict(self.evidence),
        }


def parameterized_ast() -> dict[str, Any]:
    """Return the safe baseline shape; values stay in a parameter node."""

    return {
        "kind": "select",
        "columns": ["id"],
        "from": "records",
        "where": {"kind": "comparison", "operator": "eq", "left": "column:name", "right": "parameter:sift"},
    }


def concatenated_ast(fragment_class: str) -> dict[str, Any]:
    """Build a shape-only AST for a synthetic concatenation path.

    No candidate text is parsed or executed.  ``operator_like`` and
    ``comment_like`` simply stand for two abstract interpreter-boundary
    classes used by the holdout benchmark.
    """

    fragment_class = str(fragment_class)
    if fragment_class not in FRAGMENT_CLASSES:
        raise ValueError(f"unknown SQL fragment class: {fragment_class}")
    where: dict[str, Any] = {
        "kind": "comparison",
        "operator": "eq",
        "left": "column:name",
        "right": "literal:sift_value",
    }
    if fragment_class == "operator_like":
        where = {
            "kind": "and",
            "args": [
                where,
                {"kind": "comparison", "operator": "eq", "left": "identifier:sift_marker", "right": "identifier:sift_marker"},
            ],
        }
    elif fragment_class == "comment_like":
        where = {"kind": "comment_boundary", "inner": where, "marker": "sift_comment"}
    elif fragment_class == "subquery_like":
        where = {
            "kind": "subquery",
            "correlation": "uncorrelated_marker",
            "inner": {"kind": "comparison", "operator": "eq", "left": "literal:sift", "right": "literal:sift"},
        }
    elif fragment_class == "blind_boolean":
        where = {
            "kind": "boolean_branch",
            "observable": "row_shape",
            "true_branch": where,
            "false_branch": {"kind": "constant", "value": False},
        }
    elif fragment_class == "row_shape":
        where = {
            "kind": "row_shape_branch",
            "true_shape": ["id", "name"],
            "false_shape": ["id"],
        }
    elif fragment_class == "syntax_error":
        return {
            "kind": "parse_error",
            "error_class": "unexpected_synthetic_token",
            "location_bucket": "where_clause",
        }
    elif fragment_class == "time_delay":
        where = {
            "kind": "timing_branch",
            "budget_ms": 25,
            "simulated_latency_ms": 100,
            "then": where,
        }
    elif fragment_class == "local_side_channel":
        where = {
            "kind": "side_channel_marker",
            "channel": "local_memory_queue",
            "marker": "sift_oob_marker",
            "external_network": False,
        }
    return {"kind": "select", "columns": ["id"], "from": "records", "where": where}


def _contains_boundary(node: Any) -> bool:
    if isinstance(node, dict):
        if node.get("kind") in {
            "and",
            "comment_boundary",
            "raw_fragment",
            "subquery",
            "boolean_branch",
            "row_shape_branch",
            "parse_error",
            "timing_branch",
            "side_channel_marker",
        }:
            return True
        return any(_contains_boundary(value) for value in node.values())
    if isinstance(node, list):
        return any(_contains_boundary(value) for value in node)
    return False


def run_sql_ast_oracle(fragment_class: str) -> SqlAstResult:
    fragment_class = str(fragment_class)
    baseline = parameterized_ast()
    candidate = concatenated_ast(fragment_class)
    baseline_hash = _digest(baseline)
    candidate_hash = _digest(candidate)
    differential = baseline_hash != candidate_hash
    boundary = _contains_boundary(candidate)
    is_error = fragment_class == "syntax_error"
    is_timing = fragment_class == "time_delay"
    is_blind = fragment_class in {"blind_boolean", "row_shape"}
    is_local_side_channel = fragment_class == "local_side_channel"
    modality = (
        "syntax_error" if is_error else
        "bounded_timing" if is_timing else
        "blind_response" if is_blind else
        "local_side_channel" if is_local_side_channel else
        "ast_shape"
    )
    evidence = {
        "oracle": "synthetic_sql_ast_differential_v1",
        "modality": modality,
        "candidate_signal": differential,
        "controlled_differential": differential,
        "interpreter_boundary": boundary,
        "baseline_ast_sha256": baseline_hash,
        "candidate_ast_sha256": candidate_hash,
        "execution": "not_run",
        "database_touched": False,
        "network_access": False,
        "parameterized_baseline": True,
        "syntax_error_observed": is_error,
        "error_differential": is_error,
        "blind_boolean_differential": is_blind,
        "row_shape_differential": fragment_class == "row_shape",
        "timing_differential": is_timing,
        "simulated_latency_ms": 100 if is_timing else 0,
        "timeout_budget_ms": 25 if is_timing else None,
        "timeout_observed": is_timing,
        "real_sleep_performed": False,
        "local_callback_observed": is_local_side_channel,
        "local_callback_only": is_local_side_channel,
        "external_network": False,
    }
    evidence["evidence_hash"] = _digest(evidence)
    return SqlAstResult(fragment_class, baseline, candidate, evidence)
