"""Classify whether a dataset contains useful replay context.

This is deliberately an audit, not a trainer.  It never reads raw request or
response bodies and never upgrades a dataset to training authority.  The
purpose is to make the distinction between a schema fixture, a real negative
replay, and a possible training candidate explicit before model evaluation.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


AUDIT_SCHEMA = "sift-dataset-utility-audit-v1"
REQUIRED_CONTEXT = (
    "http_method",
    "route_template_id",
    "placement_or_form_schema",
    "baseline_projection",
    "probe_projection",
    "fresh_reset",
    "typed_oracle_projection",
    "negative_control_pair",
    "evidence_sha256",
)


def _rows(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    value = dataset.get("samples")
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _method(row: dict[str, Any]) -> str | None:
    manifest = row.get("payload_manifest")
    if isinstance(manifest, dict) and manifest.get("method"):
        return str(manifest["method"]).upper()
    context = row.get("request_context")
    if isinstance(context, dict) and context.get("method"):
        return str(context["method"]).upper()
    return None


def _route(row: dict[str, Any]) -> bool:
    manifest = row.get("payload_manifest")
    if isinstance(manifest, dict) and manifest.get("route_template_id"):
        return True
    context = row.get("request_context")
    return isinstance(context, dict) and bool(context.get("route_template_id"))


def _has_placement_or_form(row: dict[str, Any]) -> bool:
    manifest = row.get("payload_manifest")
    if not isinstance(manifest, dict):
        context = row.get("request_context")
        return isinstance(context, dict) and bool(context.get("placement"))
    return bool(manifest.get("placement")) and (
        str(manifest.get("method", "")).upper() != "POST"
        or bool(manifest.get("form_field_names"))
    )


def _has_fresh_reset(row: dict[str, Any]) -> bool:
    reset = row.get("reset")
    return isinstance(reset, dict) and bool(reset.get("fresh_target")) and bool(reset.get("completed")) and bool(reset.get("evaluator_state_hidden"))


def _has_oracle(row: dict[str, Any]) -> bool:
    oracle = row.get("oracle_projection") or row.get("typed_oracle")
    return isinstance(oracle, dict) and bool(oracle)


def _typed_positive(row: dict[str, Any]) -> bool:
    oracle = row.get("oracle_projection") or row.get("typed_oracle")
    if not isinstance(oracle, dict):
        return False
    effect = str(oracle.get("confirmed_effect", oracle.get("effect", "none")))
    return bool(oracle.get("positive")) and bool(oracle.get("positive_authority", False)) and effect != "none"


def _has_negative_pair(row: dict[str, Any]) -> bool:
    pair = row.get("negative_control")
    if isinstance(pair, dict) and bool(pair):
        return True
    evidence = row.get("evidence")
    if isinstance(evidence, dict) and isinstance(evidence.get("negative_control"), dict):
        return True
    return False


def audit_dataset(dataset: dict[str, Any], *, dataset_id: str | None = None) -> dict[str, Any]:
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be an object")
    rows = _rows(dataset)
    methods = sorted({method for row in rows if (method := _method(row))})
    source = dataset.get("source")
    source_type = str(source.get("source_type", "")) if isinstance(source, dict) else ""
    runtime_replay = source_type == "authorized_local_container" or any(
        isinstance(row.get("reset"), dict) and "target_instance_id" in row.get("reset", {}) for row in rows
    )
    fields: Counter[str] = Counter()
    typed_positive_count = 0
    negative_pair_count = 0
    fresh_reset_count = 0
    evidence_hash_count = 0
    for row in rows:
        if _method(row):
            fields["http_method"] += 1
        if _route(row):
            fields["route_template_id"] += 1
        if _has_placement_or_form(row):
            fields["placement_or_form_schema"] += 1
        if isinstance(row.get("reset"), dict) and row["reset"].get("baseline_projection_sha256"):
            fields["baseline_projection"] += 1
        if isinstance(row.get("response_projection"), dict):
            fields["probe_projection"] += 1
        if _has_fresh_reset(row):
            fields["fresh_reset"] += 1
            fresh_reset_count += 1
        if _has_oracle(row):
            fields["typed_oracle_projection"] += 1
        if _has_negative_pair(row):
            fields["negative_control_pair"] += 1
            negative_pair_count += 1
        evidence = row.get("evidence")
        if isinstance(evidence, dict) and evidence.get("evidence_hash"):
            fields["evidence_sha256"] += 1
            evidence_hash_count += 1
        if _typed_positive(row):
            typed_positive_count += 1

    count = len(rows)
    def row_complete(row: dict[str, Any]) -> bool:
        return all(
            (
                _method(row),
                _route(row),
                _has_placement_or_form(row),
                isinstance(row.get("reset"), dict) and bool(row["reset"].get("baseline_projection_sha256")),
                isinstance(row.get("response_projection"), dict),
                _has_fresh_reset(row),
                _has_oracle(row),
                _has_negative_pair(row) or str(row.get("sample_role", "")) == "negative_control",
                isinstance(row.get("evidence"), dict) and bool(row["evidence"].get("evidence_hash")),
            )
        )

    complete_context_rows = sum(1 for row in rows if row_complete(row))
    methods_complete = {"GET", "POST"}.issubset(set(methods))
    evaluation_only = bool(dataset.get("evaluation_only", False)) or dataset.get("training_eligible") is False
    if not runtime_replay or not methods:
        utility_class = "schema_only"
    elif typed_positive_count == 0:
        utility_class = "negative_only_replay_evaluation"
    elif not methods_complete or complete_context_rows != count or negative_pair_count == 0:
        utility_class = "replay_candidate_incomplete"
    else:
        utility_class = "replay_training_candidate_pending_capability_gate"

    blockers: list[str] = []
    if not methods_complete:
        blockers.append("missing_get_or_post_context")
    if not runtime_replay:
        blockers.append("no_runtime_replay")
    if typed_positive_count == 0:
        blockers.append("no_typed_positive")
    if count and complete_context_rows != count:
        blockers.append("incomplete_per_row_context")
    if negative_pair_count == 0:
        blockers.append("no_negative_control_pair")
    if not fresh_reset_count:
        blockers.append("no_fresh_reset")
    if not evidence_hash_count:
        blockers.append("no_evidence_hash")
    if evaluation_only:
        blockers.append("dataset_marked_evaluation_only")

    return {
        "schema_version": AUDIT_SCHEMA,
        "dataset_id": str(dataset_id or dataset.get("manifest_id") or dataset.get("catalog_id") or "unknown"),
        "row_count": count,
        "runtime_replay": runtime_replay,
        "evaluation_only": evaluation_only,
        "training_eligible_declared": bool(dataset.get("training_eligible", False)),
        "methods": methods,
        "get_post_complete": methods_complete,
        "context_field_counts": {field: fields[field] for field in REQUIRED_CONTEXT},
        "complete_context_rows": complete_context_rows,
        "typed_positive_count": typed_positive_count,
        "negative_control_pair_count": negative_pair_count,
        "fresh_reset_count": fresh_reset_count,
        "evidence_hash_count": evidence_hash_count,
        "utility_class": utility_class,
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "blockers": sorted(set(blockers)),
    }


__all__ = ["AUDIT_SCHEMA", "REQUIRED_CONTEXT", "audit_dataset"]
