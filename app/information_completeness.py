"""Fail-closed information-completeness and information-entropy audits.

The model projection is intentionally small and must not contain raw requests,
responses, target identities, or evaluator labels.  That privacy/safety
boundary is different from the experiment catalogue boundary: the catalogue
still needs enough structured provenance and evidence to explain every row.
This module audits both layers without serialising sensitive values.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "information-completeness-gate-v1"
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

# These fields are required in the evaluator-side trace.  A field may be
# deliberately hidden from the model, but it may not silently disappear from
# the trace/catalog used to decide whether a row is trainable.
TRACE_FIELDS = (
    "schema_version",
    "episode_id",
    "step_id",
    "sampling_seed",
    "target_instance_id",
    "hypothesis",
    "belief_before",
    "action_manifest",
    "baseline_projection",
    "response_projection",
    "oracle_projection",
    "belief_after",
    "decision",
    "next_action",
    "fresh_reset",
    "evidence_sha256",
    "dataset_stage",
    "online_weight_update",
    "long_term_memory_write",
    "failure_signature",
    "echo",
    "trace_sha256",
)

NESTED_TRACE_FIELDS: dict[str, tuple[str, ...]] = {
    "action_manifest": (
        "method",
        "route_template_id",
        "placement",
        "encoding_chain",
        "probe_sha256",
        "safety",
    ),
    "baseline_projection": ("body_length_bucket", "shape_class", "status_class"),
    "response_projection": (
        "body_length_bucket",
        "shape_class",
        "status_class",
        "transition_delta",
        "response_projection_sha256",
        "scope_changed",
        "visibility_changed",
    ),
    "oracle_projection": (
        "candidate_signal",
        "modality",
        "observed_atoms",
        "oracle_contract_sha256",
        "positive",
        "positive_authority",
        "source_evidence_sha256",
        "safety",
    ),
    "fresh_reset": (
        "completed",
        "fresh_target",
        "reset_epoch",
        "reset_evidence_sha256",
        "evidence_hash",
        "state_change_allowed",
    ),
    "failure_signature": (
        "schema_version",
        "kind",
        "failed_gate",
        "observed_method",
        "methods_seen",
        "candidate_signal",
        "typed_available",
        "positive_authority",
        "probe_round",
        "remaining_probe_budget",
        "key_feature_weights",
        "key_features_ranked",
        "next_action",
        "model_visible",
        "raw_probe_retained",
        "raw_response_retained",
        "memory_promotion_allowed",
    ),
    "echo": ("sha256",),
}

# These omissions are intentional in the model-visible projection.  They must
# be represented in the companion catalogue/manifest, with a reason, rather
# than being confused with an absent observation.
SAFE_MODEL_OMISSIONS = (
    "raw_probe",
    "raw_response_body",
    "target_identity",
    "evaluator_action",
    "positive_authority",
    "family_label",
)

# Information completeness gates the *use* of a row, not whether the row can
# ever be useful.  This staged policy resolves the small-data paradox: an
# incomplete trace may teach the tokenizer/normalizer how to represent an
# unknown field, but it may not teach the action head that the field implies a
# vulnerability or a confirmed stop.
LEARNING_STAGES = (
    "schema_repair",
    "representation_pretrain",
    "capability_train",
    "memory_promotion",
)


def learning_stage(*, complete_trace: bool, replayable: bool, labels_verified: bool, cross_split_clean: bool) -> dict[str, Any]:
    """Return the highest safe stage for one row or batch.

    `schema_repair` is always available for incomplete rows.  Bounded
    next-token/normalization pretraining is allowed when no evaluator label is
    consumed.  Action/safety training and memory require all hard evidence
    gates.  The result is deliberately data-only and contains no payload or
    response content.
    """

    if complete_trace and replayable and labels_verified and cross_split_clean:
        return {
            "stage": "memory_promotion",
            "capability_training_allowed": True,
            "action_supervision_allowed": True,
            "memory_promotion_allowed": True,
        }
    if complete_trace and replayable and labels_verified:
        return {
            "stage": "capability_train",
            "capability_training_allowed": True,
            "action_supervision_allowed": True,
            "memory_promotion_allowed": False,
        }
    if not labels_verified:
        return {
            "stage": "representation_pretrain",
            "capability_training_allowed": False,
            "action_supervision_allowed": False,
            "memory_promotion_allowed": False,
        }
    return {
        "stage": "schema_repair",
        "capability_training_allowed": False,
        "action_supervision_allowed": False,
        "memory_promotion_allowed": False,
    }

COMPANION_MANIFEST_FIELDS = (
    "catalog_schema_version",
    "row_provenance_manifest_sha256",
    "tokenizer_schema_version",
    "tokenizer_config_sha256",
    "source_implementation_manifest_sha256",
    "oracle_contract_manifest_sha256",
    "split_manifest_sha256",
    "omission_policy",
)


def sha256_json(value: Any) -> str:
    """Hash JSON after removing no fields; callers exclude self-hash first."""

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _missing(value: Any) -> bool:
    return value is None or value == "" or value == {} or value == []


def _path(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _field_coverage(items: Sequence[Mapping[str, Any]], fields: Iterable[str]) -> dict[str, dict[str, Any]]:
    total = len(items)
    result: dict[str, dict[str, Any]] = {}
    for field in fields:
        missing = sum(_missing(_path(item, field)) for item in items)
        present = total - missing
        result[field] = {
            "total": total,
            "present": present,
            "missing": missing,
            "presence_rate": round(present / total, 6) if total else 0.0,
        }
    return result


def _flatten_steps(targets: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """Flatten internal traces while retaining only bounded context labels."""

    result: list[dict[str, Any]] = []
    for source_name, target_rows in targets.items():
        for target in target_rows:
            target_context = {
                "source_name": source_name,
                "target_instance_id": target.get("target_instance_id"),
                "target_seed": target.get("target_seed"),
                "target_implementation": target.get("target_implementation"),
                "target_schema_version": target.get("target_schema_version"),
            }
            for episode in target.get("episodes") or []:
                episode_context = {
                    "surface_kind": episode.get("surface_kind"),
                    "episode_variant": episode.get("episode_variant"),
                    "oracle_available": episode.get("oracle_available"),
                    "episode_id": episode.get("episode_id"),
                }
                for step in episode.get("steps") or []:
                    # Keep only the fields needed for audit metrics.  Do not
                    # copy raw request/response values into the audit report.
                    result.append({"step": step, "target": target_context, "episode": episode_context})
    return result


def audit_internal_trace(targets: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    records = _flatten_steps(targets)
    steps = [item["step"] for item in records]
    top = _field_coverage(steps, TRACE_FIELDS)
    nested = {
        parent: _field_coverage(steps, tuple(f"{parent}.{field}" for field in fields))
        for parent, fields in NESTED_TRACE_FIELDS.items()
    }

    # A null parent on the first step is meaningful, not a missing value.  A
    # later step without a parent is a broken trajectory link.
    non_first = [step for step in steps if step.get("step_id", "").rsplit("-s", 1)[-1] != "01"]
    parent_missing_non_first = sum(_missing(step.get("parent_step_id")) for step in non_first)

    # Every evidence field must be a digest, never a raw probe/response body.
    hash_fields = ("evidence_sha256", "trace_sha256", "echo.sha256")
    invalid_hashes = {
        field: sum(not HASH_RE.fullmatch(str(_path(step, field) or "")) for step in steps)
        for field in hash_fields
    }

    # Values absent from old surface schemas must become an explicit sentinel
    # (`unknown`/`not_observed`) in the next collector revision.  Counting the
    # current omissions exposes exactly where abstraction loses information.
    projection_fields = (
        "baseline_projection.shape_class",
        "response_projection.scope_changed",
        "response_projection.visibility_changed",
    )
    implicit_unknown = {
        field: sum(_missing(_path(step, field)) for step in steps) for field in projection_fields
    }

    context_coverage = _field_coverage(
        [
            {**item["target"], **item["episode"], "step_id": item["step"].get("step_id")}
            for item in records
        ],
        (
            "target_instance_id",
            "target_seed",
            "target_implementation",
            "target_schema_version",
            "surface_kind",
            "episode_id",
            "step_id",
        ),
    )
    critical_missing = sum(item["missing"] for item in top.values()) + sum(
        item["missing"] for fields in nested.values() for item in fields.values()
    )
    return {
        "step_count": len(steps),
        "target_count": len({item["target"].get("target_instance_id") for item in records}),
        "episode_count": len({item["episode"].get("episode_id") for item in records}),
        "trace_field_coverage": top,
        "nested_field_coverage": nested,
        "context_field_coverage": context_coverage,
        "parent_missing_non_first": parent_missing_non_first,
        "invalid_evidence_hashes": invalid_hashes,
        "implicit_unknown_projection_fields": implicit_unknown,
        "critical_missing_field_count": critical_missing,
        "trace_critical_fields_complete": critical_missing == 0 and parent_missing_non_first == 0 and not any(invalid_hashes.values()),
    }


def _dataset_rows(dataset: Mapping[str, Any], field: str) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for values in (dataset.get(field) or {}).values():
        if isinstance(values, list):
            rows.extend(item for item in values if isinstance(item, Mapping))
    return rows


def audit_public_dataset(dataset: Mapping[str, Any], visible: Mapping[str, Any], report: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, Any]:
    pretrain = _dataset_rows(dataset, "pretrain_sequences")
    action = _dataset_rows(dataset, "action_finetune_sequences")
    visible_pretrain = _dataset_rows(visible, "pretrain_sequences")
    model_fields = ("row_id", "split", "tokens", "token_count")
    model_coverage = _field_coverage(pretrain + action, model_fields)
    action_label_coverage = _field_coverage(action, ("action_label",))
    pretrain_label_coverage = _field_coverage(pretrain, ("action_label",))
    companion = _field_coverage([dataset], COMPANION_MANIFEST_FIELDS)
    # These fields are deliberately not model-visible, but a separate
    # evaluator-side manifest is required to make the omission auditable.
    missing_companion = [field for field, item in companion.items() if item["missing"]]
    report_row_level_fields = _field_coverage(
        [report],
        (
            "row_level_predictions",
            "row_level_evidence_manifest",
            "selection.dev_only",
        ),
    )
    trace_manifest_fields = _field_coverage(
        [trace],
        ("row_count_manifest", "source_manifest", "dataset_manifest_sha256", "report_manifest_sha256"),
    )
    dataset_hash_valid = False
    stored_dataset_hash = dataset.get("manifest_sha256")
    if stored_dataset_hash:
        without = dict(dataset)
        without.pop("manifest_sha256", None)
        dataset_hash_valid = stored_dataset_hash == sha256_json(without)
    visible_hash_valid = False
    stored_visible_hash = visible.get("manifest_sha256")
    if stored_visible_hash:
        without = dict(visible)
        without.pop("manifest_sha256", None)
        visible_hash_valid = stored_visible_hash == sha256_json(without)

    return {
        "pretrain_row_count": len(pretrain),
        "action_row_count": len(action),
        "visible_pretrain_row_count": len(visible_pretrain),
        "model_projection_field_coverage": model_coverage,
        "action_label_coverage": action_label_coverage,
        "pretrain_action_label_coverage": pretrain_label_coverage,
        "companion_manifest_coverage": companion,
        "missing_companion_manifest_fields": missing_companion,
        "report_row_level_coverage": report_row_level_fields,
        "trace_manifest_coverage": trace_manifest_fields,
        "dataset_manifest_hash_valid": dataset_hash_valid,
        "visible_manifest_hash_valid": visible_hash_valid,
        "model_projection_bounded": all(item["missing"] == 0 for item in model_coverage.values()),
        "labels_separate_from_pretrain": pretrain_label_coverage["action_label"]["missing"] == len(pretrain),
        "companion_catalog_complete": not missing_companion,
        "row_level_replay_auditable": all(item["missing"] == 0 for item in report_row_level_fields.values()),
        "trace_manifest_indexed": all(item["missing"] == 0 for item in trace_manifest_fields.values()),
    }


def audit_token_aliasing(examples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    groups: defaultdict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for example in examples:
        groups[tuple(str(token) for token in example.get("tokens") or [])].append(example)
    surface_collisions = 0
    label_collisions = 0
    method_collisions = 0
    collision_rows = 0
    for values in groups.values():
        surfaces = {str(item.get("surface_kind")) for item in values}
        labels = {str(item.get("label")) for item in values}
        methods = {str((item.get("failure_signature") or {}).get("observed_method")) for item in values}
        if len(surfaces) > 1:
            surface_collisions += 1
        if len(labels) > 1:
            label_collisions += 1
        if len(methods) > 1:
            method_collisions += 1
        if len(surfaces) > 1 or len(labels) > 1 or len(methods) > 1:
            collision_rows += len(values)
    return {
        "sequence_group_count": len(groups),
        "surface_collision_group_count": surface_collisions,
        "label_collision_group_count": label_collisions,
        "method_collision_group_count": method_collisions,
        "collision_row_count": collision_rows,
        "label_aliasing_is_safe": label_collisions == 0,
    }


def build_audit(
    *,
    targets: Mapping[str, Sequence[Mapping[str, Any]]],
    dataset: Mapping[str, Any],
    visible: Mapping[str, Any],
    report: Mapping[str, Any],
    trace: Mapping[str, Any],
    token_examples: Mapping[str, Sequence[Mapping[str, Any]]],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    internal = audit_internal_trace(targets)
    public = audit_public_dataset(dataset, visible, report, trace)
    aliasing = {name: audit_token_aliasing(values) for name, values in token_examples.items()}
    blocking_reasons: list[str] = []
    if not internal["trace_critical_fields_complete"]:
        blocking_reasons.append("internal_trace_material_field_missing_or_invalid")
    if not public["companion_catalog_complete"]:
        blocking_reasons.append("dataset_companion_catalog_manifest_missing")
    if not public["row_level_replay_auditable"]:
        blocking_reasons.append("report_row_level_evidence_manifest_missing")
    if not public["trace_manifest_indexed"]:
        blocking_reasons.append("trace_manifest_not_indexed_to_rows")
    if internal["implicit_unknown_projection_fields"]:
        blocking_reasons.append("missing_projection_values_not_explicit_unknown")
    staged_policy = {
        "incomplete_rows": learning_stage(
            complete_trace=False,
            replayable=False,
            labels_verified=False,
            cross_split_clean=False,
        ),
        "complete_replay_rows": learning_stage(
            complete_trace=True,
            replayable=True,
            labels_verified=True,
            cross_split_clean=False,
        ),
        "promotion_candidate_rows": learning_stage(
            complete_trace=True,
            replayable=True,
            labels_verified=True,
            cross_split_clean=True,
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "target_id": "pg139_value_head_loio",
        "status": "completed_pg139_information_completeness_audit",
        "hard_gate_passed": not blocking_reasons,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "sensitive_content_stored": False,
        "safe_model_omissions": list(SAFE_MODEL_OMISSIONS),
        "internal_trace": internal,
        "public_dataset": public,
        "token_aliasing": aliasing,
        "blocking_reasons": blocking_reasons,
        "staged_learning_policy": staged_policy,
        "source_hashes": dict(source_hashes),
        "promotion": {
            "training_artifact_promotion_allowed": False,
            "memory_promotion_allowed": False,
            "reason": "information completeness must pass before any score or memory promotion",
        },
    }


def finalize_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(audit)
    result["audit_sha256"] = sha256_json(result)
    return result
