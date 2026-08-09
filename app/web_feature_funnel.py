"""Feature-candidate funnel for safe webpage response datasets.

The funnel separates three things that are often mixed together:

* model-visible, bounded observations (method, response shape and paired
  differentials),
* audit-only grouping/labels (source, seed and typed family), and
* provenance/safety evidence (reset and hashes).

It rejects shortcuts such as URL tokens, source IDs, raw bodies, oracle values
and request strings before any utility score is calculated.  Utility is used
to choose among safe observations, never to smuggle the typed oracle into the
model feature vector.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable


FUNNEL_SCHEMA = "sift-web-feature-funnel-v1"
FAMILY_LABELS = (
    "xss",
    "injection",
    "authentication",
    "access_control",
    "logic",
    "url_redirect",
    "input_validation",
    "command_injection",
    "ordinary_response",
)
FORBIDDEN_MODEL_FIELDS = {
    "family",
    "source_id",
    "implementation",
    "variant",
    "sampling_seed",
    "seed",
    "surface",
    "sample_id",
    "oracle",
    "positive",
    "typed_oracle",
    "evidence",
    "body_sha256",
    "semantic_body_sha256",
    "payload_sha256",
    "raw_payload",
    "raw_response",
    "path",
    "url",
    "query",
}

FEATURE_META: dict[str, dict[str, Any]] = {
    "method_get": {"group": "transport", "cost": "cheap", "designed_variation": True},
    "placement_body": {"group": "transport", "cost": "cheap", "designed_variation": True},
    "status_2xx": {"group": "response_shape", "cost": "cheap"},
    "content_type_json": {"group": "response_shape", "cost": "cheap"},
    "shape_kind_object": {"group": "response_shape", "cost": "cheap"},
    "shape_key_count_bucket": {"group": "response_shape", "cost": "cheap"},
    "shape_scalar_count_bucket": {"group": "response_shape", "cost": "cheap"},
    "shape_array_count": {"group": "response_shape", "cost": "cheap"},
    "body_length_bucket": {"group": "response_shape", "cost": "cheap"},
    "status_changed_from_control": {"group": "paired_differential", "cost": "cheap"},
    "shape_key_delta_from_control": {"group": "paired_differential", "cost": "cheap"},
    "shape_scalar_delta_from_control": {"group": "paired_differential", "cost": "cheap"},
    "shape_array_delta_from_control": {"group": "paired_differential", "cost": "cheap"},
    "semantic_shape_changed_from_control": {"group": "paired_differential", "cost": "cheap"},
    "semantic_shape_changed_from_screen": {"group": "paired_differential", "cost": "cheap"},
    "screen_projection_present": {"group": "protocol_context", "cost": "cheap", "designed_variation": True},
    # These are useful evaluator diagnostics, but the current contract does
    # not allow them into the model: their semantics are too close to a typed
    # effect oracle until a separate leakage experiment clears them.
    "surface_boolean_field_count": {"group": "surface_observation", "cost": "cheap", "model_eligible": False},
    "surface_true_boolean_count": {"group": "surface_observation", "cost": "cheap", "model_eligible": False},
    "surface_numeric_field_count": {"group": "surface_observation", "cost": "cheap", "model_eligible": False},
    "surface_nonzero_numeric_count": {"group": "surface_observation", "cost": "cheap", "model_eligible": False},
    "surface_array_field_count": {"group": "surface_observation", "cost": "cheap", "model_eligible": False},
    "surface_key_bucket_count": {"group": "surface_observation", "cost": "cheap", "model_eligible": False},
    "surface_key_bucket_00": {"group": "surface_observation", "cost": "cheap", "model_eligible": False},
    "surface_key_bucket_01": {"group": "surface_observation", "cost": "cheap", "model_eligible": False},
    "surface_key_bucket_02": {"group": "surface_observation", "cost": "cheap", "model_eligible": False},
    "surface_key_bucket_03": {"group": "surface_observation", "cost": "cheap", "model_eligible": False},
    "surface_key_bucket_04": {"group": "surface_observation", "cost": "cheap", "model_eligible": False},
    "surface_key_bucket_05": {"group": "surface_observation", "cost": "cheap", "model_eligible": False},
    "surface_key_bucket_06": {"group": "surface_observation", "cost": "cheap", "model_eligible": False},
    "surface_key_bucket_07": {"group": "surface_observation", "cost": "cheap", "model_eligible": False},
    "surface_true_boolean_delta_from_control": {"group": "surface_differential", "cost": "cheap", "model_eligible": False},
    "surface_numeric_delta_from_control": {"group": "surface_differential", "cost": "cheap", "model_eligible": False},
    "surface_key_overlap_control": {"group": "surface_differential", "cost": "cheap", "model_eligible": False},
    "geometry_object_count": {"group": "anonymous_geometry", "cost": "cheap"},
    "geometry_array_count": {"group": "anonymous_geometry", "cost": "cheap"},
    "geometry_array_item_count": {"group": "anonymous_geometry", "cost": "cheap"},
    "geometry_boolean_count": {"group": "anonymous_geometry", "cost": "cheap"},
    "geometry_true_boolean_count": {"group": "anonymous_geometry", "cost": "cheap"},
    "geometry_numeric_count": {"group": "anonymous_geometry", "cost": "cheap"},
    "geometry_nonzero_numeric_count": {"group": "anonymous_geometry", "cost": "cheap"},
    "geometry_string_count": {"group": "anonymous_geometry", "cost": "cheap"},
    "geometry_string_length_bucket_sum": {"group": "anonymous_geometry", "cost": "cheap"},
    "geometry_leaf_count": {"group": "anonymous_geometry", "cost": "cheap"},
    "geometry_max_depth": {"group": "anonymous_geometry", "cost": "cheap"},
    "geometry_true_boolean_delta_control": {"group": "anonymous_geometry_differential", "cost": "cheap"},
    "geometry_leaf_delta_control": {"group": "anonymous_geometry_differential", "cost": "cheap"},
    "geometry_numeric_delta_control": {"group": "anonymous_geometry_differential", "cost": "cheap"},
    "geometry_array_delta_control": {"group": "anonymous_geometry_differential", "cost": "cheap"},
    "geometry_change_presence_control": {"group": "anonymous_geometry_differential", "cost": "cheap"},
    "geometry_leaf_delta_ratio_control": {"group": "anonymous_geometry_differential", "cost": "cheap"},
    "geometry_true_boolean_delta_ratio_control": {"group": "anonymous_geometry_differential", "cost": "cheap"},
    "geometry_array_item_delta_ratio_control": {"group": "anonymous_geometry_differential", "cost": "cheap"},
    "geometry_object_delta_ratio_control": {"group": "anonymous_geometry_differential", "cost": "cheap"},
    "shape_key_delta_ratio_control": {"group": "paired_differential", "cost": "cheap"},
}


def _bucket_length(value: Any) -> int:
    text = str(value or "0")
    try:
        return int(text.split("-", 1)[0])
    except (TypeError, ValueError):
        return 0


def _shape(response: dict[str, Any] | None) -> dict[str, Any]:
    return dict((response or {}).get("shape") or {})


def _same(left: Any, right: Any) -> int:
    return int(left == right)


def _relative_delta(value: Any, control: Any) -> float:
    """Return a bounded, envelope-relative delta without using field names."""

    try:
        left = float(value)
        right = float(control)
    except (TypeError, ValueError):
        return 0.0
    denominator = max(abs(right), 1.0)
    return max(-1.0, min(1.0, (left - right) / denominator))


def _feature_value_map(response: dict[str, Any], control: dict[str, Any], screen: dict[str, Any] | None, surface_obs: dict[str, Any], control_surface_obs: dict[str, Any], screen_surface_obs: dict[str, Any] | None, geometry: dict[str, Any], control_geometry: dict[str, Any], screen_geometry: dict[str, Any] | None, *, method: str) -> dict[str, int | float]:
    shape = _shape(response)
    control_shape = _shape(control)
    screen_shape = _shape(screen) if screen else {}
    key_buckets = {int(value) % 8 for value in list(surface_obs.get("key_hash_buckets") or [])}
    control_key_buckets = {int(value) % 8 for value in list(control_surface_obs.get("key_hash_buckets") or [])}
    values: dict[str, int | float] = {
        "method_get": int(str(method).upper() == "GET"),
        "placement_body": int(str(method).upper() == "POST"),
        "status_2xx": int(str(response.get("status_class", "")) == "2xx"),
        "content_type_json": int(str(response.get("content_type_class", "")) == "json"),
        "shape_kind_object": int(str(shape.get("kind", "")) == "object"),
        "shape_key_count_bucket": int(shape.get("key_count", 0)) // 2,
        "shape_scalar_count_bucket": int(shape.get("scalar_count", 0)) // 2,
        "shape_array_count": min(8, int(shape.get("array_count", 0))),
        "body_length_bucket": _bucket_length(response.get("body_length_bucket")),
        "status_changed_from_control": int(response.get("status_class") != control.get("status_class")),
        "shape_key_delta_from_control": int(shape.get("key_count", 0)) - int(control_shape.get("key_count", 0)),
        "shape_scalar_delta_from_control": int(shape.get("scalar_count", 0)) - int(control_shape.get("scalar_count", 0)),
        "shape_array_delta_from_control": int(shape.get("array_count", 0)) - int(control_shape.get("array_count", 0)),
        "semantic_shape_changed_from_control": int(response.get("semantic_body_sha256") != control.get("semantic_body_sha256")),
        "semantic_shape_changed_from_screen": int(bool(screen) and response.get("semantic_body_sha256") != screen.get("semantic_body_sha256")),
        "screen_projection_present": int(screen is not None),
        "surface_boolean_field_count": min(32, int(surface_obs.get("boolean_field_count", 0))),
        "surface_true_boolean_count": min(32, int(surface_obs.get("true_boolean_count", 0))),
        "surface_numeric_field_count": min(32, int(surface_obs.get("numeric_field_count", 0))),
        "surface_nonzero_numeric_count": min(32, int(surface_obs.get("nonzero_numeric_count", 0))),
        "surface_array_field_count": min(8, int(surface_obs.get("array_field_count", 0))),
        "surface_key_bucket_count": min(8, len(key_buckets)),
        "surface_true_boolean_delta_from_control": int(surface_obs.get("true_boolean_count", 0)) - int(control_surface_obs.get("true_boolean_count", 0)),
        "surface_numeric_delta_from_control": int(surface_obs.get("nonzero_numeric_count", 0)) - int(control_surface_obs.get("nonzero_numeric_count", 0)),
        "surface_key_overlap_control": int(bool(key_buckets) and bool(control_key_buckets)) * len(key_buckets & control_key_buckets),
        "geometry_object_count": min(64, int(geometry.get("object_count", 0))),
        "geometry_array_count": min(32, int(geometry.get("array_count", 0))),
        "geometry_array_item_count": min(64, int(geometry.get("array_item_count", 0))),
        "geometry_boolean_count": min(64, int(geometry.get("boolean_count", 0))),
        "geometry_true_boolean_count": min(64, int(geometry.get("true_boolean_count", 0))),
        "geometry_numeric_count": min(64, int(geometry.get("numeric_count", 0))),
        "geometry_nonzero_numeric_count": min(64, int(geometry.get("nonzero_numeric_count", 0))),
        "geometry_string_count": min(64, int(geometry.get("string_count", 0))),
        "geometry_string_length_bucket_sum": min(128, int(geometry.get("string_length_bucket_sum", 0))),
        "geometry_leaf_count": min(128, int(geometry.get("leaf_count", 0))),
        "geometry_max_depth": min(16, int(geometry.get("max_depth", 0))),
        "geometry_true_boolean_delta_control": int(geometry.get("true_boolean_count", 0)) - int(control_geometry.get("true_boolean_count", 0)),
        "geometry_leaf_delta_control": int(geometry.get("leaf_count", 0)) - int(control_geometry.get("leaf_count", 0)),
        "geometry_numeric_delta_control": int(geometry.get("nonzero_numeric_count", 0)) - int(control_geometry.get("nonzero_numeric_count", 0)),
        "geometry_array_delta_control": int(geometry.get("array_count", 0)) - int(control_geometry.get("array_count", 0)),
        "geometry_change_presence_control": int(any(
            geometry.get(key, 0) != control_geometry.get(key, 0)
            for key in ("object_count", "array_count", "array_item_count", "boolean_count", "true_boolean_count", "numeric_count", "nonzero_numeric_count", "string_count", "leaf_count", "max_depth")
        )),
        "geometry_leaf_delta_ratio_control": _relative_delta(geometry.get("leaf_count", 0), control_geometry.get("leaf_count", 0)),
        "geometry_true_boolean_delta_ratio_control": _relative_delta(geometry.get("true_boolean_count", 0), control_geometry.get("true_boolean_count", 0)),
        "geometry_array_item_delta_ratio_control": _relative_delta(geometry.get("array_item_count", 0), control_geometry.get("array_item_count", 0)),
        "geometry_object_delta_ratio_control": _relative_delta(geometry.get("object_count", 0), control_geometry.get("object_count", 0)),
        "shape_key_delta_ratio_control": _relative_delta(shape.get("key_count", 0), control_shape.get("key_count", 0)),
    }
    for index in range(8):
        values[f"surface_key_bucket_{index:02d}"] = int(index in key_buckets)
    return values


def build_feature_row(row: dict[str, Any]) -> dict[str, Any]:
    """Build a safe feature row from a PG-53 replay row."""

    candidate = dict(row.get("candidate", {}).get("response") or {})
    control = dict(row.get("control", {}).get("response") or {})
    screen_value = row.get("screen")
    screen = dict(screen_value.get("response") or {}) if isinstance(screen_value, dict) else None
    candidate_surface_obs = dict(row.get("candidate", {}).get("surface_observation") or {})
    control_surface_obs = dict(row.get("control", {}).get("surface_observation") or {})
    screen_surface_obs = dict(screen_value.get("surface_observation") or {}) if isinstance(screen_value, dict) else None
    candidate_geometry = dict(row.get("candidate", {}).get("generic_effect_geometry") or {})
    control_geometry = dict(row.get("control", {}).get("generic_effect_geometry") or {})
    screen_geometry = dict(screen_value.get("generic_effect_geometry") or {}) if isinstance(screen_value, dict) else None
    values = _feature_value_map(candidate, control, screen, candidate_surface_obs, control_surface_obs, screen_surface_obs, candidate_geometry, control_geometry, screen_geometry, method=str(row.get("method", "GET")))
    # Only these audit fields are persisted as metadata; none is copied into
    # model_features.  This lets the funnel measure source/seed drift without
    # allowing the model to memorize the split.
    audit = {
        "sample_id": str(row.get("sample_id", "")),
        "family": str(row.get("family", "ordinary_response")),
        "source_id": str(row.get("source_id", "")),
        "implementation": str(row.get("implementation", "")),
        "variant": str(row.get("variant", "")),
        "sampling_seed": int(row.get("sampling_seed", 0)),
        "surface": str(row.get("surface", "")),
        "decision": str(row.get("decision", "")),
    }
    return {
        "sample_id": audit["sample_id"],
        "candidate_features": values,
        "model_features": {
            name: value
            for name, value in values.items()
            if FEATURE_META.get(name, {}).get("model_eligible", True)
        },
        "audit_metadata": audit,
        "feature_schema": "web-visible-response-shape-and-paired-differential-v1",
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
    }


def build_feature_dataset(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    feature_rows = [build_feature_row(row) for row in rows]
    return {
        "schema_version": FUNNEL_SCHEMA,
        "dataset_id": "pg53-web-feature-funnel",
        "training_eligible": False,
        "evaluation_only": True,
        "candidate_feature_names": list(FEATURE_META),
        "model_feature_names": [name for name, meta in FEATURE_META.items() if meta.get("model_eligible", True)],
        "feature_meta": FEATURE_META,
        "rows": feature_rows,
        "model_feature_policy": {
            "oracle_is_label_not_feature": True,
            "source_seed_surface_are_audit_only": True,
            "raw_request_response_forbidden": True,
            "hashes_are_evidence_only": True,
            "surface_observation_model_eligible": False,
            "generic_effect_geometry_no_field_names": True,
        },
    }


def _entropy(values: list[str]) -> float:
    counts = Counter(values)
    total = len(values)
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log(count / total) for count in counts.values() if count)


def _normalised_mutual_information(feature: list[str], label: list[str]) -> float:
    if not feature or len(feature) != len(label):
        return 0.0
    total = len(feature)
    f_counts = Counter(feature)
    l_counts = Counter(label)
    joint = Counter(zip(feature, label))
    mi = 0.0
    for (f_value, l_value), count in joint.items():
        p_joint = count / total
        p_f = f_counts[f_value] / total
        p_l = l_counts[l_value] / total
        mi += p_joint * math.log(p_joint / (p_f * p_l))
    denominator = max(min(_entropy(feature), _entropy(label)), 1e-12)
    return max(0.0, min(1.0, mi / denominator))


def _source_shift(feature: list[str], sources: list[str]) -> float:
    groups = sorted(set(sources))
    if len(groups) < 2:
        return 0.0
    distributions: list[dict[str, float]] = []
    vocabulary = sorted(set(feature))
    for group in groups:
        values = [value for value, source in zip(feature, sources) if source == group]
        counts = Counter(values)
        total = max(len(values), 1)
        distributions.append({token: counts[token] / total for token in vocabulary})
    # Average pairwise total-variation distance is bounded in [0, 1].
    distances: list[float] = []
    for index, left in enumerate(distributions):
        for right in distributions[index + 1:]:
            distances.append(0.5 * sum(abs(left[token] - right[token]) for token in vocabulary))
    return sum(distances) / max(len(distances), 1)


def _seed_stability(feature: list[str], rows: list[dict[str, Any]]) -> float:
    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for value, row in zip(feature, rows):
        meta = row["audit_metadata"]
        groups[(str(meta["source_id"]), str(meta["surface"]), str(meta["sample_id"]).rsplit("-", 1)[-1])].append(value)
    # Use source/surface/method groups; sample_id's final token is GET/POST.
    if not groups:
        return 0.0
    stable = 0
    for values in groups.values():
        stable += int(len(set(values)) <= 1)
    return stable / len(groups)


def _coverage(feature: list[Any]) -> float:
    return sum(value is not None for value in feature) / max(len(feature), 1)


def _redundancy(left: list[str], right: list[str]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    return sum(a == b for a, b in zip(left, right)) / len(left)


def audit_feature_funnel(dataset: dict[str, Any]) -> dict[str, Any]:
    rows = [dict(row) for row in dataset.get("rows", []) if isinstance(row, dict)]
    names = [str(name) for name in dataset.get("candidate_feature_names", dataset.get("model_feature_names", []))]
    labels = [str(row.get("audit_metadata", {}).get("family", "ordinary_response")) for row in rows]
    sources = [str(row.get("audit_metadata", {}).get("implementation", "unknown")) for row in rows]
    feature_stats: dict[str, dict[str, Any]] = {}
    for name in names:
        values = [row.get("candidate_features", row.get("model_features", {})).get(name) for row in rows]
        tokens = ["<missing>" if value is None else str(value) for value in values]
        meta = dict(FEATURE_META.get(name, {}))
        stats = {
            **meta,
            "coverage": round(_coverage(values), 6),
            "unique_count": len(set(tokens)),
            "constant": len(set(tokens)) <= 1,
            "family_utility_nmi": round(_normalised_mutual_information(tokens, labels), 6),
            "source_leakage_nmi": round(_normalised_mutual_information(tokens, sources), 6),
            "source_distribution_shift": round(_source_shift(tokens, sources), 6),
            "seed_stability": round(_seed_stability(tokens, rows), 6),
            "missing": sum(value is None for value in values),
        }
        feature_stats[name] = stats

    # Funnel stages are deliberately explicit.  A feature cannot be accepted
    # merely because it predicts the current labels: it must survive quality,
    # source-leakage, stability and redundancy checks first.
    stage1 = [
        name for name in names
        if name not in FORBIDDEN_MODEL_FIELDS
        and bool(FEATURE_META.get(name, {}).get("model_eligible", True))
    ]
    stage2 = [
        name for name in stage1
        if feature_stats[name]["coverage"] >= 0.95
        and not feature_stats[name]["constant"]
        and feature_stats[name]["unique_count"] <= max(32, len(rows) // 2)
    ]
    stage3 = [
        name for name in stage2
        if feature_stats[name]["source_leakage_nmi"] <= 0.60
        and feature_stats[name]["source_distribution_shift"] <= 0.50
    ]
    stage4 = [
        name for name in stage3
        if bool(feature_stats[name].get("designed_variation", False))
        or feature_stats[name]["seed_stability"] >= 0.50
    ]
    stage5 = [
        name for name in stage4
        if feature_stats[name]["family_utility_nmi"] >= 0.01
    ]
    accepted: list[str] = []
    redundant: dict[str, str] = {}
    for name in sorted(stage5, key=lambda item: (-feature_stats[item]["family_utility_nmi"], item)):
        duplicate = next((kept for kept in accepted if _redundancy(
            [str(row.get("candidate_features", row.get("model_features", {})).get(name)) for row in rows],
            [str(row.get("candidate_features", row.get("model_features", {})).get(kept)) for row in rows],
        ) >= 0.98), None)
        if duplicate is None:
            accepted.append(name)
        else:
            redundant[name] = duplicate
    stage6 = list(accepted)

    decisions: dict[str, dict[str, Any]] = {}
    for name in names:
        if name in stage6:
            decision = "accepted"
            reason = "passed_quality_leakage_stability_utility_redundancy"
        elif name in redundant:
            decision = "dropped_redundant"
            reason = f"duplicate_of:{redundant[name]}"
        elif name not in stage1:
            decision = "dropped_forbidden"
            reason = "field_or_provenance_shortcut"
        elif name not in stage1 and not bool(FEATURE_META.get(name, {}).get("model_eligible", True)):
            decision = "dropped_evaluator_adjacent"
            reason = "surface_observation_requires_separate_leakage_review"
        elif name not in stage2:
            decision = "dropped_quality"
            reason = "coverage_constant_or_cardinality"
        elif name not in stage3:
            decision = "dropped_source_shift"
            reason = "source_leakage_or_distribution_shift"
        elif name not in stage4:
            decision = "dropped_seed_instability"
            reason = "unstable_across_sampling_seeds"
        else:
            decision = "dropped_low_utility"
            reason = "below_family_utility_threshold"
        decisions[name] = {"decision": decision, "reason": reason}

    return {
        "schema_version": FUNNEL_SCHEMA,
        "dataset_id": dataset.get("dataset_id", "unknown"),
        "row_count": len(rows),
        "source_count": len(set(sources)),
        "family_count": len(set(labels)),
        "stage_counts": {
            "candidate": len(names),
            "observable_safe": len(stage1),
            "quality": len(stage2),
            "source_leakage": len(stage3),
            "seed_stability": len(stage4),
            "label_utility_audit": len(stage5),
            "redundancy_pruned": len(stage6),
        },
        "feature_stats": feature_stats,
        "decisions": decisions,
        "accepted_features": stage6,
        "model_feature_policy": dataset.get("model_feature_policy", {}),
        "training_eligible": False,
        "long_term_memory_write": False,
        "promotion_reason": "funnel_output_requires_downstream_ood_model_gate",
    }


def review_feature_funnel(
    funnel_report: dict[str, Any],
    *,
    review_scope: str = "PG-53 safe visible response projections only",
) -> dict[str, Any]:
    """Apply an explicit independent reviewer checklist to the funnel.

    This is the requested Codex review stage.  It is intentionally separate
    from feature scoring and training: a passing review only authorizes a
    downstream experiment, never a capability or memory promotion.
    """

    policy = dict(funnel_report.get("model_feature_policy") or {})
    stats = dict(funnel_report.get("feature_stats") or {})
    decisions = dict(funnel_report.get("decisions") or {})
    accepted = list(funnel_report.get("accepted_features") or [])
    checks = {
        "oracle_is_label_not_feature": policy.get("oracle_is_label_not_feature") is True,
        "source_seed_surface_audit_only": policy.get("source_seed_surface_are_audit_only") is True,
        "raw_request_response_forbidden": policy.get("raw_request_response_forbidden") is True,
        "hashes_evidence_only": policy.get("hashes_are_evidence_only") is True,
        "surface_observation_not_model_eligible": policy.get("surface_observation_model_eligible") is False,
        "generic_geometry_has_no_field_names": policy.get("generic_effect_geometry_no_field_names") is True,
        "accepted_features_have_stats": bool(accepted) and all(name in stats for name in accepted),
        "accepted_features_have_accept_decision": bool(accepted) and all(decisions.get(name, {}).get("decision") == "accepted" for name in accepted),
        "accepted_features_are_model_eligible": bool(accepted) and all(FEATURE_META.get(name, {}).get("model_eligible", True) for name in accepted),
        "accepted_features_have_low_source_shift": bool(accepted) and all(
            float(stats[name].get("source_leakage_nmi", 1.0)) <= 0.60
            and float(stats[name].get("source_distribution_shift", 1.0)) <= 0.50
            for name in accepted
        ),
        "no_unsafe_feature_name": not any(str(name).casefold() in {token.casefold() for token in FORBIDDEN_MODEL_FIELDS} for name in accepted),
        "multiple_sources": int(funnel_report.get("source_count", 0)) >= 2,
        "multiple_families": int(funnel_report.get("family_count", 0)) >= 3,
        "evaluation_only": funnel_report.get("training_eligible") is False and funnel_report.get("long_term_memory_write") is False,
    }
    passed = all(checks.values())
    review = {
        "reviewer_id": "codex-primary-feature-funnel-review-v1",
        "reviewer_role": "independent_feature_quality_and_leakage_auditor",
        "review_scope": str(review_scope),
        "decision": "approved_for_downstream_ood_experiment" if passed else "rejected_pending_funnel_repairs",
        "passed": passed,
        "checks": checks,
        "notes": [
            "审核只确认特征是否满足安全、可观测、跨来源和可复核要求。",
            "typed oracle/family/source/seed 只用于审核与评估，不进入模型特征。",
            "本审核不等于漏洞检测能力证明，也不批准长期记忆写入。",
        ],
        "training_allowed": False,
        "memory_promotion_allowed": False,
    }
    # Avoid circular imports and make the attestation reproducible.
    import hashlib
    import json

    review["review_evidence_sha256"] = hashlib.sha256(json.dumps(review, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return review


__all__ = [
    "FEATURE_META",
    "FORBIDDEN_MODEL_FIELDS",
    "FUNNEL_SCHEMA",
    "audit_feature_funnel",
    "build_feature_dataset",
    "build_feature_row",
    "review_feature_funnel",
]
