"""Fail-closed gate for claims that the model itself became stronger.

Engineering tests can prove that a runner is intact, but they cannot prove
better detection.  This gate therefore requires independent dataset tests,
source/target/seed separation, an explicit baseline, typed-oracle metrics and
an improvement on held-out data.  It returns no training or memory authority
unless every requirement is satisfied.
"""

from __future__ import annotations

import re
from typing import Any, Iterable


CAPABILITY_GATE_SCHEMA = "sift-model-capability-gate-v1"
_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ROLES = frozenset({"train", "dev", "family_holdout", "ood_source", "negative_control"})
_METRICS = frozenset({
    "typed_recall",
    "precision",
    "false_positive_rate",
    "abstain_precision",
    "ece",
    "median_queries",
})
_EMPTY_METRICS = {key: 0.0 for key in _METRICS}

DEFAULT_CAPABILITY_POLICY = {
    "min_distinct_dataset_ids": 3,
    "min_distinct_source_hashes": 3,
    "min_independent_seeds": 3,
    "min_target_instances": 3,
    "min_holdout_runs": 1,
    "required_roles": sorted(_ROLES),
    "min_holdout_recall_gain": 0.01,
    "max_precision_regression": 0.0,
    "max_false_positive_rate_regression": 0.0,
    "max_abstain_precision_regression": 0.0,
    "max_ece_regression": 0.0,
    "max_false_positive_count": 0,
    "require_worst_case_metrics": True,
    "require_cell_metrics": True,
}


def _normalise_policy(policy: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_CAPABILITY_POLICY)
    merged.update(dict(policy or {}))
    for key in (
        "min_distinct_dataset_ids",
        "min_distinct_source_hashes",
        "min_independent_seeds",
        "min_target_instances",
        "min_holdout_runs",
        "max_false_positive_count",
    ):
        merged[key] = max(0, int(merged[key]))
    for key in (
        "min_holdout_recall_gain",
        "max_precision_regression",
        "max_false_positive_rate_regression",
        "max_abstain_precision_regression",
        "max_ece_regression",
    ):
        merged[key] = max(0.0, float(merged[key]))
    required_roles = {str(role) for role in merged.get("required_roles", [])}
    if not required_roles.issubset(_ROLES):
        raise ValueError("capability policy contains an unknown dataset role")
    merged["required_roles"] = sorted(required_roles)
    return merged


def _bounded_id(value: Any, label: str) -> str:
    text = str(value)
    if not _ID_RE.fullmatch(text):
        raise ValueError(f"{label} must be a bounded identifier")
    return text


def _hash(value: Any, label: str) -> str:
    text = str(value).casefold()
    if not _HASH_RE.fullmatch(text):
        raise ValueError(f"{label} must be a canonical SHA-256 digest")
    return text


def _metrics(value: Any, label: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    missing = sorted(_METRICS - set(value))
    if missing:
        raise ValueError(f"{label} is missing metrics: {', '.join(missing)}")
    result: dict[str, float] = {}
    for key in _METRICS:
        number = float(value[key])
        if key == "median_queries":
            if number < 0:
                raise ValueError(f"{label}.{key} must be non-negative")
        elif not 0.0 <= number <= 1.0:
            raise ValueError(f"{label}.{key} must be in [0, 1]")
        result[key] = number
    return result


def _normalise_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalised: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise ValueError(f"dataset_tests[{index}] must be an object")
        role = _bounded_id(raw.get("role"), f"dataset_tests[{index}].role")
        if role not in _ROLES:
            raise ValueError(f"dataset_tests[{index}] has unknown role: {role}")
        seed = int(raw.get("sampling_seed", -1))
        if seed < 0:
            raise ValueError(f"dataset_tests[{index}].sampling_seed must be non-negative")
        families = [_bounded_id(item, f"dataset_tests[{index}].family_set") for item in raw.get("family_set", [])]
        if not families:
            raise ValueError(f"dataset_tests[{index}].family_set must be non-empty")
        if len(families) != len(set(families)):
            raise ValueError(f"dataset_tests[{index}].family_set contains duplicates")
        targets = [_bounded_id(item, f"dataset_tests[{index}].target_instance_ids") for item in raw.get("target_instance_ids", [])]
        if not targets:
            raise ValueError(f"dataset_tests[{index}].target_instance_ids must be non-empty")
        if len(targets) != len(set(targets)):
            raise ValueError(f"dataset_tests[{index}].target_instance_ids contains duplicates")
        target_instance_id = _bounded_id(raw.get("target_instance_id"), f"dataset_tests[{index}].target_instance_id")
        if target_instance_id not in targets:
            raise ValueError(f"dataset_tests[{index}].target_instance_id is not in target_instance_ids")
        sample_count = int(raw.get("sample_count", 0))
        unique_sample_count = int(raw.get("unique_sample_count", -1))
        if sample_count <= 0 or unique_sample_count != sample_count:
            raise ValueError(f"dataset_tests[{index}] has invalid sample identity counts")
        denominator = int(raw.get("denominator", 0))
        positive_count = int(raw.get("positive_count", 0))
        negative_count = int(raw.get("negative_count", 0))
        abstain_count = int(raw.get("abstain_count", 0))
        if denominator <= 0 or any(value < 0 or value > denominator for value in (positive_count, negative_count, abstain_count)):
            raise ValueError(f"dataset_tests[{index}] has invalid metric denominators")
        if str(raw.get("metrics_status", "")) != "completed":
            raise ValueError(f"dataset_tests[{index}] metrics are not completed")
        cell_baseline = _metrics(raw.get("baseline_metrics"), f"dataset_tests[{index}].baseline_metrics")
        cell_candidate = _metrics(raw.get("candidate_metrics"), f"dataset_tests[{index}].candidate_metrics")
        normalised.append({
            "sample_id": _bounded_id(raw.get("sample_id"), f"dataset_tests[{index}].sample_id"),
            "dataset_id": _bounded_id(raw.get("dataset_id"), f"dataset_tests[{index}].dataset_id"),
            "source_id": _bounded_id(raw.get("source_id"), f"dataset_tests[{index}].source_id"),
            "source_hash": _hash(raw.get("source_hash"), f"dataset_tests[{index}].source_hash"),
            "target_instance_id": target_instance_id,
            "target_instance_ids": sorted(set(targets)),
            "family_set": sorted(set(families)),
            "sampling_seed": seed,
            "role": role,
            "evidence_hash": _hash(raw.get("evidence_hash"), f"dataset_tests[{index}].evidence_hash"),
            "dataset_manifest_sha256": _hash(raw.get("dataset_manifest_sha256"), f"dataset_tests[{index}].dataset_manifest_sha256"),
            "split_manifest_sha256": _hash(raw.get("split_manifest_sha256"), f"dataset_tests[{index}].split_manifest_sha256"),
            "probe_sha256": _hash(raw.get("probe_sha256"), f"dataset_tests[{index}].probe_sha256"),
            "oracle_contract_sha256": _hash(raw.get("oracle_contract_sha256"), f"dataset_tests[{index}].oracle_contract_sha256"),
            "checkpoint_sha256": _hash(raw.get("checkpoint_sha256"), f"dataset_tests[{index}].checkpoint_sha256"),
            "sample_count": sample_count,
            "unique_sample_count": unique_sample_count,
            "denominator": denominator,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "abstain_count": abstain_count,
            # ``metrics`` is retained as the candidate-cell alias for older
            # consumers; the gate compares the explicit pair below.
            "metrics": _metrics(raw.get("metrics"), f"dataset_tests[{index}].metrics"),
            "baseline_metrics": cell_baseline,
            "candidate_metrics": cell_candidate,
        })
    return normalised


def _duplicate_values(rows: list[dict[str, Any]], key: str) -> bool:
    values = [row[key] for row in rows]
    return len(values) != len(set(values))


def _metric_reasons(
    baseline: dict[str, float],
    candidate: dict[str, float],
    requirements: dict[str, Any],
    *,
    prefix: str = "",
    require_recall_gain: bool = True,
) -> list[str]:
    label = f"{prefix}_" if prefix else ""
    reasons: list[str] = []
    if require_recall_gain and candidate["typed_recall"] - baseline["typed_recall"] < requirements["min_holdout_recall_gain"]:
        reasons.append(f"{label}holdout_recall_gain_below_threshold")
    if candidate["precision"] < baseline["precision"] - requirements["max_precision_regression"]:
        reasons.append(f"{label}precision_regression")
    if candidate["false_positive_rate"] > baseline["false_positive_rate"] + requirements["max_false_positive_rate_regression"]:
        reasons.append(f"{label}false_positive_rate_regression")
    if candidate["abstain_precision"] < baseline["abstain_precision"] - requirements["max_abstain_precision_regression"]:
        reasons.append(f"{label}abstain_precision_regression")
    if candidate["ece"] > baseline["ece"] + requirements["max_ece_regression"]:
        reasons.append(f"{label}calibration_regression")
    return reasons


def evaluate_model_capability(
    evidence: dict[str, Any],
    *,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate whether a candidate model has demonstrated real improvement."""

    if not isinstance(evidence, dict):
        raise ValueError("capability evidence must be an object")
    requirements = _normalise_policy(policy)
    rows = _normalise_rows(evidence.get("dataset_tests") or [])
    structural: list[str] = []
    if not rows:
        structural.append("no_dataset_tests")

    dataset_ids = {row["dataset_id"] for row in rows}
    source_hashes = {row["source_hash"] for row in rows}
    seeds = {row["sampling_seed"] for row in rows}
    targets = {target for row in rows for target in row["target_instance_ids"]}
    roles = {row["role"] for row in rows}
    evidence_hashes = [row["evidence_hash"] for row in rows]
    if len(evidence_hashes) != len(set(evidence_hashes)):
        structural.append("duplicate_evidence_hash")
    for key in ("sample_id", "probe_sha256", "dataset_manifest_sha256", "split_manifest_sha256"):
        if _duplicate_values(rows, key):
            structural.append("duplicate_" + key)
    if len(dataset_ids) < requirements["min_distinct_dataset_ids"]:
        structural.append("insufficient_distinct_dataset_ids")
    if len(source_hashes) < requirements["min_distinct_source_hashes"]:
        structural.append("insufficient_distinct_source_hashes")
    if len(seeds) < requirements["min_independent_seeds"]:
        structural.append("insufficient_independent_seeds")
    if len(targets) < requirements["min_target_instances"]:
        structural.append("insufficient_target_instances")
    missing_roles = sorted(set(requirements["required_roles"]) - roles)
    if missing_roles:
        structural.append("missing_required_roles:" + ",".join(missing_roles))
    holdout_runs = len({(row["dataset_id"], row["sampling_seed"]) for row in rows if row["role"] == "family_holdout"})
    if holdout_runs < requirements["min_holdout_runs"]:
        structural.append("insufficient_holdout_runs")

    for role in requirements["required_roles"]:
        role_seeds = {row["sampling_seed"] for row in rows if row["role"] == role}
        if len(role_seeds) < requirements["min_independent_seeds"]:
            structural.append(f"role_{role}_insufficient_independent_seeds")

    train_rows = [row for row in rows if row["role"] in {"train", "dev"}]
    evaluation_rows = [row for row in rows if row["role"] in {"family_holdout", "ood_source", "negative_control"}]
    train_targets = {target for row in train_rows for target in row["target_instance_ids"]}
    evaluation_targets = {target for row in evaluation_rows for target in row["target_instance_ids"]}
    if train_targets & evaluation_targets:
        structural.append("train_eval_target_overlap")
    train_families = {family for row in train_rows for family in row["family_set"]}
    evaluation_families = {family for row in evaluation_rows for family in row["family_set"]}
    if train_families & evaluation_families:
        structural.append("train_eval_family_overlap")
    train_split_manifests = {row["split_manifest_sha256"] for row in train_rows}
    evaluation_split_manifests = {row["split_manifest_sha256"] for row in evaluation_rows}
    if train_split_manifests & evaluation_split_manifests:
        structural.append("train_eval_split_manifest_overlap")
    train_source_ids = {row["source_id"] for row in train_rows}
    evaluation_source_ids = {row["source_id"] for row in evaluation_rows}
    if train_source_ids & evaluation_source_ids:
        structural.append("train_eval_source_id_overlap")

    train_sources = {row["source_hash"] for row in rows if row["role"] in {"train", "dev"}}
    evaluation_sources = {row["source_hash"] for row in rows if row["role"] in {"family_holdout", "ood_source", "negative_control"}}
    if train_sources & evaluation_sources:
        structural.append("train_eval_source_overlap")
    if not bool(evidence.get("unit_tests_passed", False)):
        structural.append("unit_tests_not_passed")
    if not bool(evidence.get("oracle_validated", False)):
        structural.append("oracle_not_validated")
    if not bool(evidence.get("data_lineage_complete", False)):
        structural.append("data_lineage_incomplete")
    if not bool(evidence.get("authorized_sources_attested", False)):
        structural.append("dataset_authorization_missing")
    if bool(evidence.get("raw_data_retained", False)):
        structural.append("raw_data_retained")

    try:
        baseline = _metrics(evidence.get("baseline_metrics"), "baseline_metrics")
    except (TypeError, ValueError):
        structural.append("baseline_metrics_missing_or_invalid")
        baseline = dict(_EMPTY_METRICS)
    try:
        candidate = _metrics(evidence.get("candidate_metrics"), "candidate_metrics")
    except (TypeError, ValueError):
        structural.append("candidate_metrics_missing_or_invalid")
        candidate = dict(_EMPTY_METRICS)
    false_positive_count = int(evidence.get("false_positive_count", -1))
    if false_positive_count < 0:
        structural.append("false_positive_count_missing")

    metric_reasons: list[str] = _metric_reasons(baseline, candidate, requirements)

    if requirements.get("require_worst_case_metrics", True):
        try:
            baseline_worst = _metrics(evidence.get("baseline_worst_case_metrics"), "baseline_worst_case_metrics")
            candidate_worst = _metrics(evidence.get("candidate_worst_case_metrics"), "candidate_worst_case_metrics")
        except (TypeError, ValueError):
            structural.append("worst_case_metrics_missing_or_invalid")
            baseline_worst = dict(_EMPTY_METRICS)
            candidate_worst = dict(_EMPTY_METRICS)
        metric_reasons.extend(_metric_reasons(baseline_worst, candidate_worst, requirements, prefix="worst_case"))

    if requirements.get("require_cell_metrics", True):
        for row in rows:
            metric_reasons.extend(_metric_reasons(
                row["baseline_metrics"], row["candidate_metrics"], requirements,
                prefix=f"cell_{row['sample_id']}",
                require_recall_gain=row["role"] in {"family_holdout", "ood_source"},
            ))
    if false_positive_count > requirements["max_false_positive_count"]:
        metric_reasons.append("false_positive_ledger_nonzero")

    reasons = sorted(set(structural + metric_reasons))
    status = "blocked" if structural else "no_proven_gain" if metric_reasons else "pass"
    return {
        "schema_version": CAPABILITY_GATE_SCHEMA,
        "claim_id": _bounded_id(evidence.get("claim_id", "capability-evaluation"), "claim_id"),
        "status": status,
        "claim_allowed": status == "pass",
        "training_allowed": status == "pass",
        "memory_promotion_allowed": status == "pass",
        "unit_tests_are_not_capability_evidence": True,
        "reasons": reasons,
        "requirements": requirements,
        "summary": {
            "dataset_test_count": len(rows),
            "distinct_dataset_count": len(dataset_ids),
            "distinct_source_hash_count": len(source_hashes),
            "independent_seed_count": len(seeds),
            "target_instance_count": len(targets),
            "holdout_run_count": holdout_runs,
            "roles_observed": sorted(roles),
            "false_positive_count": false_positive_count,
        },
        "baseline_metrics": baseline,
        "candidate_metrics": candidate,
        "data_tests_are_required_for_improvement_claim": True,
        "raw_data_retained": False,
    }


__all__ = ["CAPABILITY_GATE_SCHEMA", "DEFAULT_CAPABILITY_POLICY", "evaluate_model_capability"]
