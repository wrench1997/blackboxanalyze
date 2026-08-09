"""Fail-closed promotion rule for long-term abstract Rule Memory.

An observation can guide the current episode after one success, but it cannot
become durable memory until the same rule survives multiple authorized source
hashes, datasets, sampling seeds, and target instances.  The gate consumes
only evaluator-side summaries and evidence hashes; it never exposes labels to
the probe selector.  Dataset labels and seed labels are not treated as
independent evidence when their source or evidence manifests are reused.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


PROMOTION_SCHEMA = "sift-memory-promotion-gate-v1"
DEFAULT_PROMOTION_POLICY = {
    "min_distinct_datasets": 3,
    "min_distinct_source_hashes": 3,
    "min_sampling_seeds_per_dataset": 2,
    "min_target_instances": 3,
    "min_observations_per_dataset": 2,
    "min_accepted_per_dataset": 1,
    "max_false_positive_rate": 0.0,
    "require_evidence_hash": True,
    "require_local_only": True,
    "require_oracle_revalidation": True,
}


def _normalise_policy(policy: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_PROMOTION_POLICY)
    merged.update(dict(policy or {}))
    for key in (
        "min_distinct_datasets",
        "min_distinct_source_hashes",
        "min_sampling_seeds_per_dataset",
        "min_target_instances",
        "min_observations_per_dataset",
        "min_accepted_per_dataset",
    ):
        merged[key] = max(1, int(merged[key]))
    merged["max_false_positive_rate"] = min(1.0, max(0.0, float(merged["max_false_positive_rate"])))
    merged["require_evidence_hash"] = bool(merged["require_evidence_hash"])
    merged["require_local_only"] = bool(merged["require_local_only"])
    merged["require_oracle_revalidation"] = bool(merged["require_oracle_revalidation"])
    return merged


def assess_memory_promotion(
    rule_key: str,
    evaluations: Iterable[dict[str, Any]],
    *,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return ``promote`` only when cross-dataset evidence clears every gate."""

    rule_key = str(rule_key)
    if not rule_key:
        raise ValueError("memory promotion rule_key must not be empty")
    requirements = _normalise_policy(policy)
    rows = [dict(row) for row in evaluations]
    reasons: list[str] = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_targets: set[str] = set()
    evidence_hashes: set[str] = set()
    source_hashes: set[str] = set()
    evidence_sources: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(rows):
        dataset = str(row.get("dataset_id", ""))
        target = str(row.get("target_instance_id", ""))
        row_rule = str(row.get("rule_key", rule_key))
        source_hash = str(row.get("source_hash", row.get("fixture_source_sha256", "")))
        if not dataset:
            reasons.append(f"row_{index}:missing_dataset_id")
        if not target:
            reasons.append(f"row_{index}:missing_target_instance_id")
        if row_rule != rule_key:
            reasons.append(f"row_{index}:rule_key_mismatch")
        if requirements["require_local_only"] and not bool(row.get("local_only", False)):
            reasons.append(f"row_{index}:not_local_only")
        if requirements.get("min_distinct_source_hashes", 0) > 0 and len(source_hash) < 16:
            reasons.append(f"row_{index}:missing_source_hash")
        evidence_hash = str(row.get("evidence_hash", ""))
        if requirements["require_evidence_hash"] and len(evidence_hash) < 16:
            reasons.append(f"row_{index}:missing_evidence_hash")
        # A positive row may become durable memory only after a family-specific
        # oracle has revalidated it.  Negative controls remain valid coverage
        # observations and do not need a positive oracle flag.
        if requirements["require_oracle_revalidation"] and bool(row.get("accepted", False)) and not bool(row.get("oracle_revalidated", False)):
            reasons.append(f"row_{index}:missing_oracle_revalidation")
        if evidence_hash:
            evidence_hashes.add(evidence_hash)
            if source_hash:
                evidence_sources[evidence_hash].add(source_hash)
        if source_hash:
            source_hashes.add(source_hash)
        if target:
            seen_targets.add(target)
        if dataset:
            groups[dataset].append(row)
    if not rows:
        reasons.append("no_evaluations")
    if len(groups) < requirements["min_distinct_datasets"]:
        reasons.append("insufficient_distinct_datasets")
    if len(source_hashes) < requirements["min_distinct_source_hashes"]:
        reasons.append("insufficient_distinct_source_hashes")
    reused_evidence = sorted(hash_value for hash_value, sources in evidence_sources.items() if len(sources) > 1)
    if reused_evidence:
        reasons.append("evidence_hash_reused_across_sources")
    if len(seen_targets) < requirements["min_target_instances"]:
        reasons.append("insufficient_target_instances")
    per_dataset: dict[str, dict[str, Any]] = {}
    for dataset, group in sorted(groups.items()):
        seeds = {str(row.get("sampling_seed", "")) for row in group if str(row.get("sampling_seed", ""))}
        by_evidence: dict[str, dict[str, Any]] = {}
        duplicate_conflicts = False
        for row_index, row in enumerate(group):
            evidence_hash = str(row.get("evidence_hash", ""))
            key = evidence_hash or f"row:{row_index}"
            previous = by_evidence.get(key)
            if previous is not None:
                previous_signature = (
                    bool(previous.get("accepted", False)),
                    bool(previous.get("false_positive", False)),
                    str(previous.get("source_hash", previous.get("fixture_source_sha256", ""))),
                )
                current_signature = (
                    bool(row.get("accepted", False)),
                    bool(row.get("false_positive", False)),
                    str(row.get("source_hash", row.get("fixture_source_sha256", ""))),
                )
                duplicate_conflicts = duplicate_conflicts or previous_signature != current_signature
                continue
            by_evidence[key] = row
        unique_group = list(by_evidence.values())
        false_positives = sum(bool(row.get("false_positive", False)) for row in unique_group)
        accepted = sum(bool(row.get("accepted", False)) for row in unique_group)
        attempts = len(unique_group)
        false_positive_rate = false_positives / attempts if attempts else 1.0
        dataset_reasons: list[str] = []
        if duplicate_conflicts:
            dataset_reasons.append("conflicting_duplicate_evidence")
        if attempts < requirements["min_observations_per_dataset"]:
            dataset_reasons.append("insufficient_observations")
        if len(seeds) < requirements["min_sampling_seeds_per_dataset"]:
            dataset_reasons.append("insufficient_sampling_seeds")
        seed_manifests: dict[str, tuple[str, ...]] = {}
        for seed in sorted(seeds):
            manifest = tuple(sorted({str(row.get("evidence_hash", "")) for row in group if str(row.get("sampling_seed", "")) == seed and str(row.get("evidence_hash", ""))}))
            seed_manifests[seed] = manifest
        distinct_seed_manifests = len(set(seed_manifests.values()))
        if len(seeds) >= requirements["min_sampling_seeds_per_dataset"] and distinct_seed_manifests < requirements["min_sampling_seeds_per_dataset"]:
            dataset_reasons.append("insufficient_independent_seed_evidence")
        if false_positive_rate > requirements["max_false_positive_rate"]:
            dataset_reasons.append("false_positive_rate_exceeded")
        if accepted < requirements["min_accepted_per_dataset"]:
            dataset_reasons.append("insufficient_accepted_evidence")
        if dataset_reasons:
            reasons.extend(f"{dataset}:{reason}" for reason in dataset_reasons)
        per_dataset[dataset] = {
            "observation_count": attempts,
            "sampling_seed_count": len(seeds),
            "distinct_seed_evidence_manifest_count": distinct_seed_manifests,
            "target_instance_count": len({str(row.get("target_instance_id", "")) for row in group if row.get("target_instance_id")} ),
            "false_positive_count": false_positives,
            "false_positive_rate": false_positive_rate,
            "accepted_count": accepted,
            "distinct_evidence_hash_count": len({str(row.get("evidence_hash", "")) for row in unique_group if row.get("evidence_hash")}),
            "reasons": dataset_reasons,
        }
    unique_reasons = sorted(set(reasons))
    return {
        "schema_version": PROMOTION_SCHEMA,
        "rule_key": rule_key,
        "status": "promote" if not unique_reasons else "quarantine",
        "promote": not unique_reasons,
        "reasons": unique_reasons,
        "requirements": requirements,
        "summary": {
            "evaluation_count": len(rows),
            "distinct_dataset_count": len(groups),
            "distinct_source_hash_count": len(source_hashes),
            "distinct_target_instance_count": len(seen_targets),
            "distinct_evidence_hash_count": len(evidence_hashes),
        },
        "per_dataset": per_dataset,
    }


__all__ = ["DEFAULT_PROMOTION_POLICY", "PROMOTION_SCHEMA", "assess_memory_promotion"]
