"""Uniform cross-dataset memory-promotion audit and replay queue."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .memory_promotion_gate import assess_memory_promotion


PROMOTION_RUNNER_SCHEMA = "sift-promotion-runner-v1"


def provenance_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize lineage without copying raw probes or response bodies."""

    values = [row for row in rows if isinstance(row, dict)]
    source_hashes: set[str] = set()
    targets: set[str] = set()
    datasets: set[str] = set()
    seeds: set[str] = set()
    evidence_hashes: set[str] = set()
    for row in values:
        datasets.add(str(row.get("dataset_id", "")))
        targets.add(str(row.get("target_instance_id", "")))
        seed = str(row.get("sampling_seed", ""))
        if seed:
            seeds.add(seed)
        evidence = str(row.get("evidence_hash", ""))
        if evidence:
            evidence_hashes.add(evidence)
        source = str(row.get("source_hash", row.get("fixture_source_sha256", "")))
        if source:
            source_hashes.add(source)
    return {
        "row_count": len(values),
        "dataset_ids": sorted(item for item in datasets if item),
        "target_instance_ids": sorted(item for item in targets if item),
        "sampling_seeds": sorted(seeds),
        "source_hashes": sorted(source_hashes),
        "evidence_hash_count": len(evidence_hashes),
    }


def run_promotion_audit(
    ledger_rows: Iterable[dict[str, Any]],
    *,
    rule_keys: Iterable[str],
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the same promotion gate for several rule keys and queue failures."""

    rows = [dict(row) for row in ledger_rows if isinstance(row, dict)]
    results: dict[str, dict[str, Any]] = {}
    replay_queue: list[dict[str, Any]] = []
    for rule_key in sorted({str(key) for key in rule_keys if str(key)}):
        scoped = [row for row in rows if str(row.get("rule_key", "")) == rule_key]
        result = assess_memory_promotion(rule_key, scoped, policy=policy)
        results[rule_key] = result
        if result["status"] != "promote":
            reasons = list(result.get("reasons", []))
            for dataset_id, per_dataset in result.get("per_dataset", {}).items():
                if per_dataset.get("reasons"):
                    replay_queue.append({
                        "rule_key": rule_key,
                        "dataset_id": dataset_id,
                        "target_instance_ids": sorted({str(row.get("target_instance_id", "")) for row in scoped if str(row.get("dataset_id", "")) == dataset_id}),
                        "reasons": list(per_dataset["reasons"]),
                        "action": "fresh_reset_replay_before_memory_write",
                    })
            if not replay_queue or not any(item["rule_key"] == rule_key for item in replay_queue):
                replay_queue.append({
                    "rule_key": rule_key,
                    "dataset_id": None,
                    "target_instance_ids": sorted({str(row.get("target_instance_id", "")) for row in scoped}),
                    "reasons": reasons,
                    "action": "fresh_reset_replay_before_memory_write",
                })
    return {
        "schema_version": PROMOTION_RUNNER_SCHEMA,
        "rule_keys": sorted(results),
        "promotion": results,
        "replay_queue": replay_queue,
        "provenance": provenance_summary(rows),
        "all_promoted": bool(results) and all(result["status"] == "promote" for result in results.values()),
        "memory_write_allowed": bool(results) and all(result["status"] == "promote" for result in results.values()),
    }


__all__ = ["PROMOTION_RUNNER_SCHEMA", "provenance_summary", "run_promotion_audit"]
