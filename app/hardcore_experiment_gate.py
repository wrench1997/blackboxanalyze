"""Hard fail-closed gate for non-self-deceptive local experiments."""

from __future__ import annotations

from typing import Any, Iterable

from app.cross_lab_safe_catalog import validate_sample


HARDCORE_POLICY_SCHEMA = "pg-pk-27-hardcore-evaluation-policy-v1"
PAYLOAD_FAMILIES_REQUIRING_GET_POST = frozenset({"xss", "sqli", "logic_access", "command_injection"})


def evaluate_hardcore_catalog(
    catalog: dict[str, Any],
    *,
    family: str,
    required_methods: Iterable[str] | None = None,
    min_target_instances: int = 3,
    min_seeds: int = 3,
    min_positive_replays: int = 2,
) -> dict[str, Any]:
    """Evaluate a Catalog against the PG-27 hard gates without running probes."""

    source = dict(catalog.get("source") or {})
    rows = [validate_sample(dict(row), source) for row in catalog.get("samples") or []]
    methods = {str((row.get("payload_manifest") or {}).get("method", "")).upper() for row in rows}
    targets = {str(row.get("target_instance_id", "")) for row in rows if row.get("target_instance_id")}
    seeds = {int(row.get("sampling_seed", -1)) for row in rows}
    positives = [row for row in rows if row["decision"]["evidence_status"] == "confirmed_positive"]
    negatives = [row for row in rows if row["decision"]["evidence_status"] == "confirmed_negative"]
    reasons: list[str] = []
    required = {str(method).upper() for method in (required_methods or ())}
    if not required and family in PAYLOAD_FAMILIES_REQUIRING_GET_POST:
        required = {"GET", "POST"}
    if not required.issubset(methods):
        reasons.append("missing_required_http_method")
    if len(targets) < int(min_target_instances):
        reasons.append("insufficient_target_instances")
    if len(seeds) < int(min_seeds):
        reasons.append("insufficient_sampling_seeds")
    if not positives:
        reasons.append("zero_typed_positives")
    for row in positives:
        oracle = dict(row.get("oracle_projection") or {})
        regex = dict((oracle.get("signals") or {}).get("regex_evidence") or {})
        if not bool(row["decision"].get("oracle_revalidated")):
            reasons.append("positive_not_revalidated")
        if not bool(regex.get("matched")):
            reasons.append("positive_without_regex_evidence")
        if not row.get("negative_control"):
            reasons.append("positive_without_negative_control")
        if not bool((row.get("reset") or {}).get("fresh_target")):
            reasons.append("positive_without_fresh_reset")
        if row["decision"].get("training_action") != "accept":
            reasons.append("positive_not_training_eligible")
    false_accepts = [row for row in negatives if row["decision"].get("false_positive")]
    if false_accepts:
        reasons.append("negative_false_accept")
    replay_counts: dict[str, int] = {}
    for row in positives:
        key = str((row.get("rule_ir") or {}).get("rule_key", "unknown"))
        replay_counts[key] = replay_counts.get(key, 0) + 1
    if positives and any(count < int(min_positive_replays) for count in replay_counts.values()):
        reasons.append("insufficient_positive_replays")
    reasons = sorted(set(reasons))
    if not reasons:
        status = "pass"
    elif any(reason in {"negative_false_accept", "positive_without_negative_control", "positive_without_fresh_reset", "positive_without_regex_evidence", "positive_not_revalidated"} for reason in reasons):
        status = "hard_fail"
    else:
        status = "preflight_only"
    return {
        "schema_version": HARDCORE_POLICY_SCHEMA,
        "catalog_id": str(catalog.get("catalog_id", "")),
        "source_id": str(source.get("source_id", "")),
        "family": family,
        "status": status,
        "methods_observed": sorted(methods),
        "required_methods": sorted(required),
        "target_instance_count": len(targets),
        "sampling_seed_count": len(seeds),
        "sample_count": len(rows),
        "typed_positive_count": len(positives),
        "confirmed_negative_count": len(negatives),
        "positive_replays": replay_counts,
        "reasons": reasons,
        "training_allowed": status == "pass" and bool(catalog.get("training_eligible")),
        "memory_promotion_allowed": status == "pass" and bool(catalog.get("training_eligible")),
    }


__all__ = ["HARDCORE_POLICY_SCHEMA", "evaluate_hardcore_catalog"]
