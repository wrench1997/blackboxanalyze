"""PG-326: read-only cross-implementation stability/forgetting matrix.

This command never starts Docker and never contacts a target.  It normalizes
the already-frozen PG-323/324/325 replay artifacts into one matrix so that a
high score on one report cannot hide a missing observation contract in
another.  A forgetting result is intentionally *not* fabricated: it requires
the same canaries replayed before and after an actual checkpoint update.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
REPORT = RESEARCH / "pg326_cross_impl_forgetting_matrix_v1.json"
PROTOCOL = RESEARCH / "pg326_cross_impl_forgetting_matrix_protocol_v1.json"
FORGETTING_REPORT = RESEARCH / "pg327b_paired_fresh_replay_report_v1.json"
FORGETTING_AUDIT = RESEARCH / "pg327b_paired_fresh_replay_audit_v1.json"
MODEL_CONTEXT_KEYS = frozenset(
    {
        "typed_available",
        "feedback_state",
        "replay_ready",
        "evidence_present",
        "negative_control",
        "fresh_reset",
        "surface_method",
        "surface_field_role",
        "surface_encoding",
        "history_action",
        "failure_class",
        "step_budget",
    }
)

SOURCES: tuple[dict[str, str], ...] = (
    {
        "id": "pg323",
        "report": "pg323_vulnerableapp_role_replay_report_v1.json",
        "catalog": "pg323_vulnerableapp_role_catalog_v1.json",
        "trace": "pg323_vulnerableapp_role_trace_v1.json",
        "protocol": "pg323_vulnerableapp_role_protocol_v1.json",
    },
    {
        "id": "pg324",
        "report": "pg324_juice_shop_source_heldout_report_v1.json",
        "catalog": "pg324_juice_shop_source_heldout_catalog_v1.json",
        "trace": "pg324_juice_shop_source_heldout_trace_v1.json",
        "protocol": "pg324_juice_shop_source_heldout_protocol_v1.json",
    },
    {
        "id": "pg325",
        "report": "pg325_sql_family_holdout_report_v1.json",
        "catalog": "pg325_sql_family_holdout_catalog_v1.json",
        "trace": "pg325_sql_family_holdout_trace_v1.json",
        "protocol": "pg325_sql_family_holdout_protocol_v1.json",
    },
)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read(name: str) -> dict[str, Any]:
    path = RESEARCH / name
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _rate(numerator: Any, denominator: Any) -> float | None:
    try:
        den = int(denominator)
        num = int(numerator)
    except (TypeError, ValueError):
        return None
    if den <= 0:
        return None
    return round(num / den, 6)


def _bool_or_none(mapping: Mapping[str, Any], key: str) -> bool | None:
    if key not in mapping:
        return None
    value = mapping.get(key)
    return value if isinstance(value, bool) else bool(value)


def _same_hash(document: Mapping[str, Any], field: str) -> bool:
    expected = str(document.get(field, ""))
    if len(expected) != 64:
        return False
    clone = dict(document)
    clone[field] = ""
    return _digest(clone) == expected


def _families(catalog: Mapping[str, Any]) -> list[str]:
    result: set[str] = set()
    for entry in list(catalog.get("entries") or []):
        if not isinstance(entry, Mapping):
            continue
        route = entry.get("route") if isinstance(entry.get("route"), Mapping) else {}
        value = entry.get("family_split") or route.get("family")
        if value:
            result.add(str(value))
    return sorted(result)


def _context_token_ok(context: Any) -> bool:
    if not isinstance(context, list) or not context:
        return False
    for token in context:
        text = str(token)
        if text in {"[BOS]", "[EOS]"}:
            continue
        if "=" not in text or text.split("=", 1)[0] not in MODEL_CONTEXT_KEYS:
            return False
    return True


def _derived_context_firewall(catalog: Mapping[str, Any], trace: Mapping[str, Any]) -> bool:
    contexts: list[Any] = []
    for row in list(catalog.get("entries") or []):
        if not isinstance(row, Mapping):
            return False
        model = row.get("model") if isinstance(row.get("model"), Mapping) else {}
        contexts.extend(
            entry.get("context_tokens")
            for entry in list(model.get("entries") or [])
            if isinstance(entry, Mapping)
        )
        contexts.append(model.get("failure_context"))
    contexts.extend(
        row.get("context_tokens")
        for row in list(trace.get("episodes") or [])
        if isinstance(row, Mapping)
    )
    return bool(contexts) and all(_context_token_ok(context) for context in contexts)


def _row(source: Mapping[str, str]) -> tuple[dict[str, Any], list[str]]:
    report = _read(str(source["report"]))
    catalog = _read(str(source["catalog"]))
    trace = _read(str(source["trace"]))
    protocol = _read(str(source["protocol"]))
    missing: list[str] = []
    if not report:
        missing.append("report")
    if not catalog:
        missing.append("catalog")
    if not trace:
        missing.append("trace")
    if not protocol:
        missing.append("protocol")

    counts = report.get("counts") if isinstance(report.get("counts"), Mapping) else {}
    worst = report.get("worst_seed_metrics") if isinstance(report.get("worst_seed_metrics"), Mapping) else {}
    checks = report.get("checks") if isinstance(report.get("checks"), Mapping) else {}
    promotion = report.get("promotion") if isinstance(report.get("promotion"), Mapping) else {}
    runtime = report.get("runtime") if isinstance(report.get("runtime"), Mapping) else {}
    model = report.get("model") if isinstance(report.get("model"), Mapping) else {}
    entries = catalog.get("entries") if isinstance(catalog.get("entries"), list) else []
    source_hashes = sorted(
        {
            str((entry.get("target") or {}).get("source_sha256"))
            for entry in entries
            if isinstance(entry, Mapping)
            and isinstance(entry.get("target"), Mapping)
            and len(str(entry.get("target", {}).get("source_sha256", ""))) == 64
        }
    )
    image = str(catalog.get("implementation") or runtime.get("image") or "")
    if not image:
        missing.append("implementation_digest")

    action_changed = counts.get("failure_action_changed_count")
    action_required = counts.get("failure_transition_required_count")
    action_rate = _rate(action_changed, action_required)
    if action_rate is None:
        missing.append("failure_action_change_contract")

    checks_map = {
        "fresh_reset": checks.get("fresh_container_per_route_seed", checks.get("fresh_reset_per_route")),
        "get_post_pair": checks.get("get_post_pair"),
        "typed_evidence": checks.get("typed_evidence_hash_per_route"),
        "context_firewall": checks.get("model_context_firewall"),
        "role_bound_belief_evidence": checks.get("belief_role_bound_evidence"),
        "failure_action_changed": checks.get("failure_action_changed_all"),
        "network_none": checks.get("docker_network_none"),
        "external_network_disabled": checks.get("external_network_disabled"),
        "raw_payload_excluded": (checks.get("raw_payload_in_model_context") is False) if checks.get("raw_payload_in_model_context") is not None else None,
        "raw_response_excluded": (checks.get("raw_response_bodies_stored") is False) if checks.get("raw_response_bodies_stored") is not None else None,
    }
    if checks_map["context_firewall"] is None:
        checks_map["context_firewall"] = _derived_context_firewall(catalog, trace)
    for name, value in checks_map.items():
        if value is None:
            missing.append(f"check:{name}")

    if report.get("schema_version") not in {
        "pg323-vulnerableapp-role-replay-report-v1",
        "pg324-juice-shop-source-heldout-report-v2",
        "pg325-sql-family-holdout-report-v1",
    }:
        missing.append("report_schema")
    for name, document, field in (
        ("report", report, "report_sha256"),
        ("catalog", catalog, "catalog_sha256"),
        ("trace", trace, "trace_sha256"),
        ("protocol", protocol, "protocol_sha256"),
    ):
        if not _same_hash(document, field):
            missing.append(f"hash:{name}")

    row = {
        "id": str(source["id"]),
        "report": str(source["report"]),
        "implementation_digest": image,
        "families": _families(catalog),
        "source_sha256_count": len(source_hashes),
        "seed_count": int(counts.get("seed_count", 0) or 0),
        "route_count": int(counts.get("route_count", 0) or 0),
        "get_count": int(counts.get("get_count", 0) or 0),
        "post_count": int(counts.get("post_count", 0) or 0),
        "typed_effect_rate": _rate(counts.get("positive_typed_effect_count"), counts.get("positive_route_count")),
        "variant_exact_rate": _rate(counts.get("variant_exact_count"), counts.get("variant_role_count")),
        "ask_recall": float(worst.get("multi_missing_question_recall_min")) if isinstance(worst.get("multi_missing_question_recall_min"), (int, float)) else None,
        "repair_rate": float(worst.get("failure_repair_rate_min")) if isinstance(worst.get("failure_repair_rate_min"), (int, float)) else None,
        "failure_action_change_rate": action_rate,
        "negative_violation_count": int(counts.get("negative_lane_violation_count", 0) or 0),
        "belief_duplicate_evidence_count": int(counts.get("belief_duplicate_evidence_count", 0) or 0),
        "checks": {key: value for key, value in checks_map.items()},
        "promotion": {
            "training_allowed": promotion.get("training_allowed") is True,
            "memory_promotion_allowed": promotion.get("memory_promotion_allowed") is True,
            "vulnerability_claim_allowed": promotion.get("vulnerability_claim_allowed") is True,
        },
        "checkpoint_family": str(model.get("checkpoint_family", "")),
        "raw_material_available": False,
        "missing_requirements": sorted(set(missing)),
    }
    return row, sorted(set(missing))


def build_matrix() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    source_missing: dict[str, list[str]] = {}
    for source in SOURCES:
        row, missing = _row(source)
        rows.append(row)
        source_missing[str(source["id"])] = missing

    implementations = sorted({row["implementation_digest"] for row in rows if row["implementation_digest"]})
    families = sorted({family for row in rows for family in row["families"]})
    totals = {
        "seed_count": sum(row["seed_count"] for row in rows),
        "route_count": sum(row["route_count"] for row in rows),
        "get_count": sum(row["get_count"] for row in rows),
        "post_count": sum(row["post_count"] for row in rows),
        "positive_typed_effect_count": sum(
            int(_read(str(source["report"])).get("counts", {}).get("positive_typed_effect_count", 0) or 0)
            for source in SOURCES
        ),
        "positive_route_count": sum(
            int(_read(str(source["report"])).get("counts", {}).get("positive_route_count", 0) or 0)
            for source in SOURCES
        ),
        "variant_exact_count": sum(
            int(_read(str(source["report"])).get("counts", {}).get("variant_exact_count", 0) or 0)
            for source in SOURCES
        ),
        "variant_role_count": sum(
            int(_read(str(source["report"])).get("counts", {}).get("variant_role_count", 0) or 0)
            for source in SOURCES
        ),
        "multi_missing_question_rows": sum(
            int(_read(str(source["report"])).get("counts", {}).get("multi_missing_question_rows", 0) or 0)
            for source in SOURCES
        ),
        "failure_repair_correct_count": sum(
            int(_read(str(source["report"])).get("counts", {}).get("failure_repair_correct_count", 0) or 0)
            for source in SOURCES
        ),
        "failure_repair_count": sum(
            int(_read(str(source["report"])).get("counts", {}).get("failure_repair_count", 0) or 0)
            for source in SOURCES
        ),
        "negative_lane_violation_count": sum(row["negative_violation_count"] for row in rows),
    }

    numeric_min = lambda key: min(
        (float(row[key]) for row in rows if isinstance(row.get(key), (int, float))),
        default=None,
    )
    numeric_max = lambda key: max(
        (float(row[key]) for row in rows if isinstance(row.get(key), (int, float))),
        default=None,
    )
    # Missing fields are not silently converted into passes.  This is why the
    # matrix can show perfect observed behavior while remaining blocked.
    uniform_checks = {
        "fresh_reset": all(row["checks"].get("fresh_reset") is True for row in rows),
        "get_post_pair": all(row["checks"].get("get_post_pair") is True for row in rows),
        "typed_evidence": all(row["checks"].get("typed_evidence") is True for row in rows),
        "context_firewall": all(row["checks"].get("context_firewall") is True for row in rows),
        "role_bound_belief_evidence": all(row["checks"].get("role_bound_belief_evidence") is True for row in rows),
        "failure_action_changed": all(row["checks"].get("failure_action_changed") is True for row in rows),
        "network_none": all(row["checks"].get("network_none") is True for row in rows),
        "external_network_disabled": all(row["checks"].get("external_network_disabled") is True for row in rows),
        "raw_payload_excluded": all(row["checks"].get("raw_payload_excluded") is True for row in rows),
        "raw_response_excluded": all(row["checks"].get("raw_response_excluded") is True for row in rows),
    }
    forgetting_doc = _read(FORGETTING_REPORT.name)
    forgetting_audit = _read(FORGETTING_AUDIT.name)
    forgetting = {
        "status": "not_run",
        "paired_replay_present": False,
        "before_checkpoint": "",
        "after_checkpoint": "",
        "same_canary_route_set": False,
        "reason": "PG-327B paired fresh replay report/audit 未通过或不存在。",
    }
    if (
        forgetting_doc.get("schema_version") == "pg327b-paired-fresh-replay-report-v1"
        and forgetting_doc.get("status") == "completed_local_docker_pg327b_paired_replay"
        and forgetting_audit.get("status") == "passed"
        and bool((forgetting_doc.get("forgetting") or {}).get("paired_replay_present"))
        and bool((forgetting_doc.get("forgetting") or {}).get("same_canary_route_set"))
        and bool((forgetting_doc.get("checks") or {}).get("before_after_checkpoint_distinct"))
    ):
        forgetting = {
            "status": "completed_paired_fresh_replay",
            "paired_replay_present": True,
            "before_checkpoint": dict((forgetting_doc.get("forgetting") or {}).get("before_checkpoint") or {}),
            "after_checkpoint": dict((forgetting_doc.get("forgetting") or {}).get("after_checkpoint") or {}),
            "same_canary_route_set": True,
            "paired_report": FORGETTING_REPORT.name,
            "paired_audit": FORGETTING_AUDIT.name,
            "paired_report_sha256": str(forgetting_doc.get("report_sha256", "")),
            "paired_audit_sha256": str(forgetting_audit.get("audit_sha256", "")),
            "pairs": list((forgetting_doc.get("forgetting") or {}).get("pairs") or []),
        }
    missing = sorted({item for values in source_missing.values() for item in values if item})
    if not forgetting["paired_replay_present"]:
        missing.append("forgetting:paired_before_after_replay")
    missing = sorted(set(missing))
    hard_gate = {
        "source_reports_complete": all(not values for values in source_missing.values()),
        "independent_implementation_count": len(implementations) >= 2,
        "family_diversity": len(families) >= 2,
        "worst_seed_typed_effect": (numeric_min("typed_effect_rate") or 0.0) >= 0.95,
        "worst_seed_variant": (numeric_min("variant_exact_rate") or 0.0) >= 0.90,
        "worst_seed_ask": (numeric_min("ask_recall") or 0.0) >= 0.95,
        "worst_seed_repair": (numeric_min("repair_rate") or 0.0) >= 0.90,
        "negative_zero": totals["negative_lane_violation_count"] == 0,
        "uniform_observation_contract": all(uniform_checks.values()),
        "forgetting_pair": forgetting["paired_replay_present"],
        "promotion_blocked": True,
    }
    report: dict[str, Any] = {
        "protocol_id": "pg-pk-326-cross-implementation-forgetting-matrix-v1",
        "schema_version": "pg326-cross-implementation-forgetting-matrix-report-v1",
        "status": "completed_read_only_matrix_blocked",
        "runtime": {"target_contacted": False, "docker_started": False, "network": "none_not_used", "source_count": len(SOURCES)},
        "scope": {"sources": [source["id"] for source in SOURCES], "frozen_checkpoint_family": "PG-323", "paired_forgetting_report": FORGETTING_REPORT.name, "matrix_is_evaluation_only": True},
        "implementation_digests": implementations,
        "families": families,
        "family_counts": dict(Counter(family for row in rows for family in row["families"])),
        "source_rows": rows,
        "totals": totals,
        "worst_seed_metrics": {
            "typed_effect_rate_min": numeric_min("typed_effect_rate"),
            "variant_exact_rate_min": numeric_min("variant_exact_rate"),
            "ask_recall_min": numeric_min("ask_recall"),
            "repair_rate_min": numeric_min("repair_rate"),
            "failure_action_change_rate_min": numeric_min("failure_action_change_rate"),
            "negative_violation_max": numeric_max("negative_violation_count"),
        },
        "uniform_checks": uniform_checks,
        "forgetting": forgetting,
        "missing_requirements": missing,
        "hypothesis_gate": {"status": "blocked", "checks": hard_gate, "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False, "checkpoint_role": "research_candidate_only"},
        "report_sha256": "",
    }
    report["report_sha256"] = _digest(report)
    protocol: dict[str, Any] = {
        "protocol_id": report["protocol_id"],
        "schema_version": "pg326-cross-implementation-forgetting-matrix-protocol-v1",
        "scope": {"read_only": True, "target_contacted": False, "inputs": [source["report"] for source in SOURCES] + [FORGETTING_REPORT.name, FORGETTING_AUDIT.name]},
        "required_gates": {
            "independent_implementation_count": True,
            "family_diversity": True,
            "worst_seed_typed_effect": True,
            "worst_seed_variant": True,
            "worst_seed_ask": True,
            "worst_seed_repair": True,
            "negative_zero": True,
            "uniform_observation_contract": True,
            "forgetting_pair": True,
            "raw_payload_training_excluded": True,
            "promotion_blocked": True,
        },
        "forbidden": ["docker_start", "public_target", "external_network", "payload_promotion", "memory_promotion"],
        "promotion": dict(report["promotion"]),
        "protocol_sha256": "",
    }
    protocol["protocol_sha256"] = _digest(protocol)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    report = build_matrix()
    print(json.dumps({"status": report["status"], "totals": report["totals"], "worst_seed_metrics": report["worst_seed_metrics"], "hard_gate": report["hypothesis_gate"], "missing_requirements": report["missing_requirements"], "report": str(REPORT.relative_to(ROOT)), "target_contacted": report["runtime"]["target_contacted"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
