"""Read-only audit for the PG-325 SQL family-holdout artifacts.

The audit never starts Docker and never contacts a target.  It rechecks the
fixed counts, model-context allow-list, fresh-reset/source/evidence fields,
belief/failure transitions, canary presence, and promotion firewall.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
REPORT = RESEARCH / "pg325_sql_family_holdout_report_v1.json"
CATALOG = RESEARCH / "pg325_sql_family_holdout_catalog_v1.json"
TRACE = RESEARCH / "pg325_sql_family_holdout_trace_v1.json"
PROTOCOL = RESEARCH / "pg325_sql_family_holdout_protocol_v1.json"

MODEL_KEYS = frozenset(
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


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not an object")
    return value


def _hash_matches(document: Mapping[str, Any], key: str) -> bool:
    expected = str(document.get(key, ""))
    if len(expected) != 64:
        return False
    clone = dict(document)
    clone[key] = ""
    return _digest(clone) == expected


def _context_ok(context: Any) -> bool:
    if not isinstance(context, list) or not context:
        return False
    for token in context:
        text = str(token)
        if text in {"[BOS]", "[EOS]"}:
            continue
        if "=" not in text or text.split("=", 1)[0] not in MODEL_KEYS:
            return False
    return True


def _firewall(catalog: Mapping[str, Any], trace: Mapping[str, Any]) -> bool:
    contexts: list[Any] = []
    rows = catalog.get("entries") if isinstance(catalog.get("entries"), list) else []
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        model = row.get("model") if isinstance(row.get("model"), Mapping) else {}
        contexts.extend(entry.get("context_tokens") for entry in model.get("entries", []) if isinstance(entry, Mapping))
        contexts.append(model.get("failure_context"))
    contexts.extend(row.get("context_tokens") for row in trace.get("episodes", []) if isinstance(row, Mapping))
    return bool(contexts) and all(_context_ok(context) for context in contexts)


def audit() -> dict[str, Any]:
    failures: list[str] = []
    documents: dict[str, dict[str, Any]] = {}
    for name, path in (("report", REPORT), ("catalog", CATALOG), ("trace", TRACE), ("protocol", PROTOCOL)):
        if not path.exists():
            failures.append(f"missing:{path.relative_to(ROOT)}")
            continue
        try:
            documents[name] = _load(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"unreadable:{name}:{type(exc).__name__}")

    report = documents.get("report", {})
    catalog = documents.get("catalog", {})
    trace = documents.get("trace", {})
    protocol = documents.get("protocol", {})
    expected_schema = {
        "report": "pg325-sql-family-holdout-report-v1",
        "catalog": "pg325-sql-family-holdout-catalog-v1",
        "trace": "pg325-sql-family-holdout-trace-v1",
        "protocol": "pg325-sql-family-holdout-protocol-v1",
    }
    for name, schema in expected_schema.items():
        if documents.get(name, {}).get("schema_version") != schema:
            failures.append(f"schema:{name}")

    counts = report.get("counts") if isinstance(report.get("counts"), Mapping) else {}
    for key, value in {
        "seed_count": 3,
        "route_count": 9,
        "get_count": 6,
        "post_count": 3,
        "positive_route_count": 9,
        "positive_typed_effect_count": 9,
        "variant_role_count": 27,
        "variant_exact_count": 27,
        "negative_lane_violation_count": 0,
        "failure_repair_correct_count": 9,
        "failure_repair_count": 9,
        "failure_transition_required_count": 9,
        "failure_action_changed_count": 9,
        "multi_missing_question_rows": 135,
        "multi_missing_unsafe_allow": 0,
        "belief_transition_count": 27,
        "belief_duplicate_evidence_count": 0,
    }.items():
        if counts.get(key) != value:
            failures.append(f"count:{key}")

    checks = report.get("checks") if isinstance(report.get("checks"), Mapping) else {}
    required_true = (
        "real_docker_contacted", "fresh_container_per_route_seed", "get_post_pair",
        "sql_family_holdout", "cross_implementation_replay_canaries_present",
        "docker_network_none", "external_network_disabled", "zero_volume_per_route",
        "database_health_per_route", "source_attestation_per_route",
        "typed_evidence_hash_per_route", "belief_trace_complete",
        "failure_action_changed_all", "model_context_firewall",
    )
    for key in required_true:
        if checks.get(key) is not True:
            failures.append(f"check:{key}")
    for key in ("raw_payload_in_model_context", "raw_response_bodies_stored", "public_target_contacted", "sql_time_delay", "sql_write"):
        if checks.get(key) is not False:
            failures.append(f"safety:{key}")
    if not _firewall(catalog, trace):
        failures.append("model_context_firewall")

    rows = catalog.get("entries") if isinstance(catalog.get("entries"), list) else []
    if len(rows) != 9:
        failures.append("catalog_rows")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            failures.append(f"row:{index}:object")
            continue
        target = row.get("target") if isinstance(row.get("target"), Mapping) else {}
        reset = target.get("fresh_reset") if isinstance(target.get("fresh_reset"), Mapping) else {}
        oracle = row.get("oracle") if isinstance(row.get("oracle"), Mapping) else {}
        model = row.get("model") if isinstance(row.get("model"), Mapping) else {}
        if reset.get("fresh_target") is not True or reset.get("container_recreated") is not True or reset.get("container_restart_used") is not False:
            failures.append(f"row:{index}:fresh_reset")
        if reset.get("network_mode") != "none" or reset.get("host_port_published") is not False or int(reset.get("volume_mount_count", -1)) != 0:
            failures.append(f"row:{index}:network_mount")
        if reset.get("database_health_gate") != "mysqli_root_pikachu_ok":
            failures.append(f"row:{index}:database_health")
        if len(str(target.get("source_sha256", ""))) != 64 or len(str(oracle.get("evidence_sha256", ""))) != 64:
            failures.append(f"row:{index}:attestation")
        if model.get("raw_payload_in_context") is True or model.get("raw_response_body_in_context") is True:
            failures.append(f"row:{index}:context_firewall")
        transition = model.get("failure_transition") if isinstance(model, Mapping) else {}
        if not isinstance(transition, Mapping) or transition.get("repair_transition_valid") is not True:
            failures.append(f"row:{index}:failure_transition")
        elif transition.get("repair_transition_required") is True and transition.get("action_changed") is not True:
            failures.append(f"row:{index}:action_changed")
        if row.get("training_eligible") is not False or row.get("memory_promotion_allowed") is not False or row.get("vulnerability_claim_allowed") is not False:
            failures.append(f"row:{index}:promotion")
        if not isinstance(row.get("belief_trace"), list) or row.get("belief_transition_complete") is not True:
            failures.append(f"row:{index}:belief")
        else:
            for step_index, step in enumerate(row["belief_trace"]):
                if not isinstance(step, Mapping) or step.get("evidence_scope") != "record_role_bound" or len(str(step.get("evidence_hash", ""))) != 64 or len(str(step.get("source_evidence_sha256", ""))) != 64 or step.get("duplicate_evidence") is not False:
                    failures.append(f"row:{index}:belief_role_bound:{step_index}")

    if catalog.get("raw_payloads_human_review_only") is not True or catalog.get("raw_response_bodies_stored") is not False:
        failures.append("catalog_raw_firewall")
    if trace.get("training_eligible") is not False or trace.get("memory_promotion_allowed") is not False or trace.get("raw_response_bodies_stored") is not False:
        failures.append("trace_promotion")
    canaries = report.get("cross_implementation_canaries") if isinstance(report.get("cross_implementation_canaries"), Mapping) else {}
    if set(canaries) != {"pg323_vulnerableapp_role_replay_report_v1.json", "pg324_juice_shop_source_heldout_report_v1.json"}:
        failures.append("canary_set")
    for key, canary in canaries.items():
        if not isinstance(canary, Mapping) or not isinstance(canary.get("promotion"), Mapping):
            failures.append(f"canary:{key}")
        elif any(canary["promotion"].get(field) is not False for field in ("training_allowed", "memory_promotion_allowed", "vulnerability_claim_allowed")):
            failures.append(f"canary_promotion:{key}")

    required_gates = protocol.get("required_gates") if isinstance(protocol.get("required_gates"), Mapping) else {}
    for key in ("multi_missing_question", "get_post_pair", "typed_sql_effect", "matched_negative", "fresh_reset", "database_health", "source_attestation", "evidence_hash", "belief_update", "role_bound_belief_evidence", "failure_action_changed", "model_context_firewall", "docker_network_none", "raw_payload_training_excluded"):
        if required_gates.get(key) is not True:
            failures.append(f"protocol_gate:{key}")
    promotion = report.get("promotion") if isinstance(report.get("promotion"), Mapping) else {}
    for key in ("training_allowed", "memory_promotion_allowed", "payload_catalog_promotion_allowed", "vulnerability_claim_allowed"):
        if promotion.get(key) is not False:
            failures.append(f"promotion:{key}")
    for name, field in (("report", "report_sha256"), ("catalog", "catalog_sha256"), ("trace", "trace_sha256"), ("protocol", "protocol_sha256")):
        if name in documents and not _hash_matches(documents[name], field):
            failures.append(f"hash:{name}")

    return {
        "audit_id": "pg325-sql-family-holdout-audit-v1",
        "status": "passed" if not failures else "blocked",
        "report": str(REPORT.relative_to(ROOT)),
        "failures": failures,
        "promotion_allowed": False,
        "target_contacted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else f"{result['status']}: {', '.join(result['failures']) or 'no failures'}")
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
