"""Audit the PG-324 artifact contract without contacting a target.

This is deliberately a report-level gate.  It rejects the pre-v2 report that
used a browser dialog as the typed oracle and prevents a stale/partial report
from being surfaced as a capability result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
REPORT = RESEARCH / "pg324_juice_shop_source_heldout_report_v1.json"
CATALOG = RESEARCH / "pg324_juice_shop_source_heldout_catalog_v1.json"
TRACE = RESEARCH / "pg324_juice_shop_source_heldout_trace_v1.json"
PROTOCOL = RESEARCH / "pg324_juice_shop_source_heldout_protocol_v1.json"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not an object")
    return value


def _hash_matches(document: Mapping[str, Any], field: str) -> bool:
    expected = str(document.get(field, ""))
    if len(expected) != 64:
        return False
    clone = dict(document)
    clone[field] = ""
    return _digest(clone) == expected


_MODEL_CONTEXT_KEYS = frozenset(
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


def _context_firewall(catalog: Mapping[str, Any], trace: Mapping[str, Any]) -> bool:
    """Inspect actual decoder contexts, not only report-level declarations."""

    contexts: list[Any] = []
    rows = catalog.get("entries") if isinstance(catalog.get("entries"), list) else []
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        model = row.get("model") if isinstance(row.get("model"), Mapping) else {}
        contexts.extend(entry.get("context_tokens") for entry in (model.get("entries") or []) if isinstance(entry, Mapping))
        contexts.append(model.get("failure_context"))
    episodes = trace.get("episodes") if isinstance(trace.get("episodes"), list) else []
    contexts.extend(episode.get("context_tokens") for episode in episodes if isinstance(episode, Mapping))
    if not contexts:
        return False
    for context in contexts:
        if not isinstance(context, list) or not context:
            return False
        for token in context:
            token_text = str(token)
            if token_text in {"[BOS]", "[EOS]"}:
                continue
            if "=" not in token_text or token_text.split("=", 1)[0] not in _MODEL_CONTEXT_KEYS:
                return False
    return True


def audit() -> dict[str, Any]:
    failures: list[str] = []
    missing: list[str] = []
    documents: dict[str, dict[str, Any]] = {}
    for name, path in (("report", REPORT), ("catalog", CATALOG), ("trace", TRACE), ("protocol", PROTOCOL)):
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
            continue
        try:
            documents[name] = _load(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"{name}_unreadable:{type(exc).__name__}")

    report = documents.get("report", {})
    catalog = documents.get("catalog", {})
    trace = documents.get("trace", {})
    protocol = documents.get("protocol", {})
    if missing:
        failures.extend(f"missing:{item}" for item in missing)

    if report.get("schema_version") != "pg324-juice-shop-source-heldout-report-v2":
        failures.append("report_schema_v2")
    if catalog.get("schema_version") != "pg324-juice-shop-source-heldout-catalog-v2":
        failures.append("catalog_schema_v2")
    if trace.get("schema_version") != "pg324-juice-shop-source-heldout-trace-v2":
        failures.append("trace_schema_v2")
    if protocol.get("schema_version") != "pg324-juice-shop-source-heldout-protocol-v2":
        failures.append("protocol_schema_v2")

    counts = report.get("counts") if isinstance(report.get("counts"), Mapping) else {}
    expected_counts = {
        "seed_count": 3,
        "route_count": 18,
        "get_count": 9,
        "post_count": 9,
        "variant_role_count": 54,
        "variant_exact_count": 54,
        "failure_repair_count": 18,
        "failure_transition_count": 18,
        "failure_transition_required_count": 9,
        "failure_action_changed_count": 9,
        "multi_missing_question_rows": 270,
        "multi_missing_unsafe_allow": 0,
        "negative_lane_violation_count": 0,
    }
    for key, expected in expected_counts.items():
        if counts.get(key) != expected:
            failures.append(f"count:{key}")

    checks = report.get("checks") if isinstance(report.get("checks"), Mapping) else {}
    required_checks = (
        "real_docker_contacted", "fresh_container_per_route_seed", "get_post_pair", "independent_implementation",
        "docker_network_none", "loopback_relay_only", "external_network_disabled", "zero_bind_volume_per_route",
        "source_attestation_per_route", "safety_mode_override_all", "typed_evidence_hash_per_route", "challenge_state_baseline_all", "belief_trace_complete", "failure_action_changed_all", "model_context_firewall",
        "raw_payload_in_model_context", "raw_response_bodies_stored", "public_target_contacted", "time_delay",
        "domain_data_write", "stateful_xss_write",
    )
    for key in required_checks:
        if key not in checks:
            failures.append(f"check_missing:{key}")
    if not _context_firewall(catalog, trace):
        failures.append("model_context_firewall")
    for key in ("real_docker_contacted", "fresh_container_per_route_seed", "get_post_pair", "independent_implementation", "docker_network_none", "loopback_relay_only", "external_network_disabled", "zero_bind_volume_per_route", "source_attestation_per_route", "safety_mode_override_all", "typed_evidence_hash_per_route", "challenge_state_baseline_all", "belief_trace_complete", "failure_action_changed_all", "model_context_firewall", "evaluator_state_transition_expected"):
        if checks.get(key) is not True:
            failures.append(f"check_false:{key}")
    for key in ("raw_payload_in_model_context", "raw_response_bodies_stored", "public_target_contacted", "time_delay", "domain_data_write", "stateful_xss_write"):
        if checks.get(key) is not False:
            failures.append(f"safety:{key}")

    rows = catalog.get("entries") if isinstance(catalog.get("entries"), list) else []
    if len(rows) != 18:
        failures.append("catalog_route_rows")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            failures.append(f"row:{index}:object")
            continue
        target = row.get("target") if isinstance(row.get("target"), Mapping) else {}
        reset = target.get("fresh_reset") if isinstance(target.get("fresh_reset"), Mapping) else {}
        oracle = row.get("oracle") if isinstance(row.get("oracle"), Mapping) else {}
        model = row.get("model") if isinstance(row.get("model"), Mapping) else {}
        if reset.get("challenge_state_baseline_available") is not True or reset.get("challenge_state_baseline_solved") is not False:
            failures.append(f"row:{index}:fresh_baseline")
        if len(str(reset.get("safety_mode_override_sha256", ""))) != 64:
            failures.append(f"row:{index}:safety_mode_override")
        if len(str(oracle.get("evidence_sha256", ""))) != 64:
            failures.append(f"row:{index}:evidence_hash")
        if model.get("raw_payload_in_context") is True or model.get("raw_response_body_in_context") is True:
            failures.append(f"row:{index}:model_firewall")
        if not isinstance(row.get("belief_trace"), list) or row.get("belief_transition_complete") is not True:
            failures.append(f"row:{index}:belief_trace")
        transition = row.get("model", {}).get("failure_transition") if isinstance(row.get("model"), Mapping) else None
        if not isinstance(transition, Mapping) or transition.get("repair_transition_valid") is not True:
            failures.append(f"row:{index}:failure_transition")
        elif transition.get("repair_transition_required") is True and transition.get("action_changed") is not True:
            failures.append(f"row:{index}:failure_action_changed")
        if row.get("training_eligible") is not False or row.get("memory_promotion_allowed") is not False or row.get("vulnerability_claim_allowed") is not False:
            failures.append(f"row:{index}:promotion")

    if catalog.get("raw_payloads_human_review_only") is not True or catalog.get("raw_response_bodies_stored") is not False:
        failures.append("catalog_raw_firewall")
    if trace.get("training_eligible") is not False or trace.get("memory_promotion_allowed") is not False or trace.get("raw_response_bodies_stored") is not False:
        failures.append("trace_promotion_firewall")
    episodes = trace.get("episodes") if isinstance(trace.get("episodes"), list) else []
    if not episodes:
        failures.append("trace_belief_episodes")
    for index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping) or not isinstance(episode.get("belief_before"), Mapping) or not isinstance(episode.get("belief_after"), Mapping):
            failures.append(f"trace:{index}:belief_transition")
        if isinstance(episode, Mapping) and str(episode.get("record_id", "")).endswith(":failure-repair"):
            transition = episode.get("failure_transition")
            if not isinstance(transition, Mapping) or transition.get("repair_transition_valid") is not True or (transition.get("repair_transition_required") is True and transition.get("action_changed") is not True):
                failures.append(f"trace:{index}:failure_action_changed")
    required_protocol = protocol.get("required_gates") if isinstance(protocol.get("required_gates"), Mapping) else {}
    for key in ("multi_missing_question", "get_post_pair", "typed_challenge_state_delta", "fresh_baseline_unsolved", "belief_update", "failure_action_changed", "model_context_firewall", "matched_negative", "fresh_reset", "evidence_hash", "safety_mode_override", "docker_network_none", "loopback_relay_only", "raw_payload_training_excluded"):
        if required_protocol.get(key) is not True:
            failures.append(f"protocol_gate:{key}")
    promotion = report.get("promotion") if isinstance(report.get("promotion"), Mapping) else {}
    for key in ("training_allowed", "memory_promotion_allowed", "payload_catalog_promotion_allowed", "vulnerability_claim_allowed"):
        if promotion.get(key) is not False:
            failures.append(f"promotion:{key}")

    for name, field in (("report", "report_sha256"), ("catalog", "catalog_sha256"), ("trace", "trace_sha256"), ("protocol", "protocol_sha256")):
        if name in documents and not _hash_matches(documents[name], field):
            failures.append(f"hash:{name}")

    status = "passed" if not failures else ("stale_contract" if any(item.endswith("schema_v2") for item in failures) else "blocked")
    return {
        "audit_id": "pg324-source-heldout-report-audit-v2",
        "status": status,
        "report": str(REPORT.relative_to(ROOT)),
        "failures": failures,
        "promotion_allowed": False,
        "target_contacted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else f"{result['status']}: {', '.join(result['failures']) or 'no failures'}")
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
