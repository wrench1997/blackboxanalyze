"""Read-only audit for PG-327B paired fresh replay artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
REPORT = RESEARCH / "pg327b_paired_fresh_replay_report_v1.json"
TRACE = RESEARCH / "pg327b_paired_fresh_replay_trace_v1.json"
PROTOCOL = RESEARCH / "pg327b_paired_fresh_replay_protocol_v1.json"
AUDIT = RESEARCH / "pg327b_paired_fresh_replay_audit_v1.json"
MODEL_CONTEXT_KEYS = frozenset(
    {
        "typed_available", "feedback_state", "replay_ready", "evidence_present",
        "negative_control", "fresh_reset", "surface_method", "surface_field_role",
        "surface_encoding", "history_action", "failure_class", "step_budget",
    }
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(path.name)
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _hash_matches(document: Mapping[str, Any], key: str) -> bool:
    expected = str(document.get(key, ""))
    if len(expected) != 64:
        return False
    clone = dict(document)
    clone[key] = ""
    return _digest(clone) == expected


def _context_ok(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    for token in value:
        text = str(token)
        if text in {"[BOS]", "[EOS]"}:
            continue
        if "=" not in text or text.split("=", 1)[0] not in MODEL_CONTEXT_KEYS:
            return False
    return True


def audit() -> dict[str, Any]:
    failures: list[str] = []
    documents: dict[str, dict[str, Any]] = {}
    for name, path in (("report", REPORT), ("trace", TRACE), ("protocol", PROTOCOL)):
        if not path.exists():
            failures.append(f"missing:{name}")
            documents[name] = {}
            continue
        try:
            documents[name] = _load(path)
        except (OSError, ValueError, json.JSONDecodeError):
            failures.append(f"unreadable:{name}")
            documents[name] = {}
    report = documents["report"]
    trace = documents["trace"]
    protocol = documents["protocol"]
    expected_schemas = {"report": "pg327b-paired-fresh-replay-report-v1", "trace": "pg327b-paired-fresh-replay-trace-v1", "protocol": "pg327b-paired-fresh-replay-protocol-v1"}
    for name, expected in expected_schemas.items():
        if documents[name].get("schema_version") != expected:
            failures.append(f"schema:{name}")
    for name, key in (("report", "report_sha256"), ("trace", "trace_sha256"), ("protocol", "protocol_sha256")):
        if not _hash_matches(documents[name], key):
            failures.append(f"hash:{name}")
    if report.get("status") != "completed_local_docker_pg327b_paired_replay":
        failures.append("status:report")
    runtime = report.get("runtime") if isinstance(report.get("runtime"), Mapping) else {}
    if runtime.get("target_contacted") is not True or runtime.get("docker_started") is not True:
        failures.append("runtime:live_replay_missing")
    counts = report.get("counts") if isinstance(report.get("counts"), Mapping) else {}
    for key, expected in (("phase_count", 2), ("seed_count", 3), ("route_count_per_phase", 9), ("total_phase_routes", 18)):
        if counts.get(key) != expected:
            failures.append(f"count:{key}")
    checks = report.get("checks") if isinstance(report.get("checks"), Mapping) else {}
    required_checks = (
        "fresh_reset_before_all", "fresh_reset_after_all", "distinct_container_pairs",
        "get_post_pair_before", "get_post_pair_after", "candidate_reference_negative_before",
        "candidate_reference_negative_after", "typed_evidence_before", "typed_evidence_after",
        "source_attestation_before", "source_attestation_after", "failure_action_changed_before",
        "failure_action_changed_after", "role_bound_belief_before", "role_bound_belief_after",
        "context_firewall_before", "context_firewall_after", "raw_payload_excluded",
        "raw_response_excluded", "network_none_before", "network_none_after",
        "same_canary_route_set", "before_after_checkpoint_distinct",
    )
    for key in required_checks:
        if checks.get(key) is not True:
            failures.append(f"check:{key}")
    forgetting = report.get("forgetting") if isinstance(report.get("forgetting"), Mapping) else {}
    if forgetting.get("paired_replay_present") is not True or forgetting.get("same_canary_route_set") is not True:
        failures.append("forgetting:paired_replay")
    pairs = list(forgetting.get("pairs") or [])
    if len(pairs) != 3:
        failures.append("forgetting:pairs")
    for pair in pairs:
        if pair.get("same_canary_route_set") is not True:
            failures.append("forgetting:route_set")
        if not str(pair.get("before_checkpoint_sha256", "")) or not str(pair.get("after_checkpoint_sha256", "")):
            failures.append("forgetting:checkpoint_hash")
        for key, value in dict(pair.get("after_not_worse_observed") or {}).items():
            if value is not True:
                failures.append(f"forgetting:regression:{key}")
    episodes = list(trace.get("episodes") or [])
    if len(episodes) == 0 or trace.get("raw_payload_stored") is not False or trace.get("raw_response_body_stored") is not False:
        failures.append("trace:incomplete")
    for row in episodes:
        if not _context_ok(row.get("context_tokens")):
            failures.append("trace:context_firewall")
        if row.get("raw_payload_stored") is not False or row.get("raw_response_body_stored") is not False:
            failures.append("trace:raw_material")
        for token in list(row.get("context_tokens") or []) + list(row.get("target_tokens") or []) + list(row.get("predicted_tokens") or []):
            text = str(token).lower()
            if any(marker in text for marker in ("payload", "response", "<script", "select ", "union ", "wire=")):
                failures.append("trace:forbidden_token")
                break
    promotion = report.get("promotion") if isinstance(report.get("promotion"), Mapping) else {}
    for key in ("training_allowed", "memory_promotion_allowed", "payload_catalog_promotion_allowed", "vulnerability_claim_allowed"):
        if promotion.get(key) is not False:
            failures.append(f"promotion:{key}")
    protocol_gates = protocol.get("required_gates") if isinstance(protocol.get("required_gates"), Mapping) else {}
    for key in list(required_checks) + ["paired_forgetting_replay", "raw_payload_training_excluded", "promotion_blocked"]:
        if protocol_gates.get(key) is not True:
            failures.append(f"protocol_gate:{key}")
    protocol_promotion = protocol.get("promotion") if isinstance(protocol.get("promotion"), Mapping) else {}
    if any(protocol_promotion.get(key) is not False for key in ("training_allowed", "memory_promotion_allowed", "payload_catalog_promotion_allowed", "vulnerability_claim_allowed")):
        failures.append("protocol:promotion")
    return {
        "audit_id": "pg327b-paired-fresh-replay-audit-v1",
        "status": "passed" if not failures else "blocked",
        "report": str(REPORT.relative_to(ROOT)),
        "failures": sorted(set(failures)),
        "promotion_allowed": False,
        "target_contacted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = audit()
    result["audit_sha256"] = ""
    result["audit_sha256"] = _digest(result)
    AUDIT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else f"{result['status']}: {', '.join(result['failures']) or 'no failures'}")
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
