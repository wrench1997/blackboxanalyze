"""Read-only audit for the PG-350 evaluator-only binding replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "research" / "pg350_runtime_binding_replay_report_v1.json"
DEFAULT_SIDECARS = ROOT / "research" / "pg350_runtime_binding_replay_sidecars_v1.json"
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_LITERALS = ("PG350S", "http://127.0.0.1:")
FORBIDDEN_EXACT_KEYS = ("payload", "raw_payload", "raw_value", "wire", "body", "response", "response_body", "evaluator_answer")


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _audit(report: Mapping[str, Any], sidecars: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    report_text = json.dumps(report, ensure_ascii=False, sort_keys=True)
    sidecar_text = json.dumps(sidecars, ensure_ascii=False, sort_keys=True)
    for needle in FORBIDDEN_LITERALS:
        if needle in report_text or needle in sidecar_text:
            failures.append(f"raw_or_route_literal:{needle}")
    for key in FORBIDDEN_EXACT_KEYS:
        needle = f'"{key}"'
        if needle in report_text or needle in sidecar_text:
            failures.append(f"raw_or_exact_key:{key}")
    if report.get("schema_version") != "pg350-runtime-binding-replay-v1":
        failures.append("report_schema")
    if report.get("target_contacted") is not True or report.get("network_policy") != "loopback_only":
        failures.append("loopback_contact_contract")
    if report.get("external_network") is not False:
        failures.append("external_network")
    promotion = report.get("promotion")
    if not isinstance(promotion, Mapping) or any(value is not False for value in promotion.values()):
        failures.append("promotion_not_fail_closed")
    rows = sidecars.get("sidecars")
    if not isinstance(rows, list) or not rows:
        failures.append("missing_sidecars")
        rows = []
    expected_roles = {"candidate", "reference", "negative", "replay"}
    for index, row in enumerate(rows):
        prefix = f"episode[{index}]"
        if not isinstance(row, Mapping):
            failures.append(f"{prefix}:not_mapping")
            continue
        if not HASH_RE.fullmatch(str(row.get("record_id", ""))) or not HASH_RE.fullmatch(str(row.get("route_digest", ""))):
            failures.append(f"{prefix}:identity_hash")
        roles = row.get("roles")
        if not isinstance(roles, Mapping) or set(roles) != expected_roles:
            failures.append(f"{prefix}:role_set")
            continue
        checks = row.get("checks")
        if not isinstance(checks, Mapping):
            failures.append(f"{prefix}:checks_missing")
        else:
            positive_checks = ("candidate_typed", "reference_typed", "negative_clean", "replay_consistent", "all_role_evidence", "fresh_reset_per_role", "failure_action_change")
            if any(checks.get(key) is not True for key in positive_checks):
                failures.append(f"{prefix}:positive_gate")
            if checks.get("raw_wire_stored") is not False or checks.get("raw_response_stored") is not False:
                failures.append(f"{prefix}:raw_storage")
        if row.get("confirmed_positive") is not True:
            failures.append(f"{prefix}:not_confirmed")
        for role in expected_roles:
            role_row = roles[role]
            if not isinstance(role_row, Mapping) or not HASH_RE.fullmatch(str(role_row.get("evidence_sha256", ""))):
                failures.append(f"{prefix}:{role}:evidence_hash")
            binding = role_row.get("binding") if isinstance(role_row, Mapping) else None
            if not isinstance(binding, Mapping) or binding.get("raw_payload_stored") is not False or binding.get("raw_wire_stored") is not False or binding.get("training_context_allowed") is not False:
                failures.append(f"{prefix}:{role}:binding_firewall")
        repair = row.get("failure_repair")
        if not isinstance(repair, Mapping) or repair.get("action_changed") is not True or repair.get("typed_effect_confirmed") is not False:
            failures.append(f"{prefix}:failure_repair")

    counts = report.get("counts") if isinstance(report.get("counts"), Mapping) else {}
    if counts.get("episodes") != len(rows):
        failures.append("count_episode_mismatch")
    status = "passed_evaluator_only" if not failures else "blocked"
    return {
        "schema_version": "pg350-runtime-binding-replay-audit-v1",
        "status": status,
        "failures": failures,
        "counts": {
            "episodes": len(rows),
            "confirmed_positive": sum(row.get("confirmed_positive") is True for row in rows if isinstance(row, Mapping)),
            "raw_firewall_violations": sum(
                any(needle in json.dumps(row, ensure_ascii=False) for needle in FORBIDDEN_LITERALS)
                or any(f'"{key}"' in json.dumps(row, ensure_ascii=False) for key in FORBIDDEN_EXACT_KEYS)
                for row in rows
                if isinstance(row, Mapping)
            ),
        },
        "input_hashes": {"report_sha256": _sha(report), "sidecars_sha256": _sha(sidecars)},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "scope": {"synthetic_evaluator_only": True, "neural_payload_capability_proven": False, "public_target_claim": False},
    }


def audit_files(report_path: Path = DEFAULT_REPORT, sidecars_path: Path = DEFAULT_SIDECARS) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    sidecars = json.loads(sidecars_path.read_text(encoding="utf-8"))
    return _audit(report, sidecars)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit PG-350 binding replay artifacts without network access")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--sidecars", type=Path, default=DEFAULT_SIDECARS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_files(args.report, args.sidecars)
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed_evaluator_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
