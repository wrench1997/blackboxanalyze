"""Read-only structural audit for the PG-326 matrix artifacts.

The audit validates the matrix itself and deliberately keeps the scientific
gate blocked when a strict observation contract or before/after forgetting
pair is missing.  It never starts Docker and never contacts a target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
REPORT = RESEARCH / "pg326_cross_impl_forgetting_matrix_v1.json"
PROTOCOL = RESEARCH / "pg326_cross_impl_forgetting_matrix_protocol_v1.json"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(path.name)
    return value


def _hash_matches(document: Mapping[str, Any], key: str) -> bool:
    expected = str(document.get(key, ""))
    if len(expected) != 64:
        return False
    clone = dict(document)
    clone[key] = ""
    return _digest(clone) == expected


def audit() -> dict[str, Any]:
    failures: list[str] = []
    if not REPORT.exists():
        failures.append("missing:report")
        report: dict[str, Any] = {}
    else:
        try:
            report = _load(REPORT)
        except (OSError, ValueError, json.JSONDecodeError):
            report = {}
            failures.append("unreadable:report")
    if not PROTOCOL.exists():
        failures.append("missing:protocol")
        protocol: dict[str, Any] = {}
    else:
        try:
            protocol = _load(PROTOCOL)
        except (OSError, ValueError, json.JSONDecodeError):
            protocol = {}
            failures.append("unreadable:protocol")

    if report.get("schema_version") != "pg326-cross-implementation-forgetting-matrix-report-v1":
        failures.append("schema:report")
    if protocol.get("schema_version") != "pg326-cross-implementation-forgetting-matrix-protocol-v1":
        failures.append("schema:protocol")
    if not _hash_matches(report, "report_sha256"):
        failures.append("hash:report")
    if not _hash_matches(protocol, "protocol_sha256"):
        failures.append("hash:protocol")
    if report.get("status") != "completed_read_only_matrix_blocked":
        failures.append("status:report")
    runtime = report.get("runtime") if isinstance(report.get("runtime"), Mapping) else {}
    if runtime.get("target_contacted") is not False or runtime.get("docker_started") is not False:
        failures.append("runtime:target_contacted")
    if len(list(report.get("source_rows") or [])) != 3:
        failures.append("source_rows")
    if len(list(report.get("implementation_digests") or [])) < 2:
        failures.append("implementation_count")
    totals = report.get("totals") if isinstance(report.get("totals"), Mapping) else {}
    expected_totals = {
        "seed_count": 9,
        "route_count": 45,
        "get_count": 27,
        "post_count": 18,
        "positive_typed_effect_count": 18,
        "positive_route_count": 18,
        "variant_exact_count": 135,
        "variant_role_count": 135,
        "multi_missing_question_rows": 675,
        "failure_repair_correct_count": 45,
        "failure_repair_count": 45,
        "negative_lane_violation_count": 0,
    }
    for key, expected in expected_totals.items():
        if totals.get(key) != expected:
            failures.append(f"total:{key}")
    gate = report.get("hypothesis_gate") if isinstance(report.get("hypothesis_gate"), Mapping) else {}
    checks = gate.get("checks") if isinstance(gate.get("checks"), Mapping) else {}
    if gate.get("status") != "blocked" or gate.get("claim_allowed") is not False:
        failures.append("hypothesis_gate")
    if checks.get("independent_implementation_count") is not True or checks.get("family_diversity") is not True:
        failures.append("gate:coverage")
    if checks.get("uniform_observation_contract") is not False:
        failures.append("gate:missing_contract_not_exposed")
    if checks.get("forgetting_pair") is not True:
        failures.append("gate:forgetting_pair_not_exposed")
    missing = {str(item) for item in list(report.get("missing_requirements") or [])}
    for required in ("failure_action_change_contract", "check:role_bound_belief_evidence"):
        if required not in missing:
            failures.append(f"missing_requirement:{required}")
    if "forgetting:paired_before_after_replay" in missing:
        failures.append("missing_requirement:forgetting_pair")
    promotion = report.get("promotion") if isinstance(report.get("promotion"), Mapping) else {}
    for key in ("training_allowed", "memory_promotion_allowed", "payload_catalog_promotion_allowed", "vulnerability_claim_allowed"):
        if promotion.get(key) is not False:
            failures.append(f"promotion:{key}")
    required_gates = protocol.get("required_gates") if isinstance(protocol.get("required_gates"), Mapping) else {}
    for key in ("independent_implementation_count", "family_diversity", "worst_seed_typed_effect", "worst_seed_variant", "worst_seed_ask", "worst_seed_repair", "negative_zero", "uniform_observation_contract", "forgetting_pair", "raw_payload_training_excluded", "promotion_blocked"):
        if required_gates.get(key) is not True:
            failures.append(f"protocol_gate:{key}")
    protocol_promotion = protocol.get("promotion") if isinstance(protocol.get("promotion"), Mapping) else {}
    if any(protocol_promotion.get(key) is not False for key in ("training_allowed", "memory_promotion_allowed", "payload_catalog_promotion_allowed", "vulnerability_claim_allowed")):
        failures.append("protocol_promotion")
    return {
        "audit_id": "pg326-cross-implementation-forgetting-matrix-audit-v1",
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
