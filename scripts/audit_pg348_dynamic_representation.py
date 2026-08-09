"""Read-only audit for the PG-348 dynamic context representation candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "pg348-dynamic-representation-audit-v1"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_report(report: Mapping[str, Any]) -> str:
    value = dict(report)
    value.pop("report_sha256", None)
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def audit_report(report: Mapping[str, Any], report_file_sha256: str | None = None) -> dict[str, Any]:
    failures: list[str] = []
    if report.get("report_sha256") != _sha_report(report):
        failures.append("report_hash_mismatch")
    if report.get("schema_version") != "pg331-a800-representation-smoke-v1":
        failures.append("schema_mismatch")
    training = report.get("training") or {}
    if training.get("context_only") is not True or training.get("target_tokens_read") is not False:
        failures.append("context_only_contract")
    gate = report.get("gate") or {}
    if gate.get("representation_training_allowed") is not True or gate.get("failures"):
        failures.append("runner_gate_not_ready")
    if int(gate.get("context_row_count", 0)) != 960 or int((gate.get("split_counts") or {}).get("implementation_holdout", 0)) != 1120:
        failures.append("split_count_mismatch")
    if int(gate.get("unknown_context_token_count", 1)) != 0:
        failures.append("unknown_holdout_token")
    if report.get("gate", {}).get("information_promotion_gate_passed") is not True:
        failures.append("information_gate_not_passed")
    promotion = report.get("promotion") or {}
    if any(value is not False for value in promotion.values()):
        failures.append("promotion_not_closed")
    loss_rows = report.get("loss") or []
    if len(loss_rows) != 3:
        failures.append("seed_count_mismatch")
    for row in loss_rows:
        if int((row.get("train") or {}).get("next_token_count", 0)) <= 0 or int((row.get("heldout") or {}).get("next_token_count", 0)) <= 0:
            failures.append("empty_loss_measurement")
    return {"schema_version": SCHEMA_VERSION, "status": "passed_representation_observation_blocked_capability" if failures else "passed_representation_candidate", "failures": sorted(set(failures)), "report_internal_sha256": report.get("report_sha256"), "report_file_sha256": report_file_sha256, "split_counts": gate.get("split_counts"), "information_gate_status": gate.get("information_gate_status"), "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit PG-348 dynamic representation candidate")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8-sig"))
    audit = audit_report(report, _sha_file(args.report))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
