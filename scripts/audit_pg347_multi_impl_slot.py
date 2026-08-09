"""Independent, read-only audit for the PG-347 A800 candidate report.

The report is an experiment result, not a capability or payload claim.  This
auditor checks the report hash, locked gate, split sizes, context-only slot
contract, and the hard holdout metrics without opening a target or loading a
checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "pg347-multi-impl-slot-audit-v1"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_report(report: Mapping[str, Any]) -> str:
    value = dict(report)
    value.pop("report_sha256", None)
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def audit_report(report: Mapping[str, Any], *, report_file_sha256: str | None = None) -> dict[str, Any]:
    failures: list[str] = []
    internal = str(report.get("report_sha256", ""))
    if len(internal) != 64 or internal != _sha_report(report):
        failures.append("report_hash_mismatch")
    if report.get("schema_version") != "pg346-a800-structured-target-slot-diagnostic-v1":
        failures.append("schema_mismatch")
    gate = report.get("gate") or {}
    if gate.get("training_allowed") is not True or gate.get("failures"):
        failures.append("runner_gate_not_ready")
    training = report.get("training") or {}
    if training.get("context_only_slot_inference") is not True or training.get("target_tokens_used_as_labels_only") is not True:
        failures.append("slot_context_firewall_contract")
    promotion = report.get("promotion") or {}
    if any(value is not False for value in promotion.values()):
        failures.append("promotion_not_closed")
    if list(training.get("seeds") or []) != [34701, 34702, 34703]:
        failures.append("seed_set_mismatch")
    if int((gate.get("split_counts") or {}).get("train", 0)) != 27 or int((gate.get("split_counts") or {}).get("implementation_holdout", 0)) != 12:
        failures.append("split_count_mismatch")
    worst = report.get("worst_seed") or {}
    if float(worst.get("max_relative_entropy_drop", 1.0)) > 0.25:
        failures.append("predictive_entropy_gate")
    if int(worst.get("negative_false_allow_max", 1)) != 0:
        failures.append("negative_false_allow")
    if float(worst.get("ask_recall_min", 0.0)) < 0.95:
        failures.append("ask_recall")
    if float(worst.get("repair_recall_min", 0.0)) < 0.95:
        failures.append("repair_recall")
    if float(worst.get("variant_recall_min", 0.0)) < 0.90:
        failures.append("variant_recall")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_observation_blocked_capability" if failures else "passed_candidate_diagnostic",
        "failures": sorted(set(failures)),
        "report_internal_sha256": internal,
        "report_file_sha256": report_file_sha256,
        "gate_failures": list(gate.get("failures") or []),
        "split_counts": dict(gate.get("split_counts") or {}),
        "worst_seed": {key: worst.get(key) for key in ("max_relative_entropy_drop", "entropy_gate_passed", "negative_false_allow_max", "ask_recall_min", "repair_recall_min", "variant_recall_min", "positive_recall_min")},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit PG-347 multi-implementation structured slot candidate report")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8-sig"))
    audit = audit_report(report, report_file_sha256=_sha_file(args.report))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
