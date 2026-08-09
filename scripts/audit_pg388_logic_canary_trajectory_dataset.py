"""Fail-closed audit for the PG-388 abstract canary trajectory dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:  # package import for tests
    from scripts.build_pg388_logic_canary_trajectory_dataset import CASES, PHASES, ROLES, _sha
except ModuleNotFoundError:  # direct CLI execution from scripts/
    from build_pg388_logic_canary_trajectory_dataset import CASES, PHASES, ROLES, _sha

SCHEMA_VERSION = "pg388-logic-canary-trajectory-audit-v1"


def audit_dataset(document: dict[str, Any]) -> dict[str, Any]:
    rows = document.get("rows") if isinstance(document, dict) else None
    reasons: list[str] = []
    if not isinstance(rows, list):
        reasons.append("rows_missing")
        rows = []
    if len(rows) != 90:
        reasons.append("record_count_mismatch")
    expected_cases = set(CASES)
    observed_cases = {str(row.get("case_ref")) for row in rows if isinstance(row, dict)}
    if observed_cases != expected_cases:
        reasons.append("case_matrix_mismatch")
    invalid_rows = 0
    raw_hits = 0
    target_leaks = 0
    train_contexts: set[tuple[str, ...]] = set()
    holdout_contexts: set[tuple[str, ...]] = set()
    train_pairs: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    holdout_pairs: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    seen: set[str] = set()
    forbidden_context = ("vulnerable_effect", "negative_control_clean", "evaluator", "raw_", "payload", "wire", "response_body")
    for row in rows:
        if not isinstance(row, dict):
            invalid_rows += 1
            continue
        core = {key: value for key, value in row.items() if key != "row_sha256"}
        if row.get("row_sha256") != _sha(core) or row.get("row_sha256") in seen:
            invalid_rows += 1
        seen.add(str(row.get("row_sha256")))
        if row.get("role") not in ROLES or row.get("phase") not in PHASES or row.get("training_eligible") is not False:
            invalid_rows += 1
        if any(bool(row.get(key)) for key in ("raw_source_stored", "raw_payload_stored", "raw_response_body_stored", "oracle_answer_in_context")):
            raw_hits += 1
        context_text = " ".join(str(token) for token in row.get("context_tokens", []))
        if any(marker in context_text for marker in forbidden_context):
            target_leaks += 1
        context = tuple(str(token) for token in row.get("context_tokens", []))
        target = tuple(str(token) for token in row.get("target_tokens", []))
        if row.get("split") == "train":
            train_contexts.add(context)
            train_pairs.add((context, target))
        elif row.get("split") == "implementation_holdout":
            holdout_contexts.add(context)
            holdout_pairs.add((context, target))
    if invalid_rows:
        reasons.append("invalid_rows")
    if raw_hits:
        reasons.append("raw_context_firewall_failure")
    if target_leaks:
        reasons.append("evaluator_or_raw_marker_in_context")
    context_overlap = len(train_contexts & holdout_contexts)
    pair_overlap = len(train_pairs & holdout_pairs)
    if context_overlap:
        reasons.append("cross_split_context_overlap")
    if pair_overlap:
        reasons.append("cross_split_context_target_overlap")
    status = "passed_candidate_trajectory_audit" if not reasons else "blocked_canary_trajectory_audit"
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "dataset_id": document.get("dataset_id"),
        "records": len(rows),
        "train": sum(isinstance(row, dict) and row.get("split") == "train" for row in rows),
        "implementation_holdout": sum(isinstance(row, dict) and row.get("split") == "implementation_holdout" for row in rows),
        "cases": len(expected_cases),
        "phases": len(PHASES),
        "roles": len(ROLES),
        "invalid_rows": invalid_rows,
        "raw_context_hits": raw_hits,
        "context_target_leaks": target_leaks,
        "cross_split_context_overlap": context_overlap,
        "cross_split_context_target_overlap": pair_overlap,
        "failure_reasons": reasons,
        "training_eligible": 0,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    report["audit_sha256"] = _sha(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="research/pg388_logic_canary_trajectory_dataset_v1.json")
    parser.add_argument("--output", default="research/pg388_logic_canary_trajectory_audit_v1.json")
    args = parser.parse_args()
    document = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    report = audit_dataset(document)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
