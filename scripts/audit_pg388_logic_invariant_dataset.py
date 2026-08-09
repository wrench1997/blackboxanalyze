"""Read-only audit for the PG-388 abstract business-logic dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "pg388-logic-invariant-dataset-audit-v1"
FORBIDDEN = ("http://", "https://", "payload=", "payload:", "wire=", "wire:", "response_body=", "response_body:", "credential=", "credential:", "cookie_value=", "cookie_value:", "<script")


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _marker_hits(value: Any) -> list[str]:
    text = json.dumps(value, ensure_ascii=False).casefold()
    return sorted({marker for marker in FORBIDDEN if marker in text})


def _row_hash_valid(row: Mapping[str, Any]) -> bool:
    declared = row.get("row_sha256")
    if not isinstance(declared, str):
        return False
    body = {key: value for key, value in row.items() if key != "row_sha256"}
    return _sha(body) == declared


def audit_dataset(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    artifact = json.loads(source.read_text(encoding="utf-8-sig"))
    if artifact.get("schema_version") != "pg388-logic-invariant-dataset-v1":
        raise ValueError("pg388_dataset_schema_mismatch")
    rows = artifact.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("pg388_dataset_rows_missing")
    split_counts = Counter(str(row.get("split")) for row in rows if isinstance(row, Mapping))
    role_counts = Counter(str(row.get("role")) for row in rows if isinstance(row, Mapping))
    markers: set[str] = set()
    invalid_hash_rows = 0
    invalid_rows = 0
    unsafe_rows = 0
    for row in rows:
        if not isinstance(row, Mapping):
            invalid_rows += 1
            continue
        if not _row_hash_valid(row):
            invalid_hash_rows += 1
        markers.update(_marker_hits({"context_tokens": row.get("context_tokens"), "target_tokens": row.get("target_tokens"), "logic_context": row.get("logic_context")}))
        if row.get("training_eligible") is not False or row.get("raw_source_stored") is not False or row.get("raw_payload_stored") is not False or row.get("raw_response_body_stored") is not False or row.get("oracle_answer_in_context") is not False:
            unsafe_rows += 1
    entropy = artifact.get("information_preservation", {}).get("entropy", {})
    required_axes = ("logic_surface", "logic_invariant", "precondition", "transition", "counterfactual", "observation_shape", "failure_shape", "feedback_state", "role")
    entropy_missing = [axis for axis in required_axes if not isinstance(entropy.get(axis), (int, float)) or float(entropy.get(axis, 0)) <= 0]
    source_contract = artifact.get("source_contract", {})
    promotion = artifact.get("promotion", {})
    failures = []
    if invalid_rows:
        failures.append("invalid_rows")
    if invalid_hash_rows:
        failures.append("row_hash_mismatch")
    if markers:
        failures.append("raw_or_evaluator_marker")
    if unsafe_rows:
        failures.append("context_or_training_firewall")
    if entropy_missing:
        failures.append("entropy_axis_missing")
    if source_contract.get("live_rows_emitted") is not False or source_contract.get("typed_evidence") is not False:
        failures.append("source_contract_open")
    if promotion.get("training_allowed") is not False or promotion.get("vulnerability_claim_allowed") is not False:
        failures.append("promotion_open")
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_candidate_audit" if not failures else "blocked_logic_dataset_audit",
        "dataset_path": str(source).replace("\\", "/"),
        "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "counts": {"records": len(rows), "split": dict(split_counts), "roles": dict(role_counts), "invalid_rows": invalid_rows, "invalid_hash_rows": invalid_hash_rows, "unsafe_rows": unsafe_rows},
        "information_preservation": {"entropy": {key: float(value) for key, value in entropy.items() if isinstance(value, (int, float))}, "required_axes": list(required_axes), "missing_axes": entropy_missing},
        "context_firewall": {"marker_hits": sorted(markers), "raw_context_allowed": False, "external_network": False},
        "source_contract": {"fresh_role_reset": False, "candidate_reference_negative_replay": False, "typed_evidence": False, "operator_reviewed": False, "live_rows_emitted": False},
        "training_eligible": 0,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "failures": failures,
    }
    audit["audit_sha256"] = _sha({key: value for key, value in audit.items() if key != "audit_sha256"})
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="research/pg388_logic_invariant_dataset_v1.json")
    parser.add_argument("--output", default="research/pg388_logic_invariant_dataset_audit_v1.json")
    args = parser.parse_args()
    report = audit_dataset(args.dataset)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"], "counts": report["counts"], "failures": report["failures"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
