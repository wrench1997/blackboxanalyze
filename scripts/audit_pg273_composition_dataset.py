"""Audit PG-273 v1→v2 implementation-disjoint abstract dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "research" / "pg273_composition_dataset_v1.json"


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def main() -> None:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    failures: list[str] = []

    def check(name: str, value: bool) -> None:
        if not value:
            failures.append(name)

    records = list(data.get("records", []))
    train = [row for row in records if row.get("split") == "implementation_v1_train"]
    holdout = [row for row in records if row.get("split") == "implementation_v2_holdout"]
    check("dataset_hash", data.get("dataset_sha256") == sha({key: value for key, value in data.items() if key != "dataset_sha256"}))
    check("counts_36_36", len(train) == 36 and len(holdout) == 36)
    check("implementation_disjoint", {row.get("implementation") for row in train} == {"heterogeneous_surface_v1"} and {row.get("implementation") for row in holdout} == {"heterogeneous_surface_v2"})
    check("positive_support", sum(bool(row.get("labels", {}).get("expected_positive")) for row in train) == 6 and sum(bool(row.get("labels", {}).get("expected_positive")) for row in holdout) == 6)
    check("record_ids_disjoint", not ({row.get("record_id") for row in train} & {row.get("record_id") for row in holdout}))
    check("context_firewall", all(not any(term in token.casefold() for term in ("oracle", "payload", "response", "body_sha", "surface_role", "typed_surface")) for row in records for token in row.get("context_tokens", [])))
    check("question_tokens", all(any(token.startswith("question=") for token in row.get("context_tokens", [])) for row in records))
    check("observation_tokens", all(any(token.startswith("observe_") for token in row.get("context_tokens", [])) for row in records))
    check("teacher_score_fields", all(set(row.get("teacher_components", {})) == {"scope_and_safety", "information_completeness", "probe_utility", "failure_diagnosis", "repair_quality", "oracle_and_evidence_alignment", "calibrated_abstain"} for row in records))
    check("preference_fields", all(row.get("preference", {}).get("rejected_target_tokens") != row.get("target_tokens") for row in records))
    check("raw_values_not_stored", all(row.get("raw_payload_strings_stored") is False and row.get("raw_response_bodies_stored") is False and row.get("oracle_in_context") is False for row in records))
    check("promotion_blocked", data.get("training_contract", {}).get("promotion_blocked") is True and data.get("training_contract", {}).get("memory_promotion_blocked") is True)
    audit = {
        "audit_id": "pg273-composition-dataset-independent-audit-v1",
        "status": "passed" if not failures else "failed",
        "all_required_fields_complete": not failures,
        "audit_checks": {"dataset_hash": "dataset_hash" not in failures, "split_counts": "counts_36_36" not in failures, "implementation_disjoint": "implementation_disjoint" not in failures and "record_ids_disjoint" not in failures, "positive_support": "positive_support" not in failures, "context_firewall": "context_firewall" not in failures and "question_tokens" not in failures and "observation_tokens" not in failures, "teacher_preference_process_fields": "teacher_score_fields" not in failures and "preference_fields" not in failures, "raw_and_promotion_gates": "raw_values_not_stored" not in failures and "promotion_blocked" not in failures},
        "dataset": str(DATASET.relative_to(ROOT)),
        "train_count": len(train),
        "holdout_count": len(holdout),
        "failures": failures,
    }
    audit["audit_sha256"] = sha(audit)
    output = ROOT / "research" / "pg273_composition_dataset_audit_v1.json"
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
