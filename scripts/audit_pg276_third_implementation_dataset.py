"""Audit PG-276 train/v2-canary/v3-holdout split and context firewall."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "research" / "pg276_third_implementation_dataset_v1.json"
AUDIT = ROOT / "research" / "pg276_third_implementation_dataset_audit_v1.json"


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

    rows = list(data.get("records", []))
    train = [x for x in rows if x.get("split") == "implementation_v1_train"]
    canary = [x for x in rows if x.get("split") == "implementation_v2_canary"]
    holdout = [x for x in rows if x.get("split") == "implementation_v3_holdout"]
    check("dataset_hash", data.get("dataset_sha256") == sha({k: v for k, v in data.items() if k != "dataset_sha256"}))
    check("counts", len(train) == 36 and len(canary) == 36 and len(holdout) == 36)
    check("implementation_disjoint", {x.get("implementation") for x in train} == {"heterogeneous_surface_v1"} and {x.get("implementation") for x in canary} == {"heterogeneous_surface_v2"} and {x.get("implementation") for x in holdout} == {"heterogeneous_surface_v3"})
    train_ids = {x.get("record_id") for x in train}
    canary_ids = {x.get("record_id") for x in canary}
    holdout_ids = {x.get("record_id") for x in holdout}
    check("record_disjoint", not ((train_ids & canary_ids) or (train_ids & holdout_ids) or (canary_ids & holdout_ids)))
    check("positive_support", sum(bool(x.get("labels", {}).get("expected_positive")) for x in train) == 6 and sum(bool(x.get("labels", {}).get("expected_positive")) for x in canary) == 6 and sum(bool(x.get("labels", {}).get("expected_positive")) for x in holdout) == 6)
    check("context_firewall", all(not any(word in token.casefold() for word in ("oracle", "payload", "response", "body_sha", "surface_role", "typed_surface")) for row in rows for token in row.get("context_tokens", [])))
    check("question_observation_tokens", all(any(token.startswith("question=") for token in row.get("context_tokens", [])) and any(token.startswith("observe_") for token in row.get("context_tokens", [])) for row in rows))
    check("raw_off", all(row.get("raw_payload_strings_stored") is False and row.get("raw_response_bodies_stored") is False and row.get("oracle_in_context") is False for row in rows))
    check("old_canary_not_training", data.get("training_contract", {}).get("old_canary_not_used_for_update") is True and data.get("split_contract", {}).get("promotion_blocked") is True)
    audit = {"audit_id": "pg276-third-implementation-dataset-audit-v1", "status": "passed" if not failures else "failed", "audit_checks": {"dataset_hash": "dataset_hash" not in failures, "counts": "counts" not in failures, "implementation_disjoint": "implementation_disjoint" not in failures, "record_disjoint": "record_disjoint" not in failures, "positive_support": "positive_support" not in failures, "context_firewall": "context_firewall" not in failures and "question_observation_tokens" not in failures, "raw_off": "raw_off" not in failures, "old_canary_not_training": "old_canary_not_training" not in failures}, "counts": {"train": len(train), "old_canary": len(canary), "holdout": len(holdout)}, "failures": failures}
    audit["audit_sha256"] = sha(audit)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
