"""Build PG-299 question-first causal target data."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg293_failure_next_action import sha256_json  # noqa: E402
from app.pg297_slot_canonical import audit_canonical_records  # noqa: E402


RESEARCH = ROOT / "research"
SOURCE = RESEARCH / "pg298_slot_dropout_dataset_v1.json"
DATASET = RESEARCH / "pg299_question_first_dataset_v1.json"
AUDIT = RESEARCH / "pg299_question_first_audit_v1.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def reorder(row: dict, pattern: str | None = None) -> dict:
    clone = copy.deepcopy(row)
    target = list(clone.get("target_tokens") or [])
    values = {token.split("=", 1)[0]: token for token in target if "=" in str(token)}
    ordered = ["[TARGET_BOS]", values.get("question", "question=none"), values.get("next_action", "next_action=abstain"), values.get("repair_action", "repair_action=none"), values.get("safe_to_send", "safe_to_send=0"), "[TARGET_EOS]"]
    clone["target_tokens"] = ordered
    clone["target_order"] = "question_first"
    if pattern:
        clone["context_tokens"] = [("phase=unknown" if str(token).startswith("phase=") else token) for token in clone.get("context_tokens", [])]
        clone["source_group"] = "phase_dropout_augmentation"
        clone["dropout_pattern"] = pattern
    clone["record_id"] = f"pg299:{pattern or 'base'}:{sha256_json(clone.get('context_tokens', []) + ordered)[:16]}"
    clone["record_sha256"] = sha256_json(clone)
    return clone


def main() -> None:
    source = load(SOURCE)
    src = list(source.get("records") or [])
    train_base = [row for row in src if row.get("split") == "train" and row.get("training_eligible") is True]
    holdout = [row for row in src if row.get("split") == "implementation_holdout"]
    hard = [row for row in src if row.get("split") == "hard_negative_eval"]
    records = [reorder(row) for row in src]
    for row in train_base:
        records.append(reorder(row, "phase_unknown"))
    train = [row for row in records if row.get("split") == "train" and row.get("training_eligible") is True]
    audit = audit_canonical_records(records)
    dataset = {"schema_version": "pg299-question-first-dataset-v1", "purpose": "question-first causal next-token target with phase dropout", "source": {"path": str(SOURCE.relative_to(ROOT).as_posix()), "sha256": source.get("dataset_sha256")}, "records": records, "counts": {"total": len(records), "train": len(train), "implementation_holdout": len(holdout), "hard_negative_eval": len(hard), "question_first": True, "phase_dropout_rows": sum(int(row.get("dropout_pattern") == "phase_unknown") for row in records)}, "contract": {"question_token_first": True, "causal_next_token": True, "canonical_slot_order": True, "oracle_blind": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "wire_emission_allowed": False, "memory_promotion_allowed": False}}
    dataset["dataset_sha256"] = sha256_json(dataset)
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_payload = {"audit_id": "pg299-question-first-audit-v1", "schema_version": "pg299-question-first-audit-v1", "dataset": str(DATASET.relative_to(ROOT).as_posix()), "dataset_sha256": dataset["dataset_sha256"], **audit, "checks": {**dict(audit.get("checks") or {}), "train_present": bool(train), "holdout_present": bool(holdout), "hard_negative_present": bool(hard), "question_first": all(row.get("target_tokens", [])[1].startswith("question=") for row in records), "phase_dropout_present": any(row.get("dropout_pattern") == "phase_unknown" for row in records)}}
    audit_payload["status"] = "passed" if all(bool(value) for value in audit_payload["checks"].values()) else "failed"
    audit_payload["audit_sha256"] = sha256_json(audit_payload)
    AUDIT.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": str(DATASET.relative_to(ROOT)), "audit": str(AUDIT.relative_to(ROOT)), "counts": dataset["counts"], "status": audit_payload["status"], "dataset_sha256": dataset["dataset_sha256"], "audit_sha256": audit_payload["audit_sha256"]}, ensure_ascii=False, indent=2))
    if audit_payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
