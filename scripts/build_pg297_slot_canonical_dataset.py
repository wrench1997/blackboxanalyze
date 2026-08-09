"""Build PG-297 canonical slot-token data from the PG-296B split."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg293_failure_next_action import sha256_json  # noqa: E402
from app.pg297_slot_canonical import audit_canonical_records, canonicalize_context, canonicalize_record  # noqa: E402


RESEARCH = ROOT / "research"
SOURCE = RESEARCH / "pg296b_missing_augmentation_dataset_v1.json"
DATASET = RESEARCH / "pg297_slot_canonical_dataset_v1.json"
AUDIT = RESEARCH / "pg297_slot_canonical_audit_v1.json"


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8-sig"))
    records = [canonicalize_record(row) for row in list(source.get("records") or [])]
    audit = audit_canonical_records(records)
    train = [row for row in records if row.get("split") == "train" and row.get("training_eligible") is True]
    holdout = [row for row in records if row.get("split") == "implementation_holdout"]
    hard = [row for row in records if row.get("split") == "hard_negative_eval"]
    dataset = {
        "schema_version": "pg297-slot-canonical-dataset-v1",
        "purpose": "canonical key/value slot tokenization before causal MoE assembly",
        "source": {"path": str(SOURCE.relative_to(ROOT).as_posix()), "sha256": source.get("dataset_sha256")},
        "records": records,
        "counts": {"total": len(records), "train": len(train), "implementation_holdout": len(holdout), "hard_negative_eval": len(hard), "required_slots": ["method", "channel", "status", "field_bucket", "typed_available", "feedback_state", "replay_ready", "evidence_present"]},
        "contract": {"canonical_slot_order": True, "unknown_values_bucketed": True, "oracle_blind": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "wire_emission_allowed": False, "memory_promotion_allowed": False},
    }
    dataset["dataset_sha256"] = sha256_json(dataset)
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_payload = {"audit_id": "pg297-slot-canonical-audit-v1", "schema_version": "pg297-slot-canonical-audit-v1", "dataset": str(DATASET.relative_to(ROOT).as_posix()), "dataset_sha256": dataset["dataset_sha256"], **audit, "checks": {**dict(audit.get("checks") or {}), "train_present": bool(train), "holdout_present": bool(holdout), "hard_negative_present": bool(hard), "slot_order_stable": all(canonicalize_context(row.get("context_tokens", [])) == row.get("context_tokens", []) for row in records)}}
    audit_payload["status"] = "passed" if all(bool(value) for value in audit_payload["checks"].values()) else "failed"
    audit_payload["audit_sha256"] = sha256_json(audit_payload)
    AUDIT.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": str(DATASET.relative_to(ROOT)), "audit": str(AUDIT.relative_to(ROOT)), "counts": dataset["counts"], "status": audit_payload["status"], "dataset_sha256": dataset["dataset_sha256"], "audit_sha256": audit_payload["audit_sha256"]}, ensure_ascii=False, indent=2))
    if audit_payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
