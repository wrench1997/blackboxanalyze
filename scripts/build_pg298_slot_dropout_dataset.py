"""Build PG-298 canonical-slot dropout augmentation without oracle leakage."""

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
SOURCE = RESEARCH / "pg297_slot_canonical_dataset_v1.json"
DATASET = RESEARCH / "pg298_slot_dropout_dataset_v1.json"
AUDIT = RESEARCH / "pg298_slot_dropout_audit_v1.json"
SURFACE_KEYS = ("method", "channel", "status", "field_bucket")
PROCESS_KEYS = ("history_bucket", "candidate_error_shape", "backend_observed", "database_health", "binding_valid", "result_mismatch")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dropout(row: dict, pattern: str) -> dict:
    clone = copy.deepcopy(row)
    values = set(SURFACE_KEYS if pattern == "surface_unknown" else PROCESS_KEYS if pattern == "process_unknown" else SURFACE_KEYS + PROCESS_KEYS)
    clone["context_tokens"] = [
        (token.split("=", 1)[0] + "=unknown") if "=" in str(token) and token.split("=", 1)[0] in values else token
        for token in clone.get("context_tokens", [])
    ]
    clone["source_group"] = "slot_dropout_augmentation"
    clone["dropout_pattern"] = pattern
    clone["record_id"] = f"pg298:{pattern}:{sha256_json(clone['context_tokens'] + list(clone.get('target_tokens', [])))[:16]}"
    clone["record_sha256"] = sha256_json(clone)
    return clone


def main() -> None:
    source = load(SOURCE)
    source_records = list(source.get("records") or [])
    train_base = [row for row in source_records if row.get("split") == "train" and row.get("training_eligible") is True]
    holdout = [row for row in source_records if row.get("split") == "implementation_holdout"]
    hard = [row for row in source_records if row.get("split") == "hard_negative_eval"]
    records = list(train_base) + list(holdout) + list(hard)
    for row in train_base:
        for pattern in ("surface_unknown", "process_unknown", "all_unknown"):
            records.append(dropout(row, pattern))
    audit = audit_canonical_records(records)
    train = [row for row in records if row.get("split") == "train" and row.get("training_eligible") is True]
    dataset = {"schema_version": "pg298-slot-dropout-dataset-v1", "purpose": "slot dropout to prevent causal MoE memorizing surface combinations", "source": {"path": str(SOURCE.relative_to(ROOT).as_posix()), "sha256": source.get("dataset_sha256")}, "records": records, "counts": {"total": len(records), "train": len(train), "implementation_holdout": len(holdout), "hard_negative_eval": len(hard), "dropout_patterns": ["surface_unknown", "process_unknown", "all_unknown"]}, "contract": {"canonical_slot_order": True, "unknown_value_dropout": True, "oracle_blind": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "wire_emission_allowed": False, "memory_promotion_allowed": False}}
    dataset["dataset_sha256"] = sha256_json(dataset)
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_payload = {"audit_id": "pg298-slot-dropout-audit-v1", "schema_version": "pg298-slot-dropout-audit-v1", "dataset": str(DATASET.relative_to(ROOT).as_posix()), "dataset_sha256": dataset["dataset_sha256"], **audit, "checks": {**dict(audit.get("checks") or {}), "train_present": bool(train), "holdout_present": bool(holdout), "hard_negative_present": bool(hard), "dropout_present": any(row.get("dropout_pattern") for row in records)}}
    audit_payload["status"] = "passed" if all(bool(value) for value in audit_payload["checks"].values()) else "failed"
    audit_payload["audit_sha256"] = sha256_json(audit_payload)
    AUDIT.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": str(DATASET.relative_to(ROOT)), "audit": str(AUDIT.relative_to(ROOT)), "counts": dataset["counts"], "status": audit_payload["status"], "dataset_sha256": dataset["dataset_sha256"], "audit_sha256": audit_payload["audit_sha256"]}, ensure_ascii=False, indent=2))
    if audit_payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
