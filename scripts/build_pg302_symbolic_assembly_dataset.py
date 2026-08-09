"""Project PG-301 records to symbolic slot-reference targets for PG-302."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg293_failure_next_action import sha256_json  # noqa: E402
from app.pg302_symbolic_assembly import audit_symbolic_records, symbolic_record  # noqa: E402


RESEARCH = ROOT / "research"
SOURCE = RESEARCH / "pg301_payload_assembly_dataset_v1.json"
DATASET = RESEARCH / "pg302_symbolic_assembly_dataset_v1.json"
AUDIT = RESEARCH / "pg302_symbolic_assembly_audit_v1.json"


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8-sig"))
    records = [symbolic_record(row) for row in list(source.get("records") or [])]
    audit = audit_symbolic_records(records)
    dataset = {
        "schema_version": "pg302-symbolic-assembly-dataset-v1",
        "purpose": "causal model predicts slot references; binder resolves only bounded abstract values",
        "source": {"path": str(SOURCE.relative_to(ROOT).as_posix()), "sha256": source.get("dataset_sha256"), "literal_payload_strings_stored": False, "wire_emission": False},
        "records": records,
        "counts": {
            "total": len(records),
            "train": sum(row.get("split") == "train" for row in records),
            "implementation_holdout": sum(row.get("split") == "implementation_holdout" for row in records),
            "hard_negative_eval": sum(row.get("split") == "hard_negative_eval" for row in records),
            "counterfactual_rows": sum(1 for row in records if str(row.get("counterfactual_group", "")).startswith("cf-hidden")),
        },
        "contract": {
            "causal_next_token": True,
            "symbolic_slot_references": True,
            "deterministic_binder": True,
            "abstract_values_only": True,
            "typed_oracle_required": True,
            "fresh_reset_required": True,
            "negative_control_required": True,
            "literal_payload_strings_stored": False,
            "wire_emission_allowed": False,
            "memory_promotion_allowed": False,
        },
    }
    dataset["dataset_sha256"] = sha256_json(dataset)
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_payload = {"audit_id": "pg302-symbolic-assembly-audit-v1", "schema_version": "pg302-symbolic-assembly-audit-v1", "dataset": str(DATASET.relative_to(ROOT).as_posix()), "dataset_sha256": dataset["dataset_sha256"], **audit}
    audit_payload["audit_sha256"] = sha256_json(audit_payload)
    AUDIT.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": str(DATASET.relative_to(ROOT)), "audit": str(AUDIT.relative_to(ROOT)), "counts": dataset["counts"], "status": audit_payload["status"], "dataset_sha256": dataset["dataset_sha256"], "audit_sha256": audit_payload["audit_sha256"]}, ensure_ascii=False, indent=2))
    if audit_payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
