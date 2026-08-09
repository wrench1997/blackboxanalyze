"""Build PG-294 availability/feedback-state data from PG-293 abstractions."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg294_active_repair import audit_records, build_records  # noqa: E402
from app.pg293_failure_next_action import sha256_json  # noqa: E402


RESEARCH = ROOT / "research"
SOURCE = RESEARCH / "pg293_failure_next_action_dataset_v1.json"
DATASET = RESEARCH / "pg294_active_repair_dataset_v1.json"
AUDIT = RESEARCH / "pg294_active_repair_dataset_audit_v1.json"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    source = load(SOURCE)
    records = build_records(list(source.get("records") or []))
    train = [row for row in records if row.get("split") == "train" and row.get("training_eligible") is True]
    source_holdout = [row for row in records if row.get("split") == "source_holdout"]
    seed_holdout = [row for row in records if row.get("split") == "seed_holdout"]
    hard_negative = [row for row in records if row.get("split") == "hard_negative_eval"]
    audit = audit_records(records)
    dataset = {
        "schema_version": "pg294-active-repair-dataset-v1",
        "purpose": "oracle-blind typed availability / feedback state active repair",
        "source": {
            "path": str(SOURCE.relative_to(ROOT).as_posix()),
            "sha256": source.get("dataset_sha256"),
        },
        "records": records,
        "counts": {
            "total": len(records),
            "train": len(train),
            "source_holdout": len(source_holdout),
            "seed_holdout": len(seed_holdout),
            "hard_negative_eval": len(hard_negative),
            "positive_safe": sum(int(row.get("safe_to_send", False)) for row in records),
            "state_cells": sorted({str(row.get("state_id")) for row in records}),
        },
        "contract": {
            "typed_availability_is_not_verdict": True,
            "context_excludes_oracle_verdict": True,
            "context_excludes_family_lane_route": True,
            "target_is_abstract": True,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
            "hard_negative_eval_only": True,
            "wire_emission_allowed": False,
            "memory_promotion_allowed": False,
        },
    }
    dataset["dataset_sha256"] = sha256_json(dataset)
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_payload = {
        "audit_id": "pg294-active-repair-independent-audit-v1",
        "schema_version": "pg294-active-repair-audit-v1",
        "dataset": str(DATASET.relative_to(ROOT).as_posix()),
        "dataset_sha256": dataset["dataset_sha256"],
        **audit,
        "checks": {
            **dict(audit.get("checks") or {}),
            "train_present": bool(train),
            "source_holdout_present": bool(source_holdout),
            "seed_holdout_present": bool(seed_holdout),
            "hard_negative_present": bool(hard_negative),
            "progress_positive_present": any(row.get("state_id") == "progress" and row.get("safe_to_send") for row in records),
        },
        "interpretation": "PG-294 只验证可用性/反馈状态的抽象组合与主动修复；没有真实 evaluator gold 时不得宣称 payload 成功。",
    }
    audit_payload["status"] = "passed" if all(bool(value) for value in audit_payload["checks"].values()) else "failed"
    audit_payload["audit_sha256"] = sha256_json(audit_payload)
    AUDIT.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": str(DATASET.relative_to(ROOT)), "audit": str(AUDIT.relative_to(ROOT)), "counts": dataset["counts"], "status": audit_payload["status"], "dataset_sha256": dataset["dataset_sha256"], "audit_sha256": audit_payload["audit_sha256"]}, ensure_ascii=False, indent=2))
    if audit_payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
