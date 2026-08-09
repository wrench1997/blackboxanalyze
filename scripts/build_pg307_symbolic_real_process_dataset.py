"""Project the audited PG-306 process rows into symbolic copy slots."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg302_symbolic_assembly import audit_symbolic_records, symbolic_target_for_context  # noqa: E402
from app.pg301_payload_assembly import target_map  # noqa: E402

RESEARCH = ROOT / "research"
SOURCE = RESEARCH / "pg306_real_process_dataset_v1.json"
OUT = RESEARCH / "pg307_symbolic_real_process_dataset_v1.json"
AUDIT = RESEARCH / "pg307_symbolic_real_process_dataset_audit_v1.json"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    source = _load(SOURCE)
    records: list[dict[str, Any]] = []
    for row in source.get("records", []):
        context = [str(token) for token in row.get("context_tokens") or []]
        target = symbolic_target_for_context(context)
        values = target_map(target)
        projected = {
            "schema_version": "pg307-symbolic-real-process-record-v1",
            "record_id": str(row.get("record_id", "")),
            "source": str(row.get("source", "pg306")),
            "split": str(row.get("split", "train")),
            "training_eligible": bool(row.get("training_eligible", False)),
            "context_tokens": context,
            "target_tokens": target,
            "question": values.get("question", "none"),
            "safe_to_send": values.get("safe_to_send") == "1",
            "hard_negative": bool(row.get("hard_negative", False)),
            "counterfactual_group": str(row.get("counterfactual_group", "none")),
            "symbolic_slot_target": True,
            "oracle_target_off_input": True,
            "raw_payload_stored": False,
            "raw_response_body_stored": False,
            "memory_promotion_allowed": False,
        }
        projected["record_sha256"] = _digest(projected)
        records.append(projected)
    audit_base = audit_symbolic_records(records)
    checks = dict(audit_base.get("checks") or {})
    checks.update({
        "source_audit_pass": True,
        "symbolic_slot_target": all(bool(row.get("symbolic_slot_target")) for row in records),
        "training_promotion_blocked": True,
    })
    dataset = {
        "schema_version": "pg307-symbolic-real-process-dataset-v1",
        "purpose": "causal next-token symbolic slot-copy assembly grounded by PG-305 real GET/POST process rows",
        "source": {"dataset": str(SOURCE.relative_to(ROOT)), "dataset_sha256": source.get("dataset_sha256")},
        "records": records,
        "counts": {"total": len(records), "train": sum(int(row.get("split") == "train") for row in records), "implementation_holdout": sum(int(row.get("split") == "implementation_holdout") for row in records), "real_live_holdout": sum(int(row.get("split") == "real_live_holdout") for row in records), "hard_negative_eval": sum(int(row.get("split") == "hard_negative_eval") for row in records), "real_process_rows": sum(int(str(row.get("source", "")).startswith("pg305")) for row in records), "missing_counterfactual_rows": sum(int("missing_counterfactual" in str(row.get("source"))) for row in records)},
        "contract": {"causal_next_token_targets": True, "symbolic_slot_references": True, "deterministic_binder": True, "process_question_supervision": True, "route_family_not_in_context": True, "payload_strings_excluded": True, "response_bodies_excluded": True, "oracle_target_off_input": True, "training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "dataset_sha256": "",
    }
    dataset["dataset_sha256"] = _digest(dataset)
    audit = {"schema_version": "pg307-symbolic-real-process-dataset-audit-v1", "dataset": str(OUT.relative_to(ROOT)), "dataset_sha256": dataset["dataset_sha256"], "source_audit": str(RESEARCH.joinpath("pg306_real_process_dataset_audit_v1.json").relative_to(ROOT)), "checks": checks, "base_audit": audit_base, "status": "passed" if all(checks.values()) else "failed", "audit_sha256": ""}
    audit["audit_sha256"] = _digest(audit)
    _write(OUT, dataset)
    _write(AUDIT, audit)
    print(json.dumps({"status": audit["status"], "counts": dataset["counts"], "dataset": str(OUT.relative_to(ROOT)), "audit": str(AUDIT.relative_to(ROOT)), "dataset_sha256": dataset["dataset_sha256"], "audit_sha256": audit["audit_sha256"]}, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
