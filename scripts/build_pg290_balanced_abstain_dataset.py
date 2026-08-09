"""Build a smaller PG-290 abstain augmentation to test over-rejection."""

from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_PG289_SPEC = importlib.util.spec_from_file_location("pg289_builder", ROOT / "scripts/build_pg289_safe_abstain_dataset.py")
if _PG289_SPEC is None or _PG289_SPEC.loader is None:
    raise RuntimeError("cannot load PG-289 builder")
_PG289 = importlib.util.module_from_spec(_PG289_SPEC)
_PG289_SPEC.loader.exec_module(_PG289)
SIGNATURES = _PG289.SIGNATURES
SOURCE = _PG289.SOURCE
digest = _PG289.digest
make_record = _PG289.make_record


RESEARCH = ROOT / "research"
OUTPUT = RESEARCH / "pg290_balanced_abstain_dataset_v1.json"
AUDIT = RESEARCH / "pg290_balanced_abstain_dataset_audit_v1.json"
COUNT = 504


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    base_rows = [row for row in list(source.get("records") or []) if row.get("split") == "train"]
    records = []
    for index in range(COUNT):
        record = make_record(base_rows[index % len(base_rows)], index, SIGNATURES[index % len(SIGNATURES)])
        record["record_id"] = record["record_id"].replace("pg289:", "pg290:", 1)
        record["source_group_id"] = "pg290:balanced-unresolved-evaluator-counterfactual"
        record["source"] = "pg290_counterfactual"
        records.append(record)
    dataset = {
        "schema_version": "pg290-balanced-abstain-dataset-v1",
        "purpose": "smaller training-only evaluator-gap counterfactual mix to measure abstain/recall tradeoff",
        "source": {"base_dataset": str(SOURCE.relative_to(ROOT).as_posix()), "base_dataset_sha256": source["dataset_sha256"], "generation": "train-template-only", "signatures": list(SIGNATURES), "augmentation_count": COUNT},
        "records": records,
        "counts": {"train": COUNT, "total": COUNT, "signature_count": len(SIGNATURES)},
        "training_contract": {"remote_a800_required": True, "literal_probe_values_out_of_context": True, "raw_response_bodies_out_of_context": True, "family_labels_out_of_context": True, "hard_negative_eval_only": True, "memory_promotion_allowed": False},
        "scientific_contract": {"not_real_application_gold": True, "not_a_vulnerability_claim": True, "must_evaluate_on_source_heldout_and_family_holdout": True},
    }
    dataset["dataset_sha256"] = digest(dataset)
    OUTPUT.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checks = {
        "base_hash_present": dataset["source"]["base_dataset_sha256"] == source["dataset_sha256"],
        "row_quota": len(records) == COUNT,
        "train_only": all(row.get("split") == "train" for row in records),
        "target_abstain": all(row.get("target", {}).get("next_action") == "abstain" and row.get("target", {}).get("safe_to_send") is False for row in records),
        "family_agnostic": all(row.get("family") is None for row in records),
        "no_raw_probe": all(not row.get("raw_payload_strings_stored") and not row.get("raw_response_bodies_stored") for row in records),
        "signature_coverage": {signature: sum(1 for row in records if f"failure_signature={signature}" in row.get("context_tokens", [])) for signature in SIGNATURES},
    }
    checks["all_signature_rows_present"] = all(value > 0 for value in checks["signature_coverage"].values())
    audit = {"audit_id": "pg290-balanced-abstain-dataset-independent-audit-v1", "status": "passed" if all(bool(value) for key, value in checks.items() if key != "signature_coverage") else "failed", "dataset": str(OUTPUT.relative_to(ROOT).as_posix()), "dataset_sha256": dataset["dataset_sha256"], "checks": checks, "interpretation": "PG-290 用较小 unresolved/evaluator-gap 训练混合测试安全拒答与可用 candidate 的权衡；不含真实 gold。"}
    audit["audit_sha256"] = digest(audit)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "records": COUNT, "dataset_sha256": dataset["dataset_sha256"], "audit_sha256": audit["audit_sha256"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
