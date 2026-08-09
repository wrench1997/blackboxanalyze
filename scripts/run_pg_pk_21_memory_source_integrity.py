"""PG-PK-21: offline provenance and independent-seed memory-gate audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.memory_promotion_gate import assess_memory_promotion  # noqa: E402


PROTOCOL_PATH = ROOT / "research" / "pg_pk_21_memory_source_integrity_protocol_v1.json"
REPORT_PATH = ROOT / "research" / "pg_pk_21_memory_source_integrity_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg_pk_21_memory_source_integrity_v1.md"


def _row(dataset: str, seed: int, source: str | None, evidence: str, index: int) -> dict[str, Any]:
    row = {
        "dataset_id": dataset,
        "sampling_seed": seed,
        "target_instance_id": f"{dataset}-target-{seed}",
        "rule_key": "xss::source_integrity",
        "accepted": True,
        "oracle_revalidated": True,
        "false_positive": False,
        "evidence_hash": evidence,
        "local_only": True,
    }
    if source is not None:
        row["source_hash"] = source
    return row


def _synthetic_cases() -> list[dict[str, Any]]:
    one_source = [_row(dataset, seed, "s" * 64, "e" * 64, index) for index, (dataset, seed) in enumerate((
        (dataset, seed) for dataset in ("a", "b", "c") for seed in (1, 2)
    ))]
    missing_source = [_row(dataset, seed, None, f"{dataset}{seed}" * 32, index) for index, (dataset, seed) in enumerate((
        (dataset, seed) for dataset in ("a", "b", "c") for seed in (1, 2)
    ))]
    repeated_seed_evidence = [_row(dataset, seed, f"{dataset}" * 64, f"{dataset}" * 64, index) for index, (dataset, seed) in enumerate((
        (dataset, seed) for dataset in ("a", "b", "c") for seed in (1, 2)
    ))]
    valid = [_row(dataset, seed, f"{dataset}" * 64, f"{dataset}{seed}" * 32, index) for index, (dataset, seed) in enumerate((
        (dataset, seed) for dataset in ("a", "b", "c") for seed in (1, 2)
    ))]
    return [
        {"case_id": "one_source_three_labels", "rows": one_source, "expected": "quarantine"},
        {"case_id": "missing_source_hash", "rows": missing_source, "expected": "quarantine"},
        {"case_id": "repeated_evidence_across_seeds", "rows": repeated_seed_evidence, "expected": "quarantine"},
        {"case_id": "three_sources_distinct_seed_manifests", "rows": valid, "expected": "promote"},
    ]


def _evaluate(case: dict[str, Any]) -> dict[str, Any]:
    result = assess_memory_promotion("xss::source_integrity", case["rows"])
    return {
        "case_id": case["case_id"],
        "expected_status": case["expected"],
        "observed_status": result["status"],
        "passed": result["status"] == case["expected"],
        "summary": result["summary"],
        "reasons": result["reasons"],
        "per_dataset": result["per_dataset"],
    }


def _real_report_case(case_id: str, path: Path, rule_key: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in report.get("promotion_ledger", []) if row.get("rule_key") == rule_key]
    result = assess_memory_promotion(rule_key, rows)
    return {
        "case_id": case_id,
        "source_report": str(path.relative_to(ROOT)),
        "expected_status": "promote",
        "observed_status": result["status"],
        "passed": result["status"] == "promote",
        "summary": result["summary"],
        "reasons": result["reasons"],
    }


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    cases = [_evaluate(case) for case in _synthetic_cases()]
    cases.append(_real_report_case(
        "real_pg15_sql_v3",
        ROOT / "research" / "pg_pk_15_sql_v3_cross_source_promotion_v1.json",
        "injection::synthetic_sql_channel",
    ))
    for family in ("access_control", "logic"):
        cases.append(_real_report_case(
            f"real_pg18_{family}_v3",
            ROOT / "research" / "pg_pk_18_logic_v3_cross_source_promotion_v1.json",
            f"{family}::typed_boundary",
        ))
    result = {
        "schema_version": "sift-pg-pk-21-memory-source-integrity-report-v1",
        "protocol_id": protocol["protocol_id"],
        "status": "pass" if all(case["passed"] for case in cases) else "fail",
        "case_count": len(cases),
        "passed_case_count": sum(int(case["passed"]) for case in cases),
        "cases": cases,
        "note": "The PG-PK-04 summary has no raw promotion ledger or source hash field, so its XSS promotion cannot be revalidated and remains diagnostic_only until fresh evidence is collected.",
        "local_only": True,
        "model_or_checkpoint_modified": False,
        "oracle_or_payload_modified": False,
    }
    REPORT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(
        "# PG-PK-21 memory source integrity 实验\n\n"
        f"状态：`{result['status']}`；通过：{result['passed_case_count']}/{result['case_count']}。\n\n"
        "修复前，同一 source hash 通过 dataset 标签重命名即可满足多数据集门；同一 evidence hash 通过 seed 标签复制也会被当成独立采样。"
        "修复后，source hash、独立 seed evidence manifest 和去重后的证据计数共同决定晋升。PG-PK-15/18 的真实三 source ledger 仍通过；PG-PK-04 因缺少可重放 ledger 保持 diagnostic_only。\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "protocol_id": result["protocol_id"],
        "status": result["status"],
        "passed": f"{result['passed_case_count']}/{result['case_count']}",
        "report": str(REPORT_PATH.relative_to(ROOT)),
        "markdown": str(MARKDOWN_PATH.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
