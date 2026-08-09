"""PG-PK-19: verify that triage never drops caller-classified failures."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.experiment_engineering_triage import triage_failure  # noqa: E402


PROTOCOL_PATH = ROOT / "research" / "pg_pk_19_triage_signal_integrity_protocol_v1.json"
REPORT_PATH = ROOT / "research" / "pg_pk_19_triage_signal_integrity_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg_pk_19_triage_signal_integrity_v1.md"
TRIAGE_PATH = ROOT / "app" / "experiment_engineering_triage.py"
TEST_PATH = ROOT / "tests" / "test_experiment_engineering_triage.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(
    *,
    case_id: str,
    expected_classification: str,
    expected_model_authorized: bool,
    expected_scale_authorized: bool,
    experiment_signals: list[str] | None = None,
    engineering_signals: list[str] | None = None,
) -> dict[str, Any]:
    experiment_signals = experiment_signals or []
    engineering_signals = engineering_signals or []
    result = triage_failure(
        experiment_signals=experiment_signals,
        engineering_signals=engineering_signals,
        experiment_gate_passed=not experiment_signals,
        engineering_gate_passed=not engineering_signals,
    )
    passed = bool(
        result["classification"] == expected_classification
        and result["model_change_authorized"] is expected_model_authorized
        and result["infrastructure_scale_authorized"] is expected_scale_authorized
        and result["experiment_path"]["signals"] == sorted(set(experiment_signals))
        and result["engineering_path"]["signals"] == sorted(set(engineering_signals))
    )
    return {
        "case_id": case_id,
        "inputs": {
            "experiment_signals": experiment_signals,
            "engineering_signals": engineering_signals,
        },
        "expected": {
            "classification": expected_classification,
            "model_change_authorized": expected_model_authorized,
            "infrastructure_scale_authorized": expected_scale_authorized,
        },
        "observed": result,
        "passed": passed,
    }


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    cases = [
        _case(
            case_id="domain_experiment_signal",
            experiment_signals=["sql_cross_source_promotion_regression"],
            expected_classification="experiment_problem",
            expected_model_authorized=False,
            expected_scale_authorized=False,
        ),
        _case(
            case_id="domain_engineering_signal",
            engineering_signals=["missing_joint_regression_artifact"],
            expected_classification="engineering_capability_problem",
            expected_model_authorized=False,
            expected_scale_authorized=False,
        ),
        _case(
            case_id="registered_experiment_signal",
            experiment_signals=["family_holdout_regression"],
            expected_classification="experiment_problem",
            expected_model_authorized=True,
            expected_scale_authorized=False,
        ),
        _case(
            case_id="registered_engineering_signal",
            engineering_signals=["data_hash_or_lineage_mismatch"],
            expected_classification="engineering_capability_problem",
            expected_model_authorized=False,
            expected_scale_authorized=True,
        ),
        _case(
            case_id="clean_run",
            expected_classification="inconclusive",
            expected_model_authorized=False,
            expected_scale_authorized=False,
        ),
    ]
    result = {
        "schema_version": "sift-pg-pk-19-triage-signal-integrity-report-v1",
        "protocol_id": protocol["protocol_id"],
        "status": "pass" if all(case["passed"] for case in cases) else "fail",
        "case_count": len(cases),
        "passed_case_count": sum(int(case["passed"]) for case in cases),
        "pre_fix_negative_controls": protocol["pre_fix_negative_controls"],
        "post_fix_cases": cases,
        "protected_invariants": protocol["protected_invariants"],
        "evidence_sha256": {
            "protocol": _sha256(PROTOCOL_PATH),
            "triage_implementation": _sha256(TRIAGE_PATH),
            "triage_tests": _sha256(TEST_PATH),
        },
        "local_only": True,
        "model_or_checkpoint_modified": False,
        "payload_or_oracle_modified": False,
    }
    REPORT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(
        "# PG-PK-19 triage 信号完整性实验\n\n"
        f"状态：`{result['status']}`；通过：{result['passed_case_count']}/{result['case_count']}。\n\n"
        "修复前，PG-PK-17 的领域化实验失败与缺失产物信号都会被静默过滤成 `inconclusive`。"
        "修复后，信号保留在调用者指定的路径；未注册信号能正确分类，但在映射进正式 taxonomy 前不会授权改模型或扩容。\n\n"
        "该实验只修改 triage 信号传输与测试，没有修改模型、checkpoint、payload 或族特异 oracle。\n",
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
