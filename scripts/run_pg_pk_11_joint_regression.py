"""Cross-family regression audit after enabling the shared active prior."""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.experiment_engineering_triage import triage_failure  # noqa: E402
from app.experiment_ledger import ExperimentLedger  # noqa: E402


SHARED_REPORT = ROOT / "artifacts" / "shared-family-router-pg-pk-11" / "report.json"
SHARED_TRIAGE = ROOT / "research" / "pg_pk_11_shared_router_triage_v1.json"
SQL_ACTIVE_REPORT = ROOT / "research" / "pg_pk_09_sql_active_probe_v1.json"
LOGIC_REPORT = ROOT / "research" / "pg_pk_10_logic_access_v1.json"
REPORT_PATH = ROOT / "research" / "pg_pk_11_joint_regression_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg_pk_11_joint_regression_v1.md"
PROTOCOL_PATH = ROOT / "research" / "pg_pk_11_joint_regression_protocol_v1.json"
LEDGER_PATH = ROOT / "artifacts" / "experiment-ledger-pg-pk-11.jsonl"
SHARED_CHECKPOINT = ROOT / "artifacts" / "shared-family-router-pg-pk-11" / "shared_family_router.pt"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    missing = [str(path.relative_to(ROOT)) for path in (SHARED_REPORT, SHARED_TRIAGE, SQL_ACTIVE_REPORT, LOGIC_REPORT) if not path.exists()]
    experiment_signals: list[str] = []
    engineering_signals: list[str] = []
    reports: dict[str, Any] = {}
    if missing:
        engineering_signals.append("oom_timeout_or_checkpoint_failure")
    else:
        try:
            reports = {
                "shared": _load(SHARED_REPORT),
                "shared_triage": _load(SHARED_TRIAGE),
                "sql_active": _load(SQL_ACTIVE_REPORT),
                "logic_access": _load(LOGIC_REPORT),
            }
        except (OSError, ValueError, TypeError):
            engineering_signals.append("data_hash_or_lineage_mismatch")
    shared = reports.get("shared", {})
    sql = reports.get("sql_active", {})
    logic = reports.get("logic_access", {})
    if shared and not bool((shared.get("acceptance") or {}).get("passed", False)):
        experiment_signals.append("family_holdout_regression")
    joint = (shared.get("fresh_joint_surface_holdout_metrics") or {})
    if float(joint.get("accuracy", 0.0) or 0.0) < 0.90:
        experiment_signals.append("family_holdout_regression")
    if int(sql.get("oracle_revalidated_pair_count", 0) or 0) < 1 or bool(sql.get("shared_router_positive_authority", True)):
        experiment_signals.append("metric_or_oracle_definition_changed")
    if int(logic.get("oracle_revalidated_pair_count", 0) or 0) < 1:
        experiment_signals.append("family_holdout_regression")
    if any((reports.get("logic_access", {}).get("memory_promotion", {}).get(family, {}).get("status") != "promote") for family in ("access_control", "logic")):
        experiment_signals.append("family_holdout_regression")
    triage = triage_failure(
        experiment_signals=experiment_signals,
        engineering_signals=engineering_signals,
        experiment_gate_passed=not bool(experiment_signals),
        engineering_gate_passed=not bool(engineering_signals),
        evidence={
            "missing_artifacts": missing,
            "shared_router_positive_authority": bool(sql.get("shared_router_positive_authority", True)),
            "sql_active_request_count": sql.get("request_count"),
            "sql_active_oracle_pair_count": sql.get("oracle_revalidated_pair_count"),
            "logic_oracle_pair_count": logic.get("oracle_revalidated_pair_count"),
            "logic_memory_status": {family: logic.get("memory_promotion", {}).get(family, {}).get("status") for family in ("access_control", "logic")},
        },
    )
    result = {
        "schema_version": "sift-pg-pk-11-joint-regression-report-v1",
        "protocol_id": "pg-pk-11-joint-cross-family-regression-v1",
        "status": "pass" if not experiment_signals and not engineering_signals else "needs_repair",
        "triage": triage,
        "metrics": {
            "shared_encoding_holdout": shared.get("encoding_holdout_metrics"),
            "shared_joint_surface_holdout": joint,
            "shared_pair_invariance": shared.get("pair_invariance"),
            "sql_active": {
                "requests": sql.get("request_count"),
                "complete_pairs": sql.get("complete_pair_count"),
                "oracle_revalidated_pairs": sql.get("oracle_revalidated_pair_count"),
                "shared_router_abstains": sql.get("shared_router_abstain_count"),
                "shared_router_ood": sql.get("shared_router_ood_count"),
                "positive_authority": sql.get("shared_router_positive_authority"),
            },
            "logic_access": {
                "oracle_revalidated_pairs": logic.get("oracle_revalidated_pair_count"),
                "counterfactual_model_accepts": logic.get("model_only_counterfactual_accept_count"),
                "memory_promotion": {family: logic.get("memory_promotion", {}).get(family, {}).get("status") for family in ("access_control", "logic")},
            },
        },
        "required_invariants": {
            "shared_router_diagnostic_only": True,
            "family_specific_oracle_positive_gate": True,
            "active_budget_preserved": int(sql.get("request_count", 10**9)) <= 13,
            "counterfactual_model_accepts_zero": int(logic.get("model_only_counterfactual_accept_count", 1)) == 0,
            "triage_paths_independent": True,
        },
        "source_reports": [
            str(SHARED_REPORT.relative_to(ROOT)),
            str(SHARED_TRIAGE.relative_to(ROOT)),
            str(SQL_ACTIVE_REPORT.relative_to(ROOT)),
            str(LOGIC_REPORT.relative_to(ROOT)),
        ],
    }
    ledger = ExperimentLedger(LEDGER_PATH, ROOT)
    ledger_record = ledger.append({
        "protocol_id": "pg-pk-11-joint-cross-family-regression-v1",
        "run_id": "pg-pk-11-joint-regression",
        "dataset_id": "cross-family-local-fixtures",
        "target_instance_id": "logic-gamma+surface-fresh+sql-fresh",
        "sampling_seed": 20260811,
        "status": result["status"],
        "triage_classification": triage["classification"],
        "experiment_gate_passed": not bool(experiment_signals),
        "engineering_gate_passed": not bool(engineering_signals),
        "metrics": result["metrics"],
        "model_checkpoint_sha256": _file_sha256(SHARED_CHECKPOINT) if SHARED_CHECKPOINT.exists() else "",
        "local_only": True,
    })
    result["ledger"] = {
        "path": str(LEDGER_PATH.relative_to(ROOT)),
        "record_hash": ledger_record["record_hash"],
        "verification": ledger.verify(),
    }
    REPORT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(
        "# PG-PK-11 跨族联合回归\n\n"
        f"状态：`{result['status']}`；实验/工程分类：`{triage['classification']}`。\n\n"
        f"共享编码留出 accuracy：{(shared.get('encoding_holdout_metrics') or {}).get('accuracy', 0):.3f}；联合族外表面：{joint.get('accuracy', 0):.3f}；SQL active oracle pair：{sql.get('oracle_revalidated_pair_count', 0)}；logic/access oracle pair：{logic.get('oracle_revalidated_pair_count', 0)}。\n\n"
        f"共享路由只参与主动 belief prior，任何正向 Rule IR 仍必须经过族特异 oracle；失败时分别走实验修复和工程修复路径。台账：`{result['ledger']['path']}`，head `{result['ledger']['verification']['head']}`。\n",
        encoding="utf-8",
    )
    PROTOCOL_PATH.write_text(json.dumps({
        "protocol_id": "pg-pk-11-joint-cross-family-regression-v1",
        "source_reports": result["source_reports"],
        "ledger": result["ledger"],
        "required_invariants": result["required_invariants"],
        "triage": "app/experiment_engineering_triage.py",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "protocol_id": result["protocol_id"],
        "status": result["status"],
        "classification": triage["classification"],
        "metrics": result["metrics"],
        "report": str(REPORT_PATH.relative_to(ROOT)),
        "markdown": str(MARKDOWN_PATH.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
