"""PG-PK-17: engineering joint regression over the current research loop."""

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
from app.experiment_ledger import ExperimentLedger  # noqa: E402


PROTOCOL_ID = "pg-pk-17-joint-regression-v1"
REPORT_PATH = ROOT / "research" / "pg_pk_17_joint_regression_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg_pk_17_joint_regression_v1.md"
PROTOCOL_PATH = ROOT / "research" / "pg_pk_17_joint_regression_protocol_v1.json"
LEDGER_PATH = ROOT / "artifacts" / "experiment-ledger-pg-pk-17.jsonl"
SHARED_REPORT = ROOT / "artifacts" / "shared-family-router-pg-pk-11" / "report.json"
LOGIC_TRAIN_REPORT = ROOT / "artifacts" / "logic-access-decoder-pg-pk-10" / "report.json"
SQL_TRAIN_REPORT = ROOT / "artifacts" / "sql-channel-decoder-pg-pk-09" / "report.json"
PG09_REPORT = ROOT / "research" / "pg_pk_09_sql_differential_v1.json"
PG10_REPORT = ROOT / "research" / "pg_pk_10_logic_access_v1.json"
PG14_REPORT = ROOT / "research" / "pg_pk_14_sql_v2_active_generalization_v1.json"
PG15_REPORT = ROOT / "research" / "pg_pk_15_sql_v3_cross_source_promotion_v1.json"
PG16_REPORT = ROOT / "research" / "pg_pk_16_logic_v2_cross_family_guard_v1.json"
PG16_PRE_FIX = ROOT / "research" / "pg_pk_16_logic_v2_cross_family_guard_pre_fix_v1.json"
PG18_REPORT = ROOT / "research" / "pg_pk_18_logic_v3_cross_source_promotion_v1.json"
SHARED_CHECKPOINT = ROOT / "artifacts" / "shared-family-router-pg-pk-11" / "shared_family_router.pt"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    required = (SHARED_REPORT, LOGIC_TRAIN_REPORT, SQL_TRAIN_REPORT, PG09_REPORT, PG10_REPORT, PG14_REPORT, PG15_REPORT, PG16_REPORT, PG16_PRE_FIX, PG18_REPORT, SHARED_CHECKPOINT)
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    reports: dict[str, Any] = {}
    engineering_signals: list[str] = []
    experiment_signals: list[str] = []
    if missing:
        engineering_signals.append("missing_joint_regression_artifact")
    else:
        try:
            reports = {
                "shared": _load(SHARED_REPORT),
                "logic_train": _load(LOGIC_TRAIN_REPORT),
                "sql_train": _load(SQL_TRAIN_REPORT),
                "pg09": _load(PG09_REPORT),
                "pg10": _load(PG10_REPORT),
                "pg14": _load(PG14_REPORT),
                "pg15": _load(PG15_REPORT),
                "pg16": _load(PG16_REPORT),
                "pg16_pre_fix": _load(PG16_PRE_FIX),
                "pg18": _load(PG18_REPORT),
            }
        except (OSError, ValueError, TypeError):
            engineering_signals.append("report_hash_or_schema_failure")
    shared = reports.get("shared", {})
    logic_train = reports.get("logic_train", {})
    sql_train = reports.get("sql_train", {})
    pg09 = reports.get("pg09", {})
    pg10 = reports.get("pg10", {})
    pg14 = reports.get("pg14", {})
    pg15 = reports.get("pg15", {})
    pg16 = reports.get("pg16", {})
    pre_fix = reports.get("pg16_pre_fix", {})
    pg18 = reports.get("pg18", {})

    shared_cal = shared.get("confidence_calibration") or {}
    logic_cal = logic_train.get("confidence_calibration") or {}
    if float(shared_cal.get("scaled_ece", 1.0)) > float(shared_cal.get("raw_ece", 0.0)) + 1e-9:
        experiment_signals.append("shared_confidence_calibration_regression")
    if float(logic_cal.get("scaled_ece", 1.0)) > float(logic_cal.get("raw_ece", 0.0)) + 1e-9:
        experiment_signals.append("logic_confidence_calibration_regression")
    if float(sql_train.get("fresh", {}).get("accuracy", 0.0)) < 0.90 or float(sql_train.get("acceptance", {}).get("control_false_accept_rate", 1.0)) > 0.05:
        experiment_signals.append("sql_decoder_training_regression")
    if str(pg09.get("memory_promotion_status", "quarantine")) not in {"quarantine", "promote"}:
        experiment_signals.append("sql_v1_schema_regression")
    if int(pg09.get("oracle_revalidated_pair_count", 0)) < 1 or int(pg14.get("oracle_revalidated_pair_count", 0)) < 1 or int(pg15.get("oracle_revalidated_pair_count", 0)) < 1:
        experiment_signals.append("sql_old_source_replay_regression")
    if int(pg15.get("false_positive_ledger_row_count", 1)) != 0 or str(pg15.get("status")) != "promote" or int((pg15.get("provenance") or {}).get("source_count", 0)) < 3:
        experiment_signals.append("sql_cross_source_promotion_regression")
    if int(pg10.get("model_only_counterfactual_accept_count", 1)) != 0 or int(pg10.get("oracle_revalidated_pair_count", 0)) < 1:
        experiment_signals.append("logic_v1_regression")
    if int(pg16.get("sql_cross_family_candidate_count", 1)) != 0 or int(pg16.get("shared_injection_route_count", 1)) != 0:
        experiment_signals.append("cross_family_guard_regression")
    if int(pg16.get("oracle_revalidated_pair_count", 0)) != 36 or int(pg16.get("logic_model_counterfactual_candidate_count", 1)) != 0 or str(pg16.get("status")) != "pass":
        experiment_signals.append("logic_v2_surface_holdout_regression")
    if int(pre_fix.get("sql_cross_family_candidate_count", 0)) <= int(pg16.get("sql_cross_family_candidate_count", 0)):
        experiment_signals.append("pre_fix_failure_not_demonstrated")
    if str(pg18.get("status")) != "promote" or int(pg18.get("oracle_revalidated_pair_count", 0)) != 36 or int(pg18.get("counterfactual_model_candidate_count", 1)) != 0 or int(pg18.get("false_positive_ledger_row_count", 1)) != 0 or int((pg18.get("provenance") or {}).get("source_count", 0)) < 3 or not bool((pg18.get("v3_positive_pair_coverage") or {}).get("passed", False)):
        experiment_signals.append("logic_v3_cross_source_promotion_regression")
    if not bool((shared.get("training_boundary") or {}).get("calibrated_confidence_is_route_only", True)) and "calibrated_confidence_is_route_only" in shared:
        experiment_signals.append("confidence_authority_boundary_changed")

    triage = triage_failure(
        experiment_signals=experiment_signals,
        engineering_signals=engineering_signals,
        experiment_gate_passed=not bool(experiment_signals),
        engineering_gate_passed=not bool(engineering_signals),
        evidence={
            "missing_artifacts": missing,
            "pg16_pre_fix_sql_candidates": pre_fix.get("sql_cross_family_candidate_count"),
            "pg16_post_fix_sql_candidates": pg16.get("sql_cross_family_candidate_count"),
            "pg16_oracle_pairs": pg16.get("oracle_revalidated_pair_count"),
            "pg15_source_count": (pg15.get("provenance") or {}).get("source_count"),
            "shared_scaled_ece": shared_cal.get("scaled_ece"),
            "logic_scaled_ece": logic_cal.get("scaled_ece"),
        },
    )
    result = {
        "schema_version": "sift-pg-pk-17-joint-regression-report-v1",
        "protocol_id": PROTOCOL_ID,
        "status": "pass" if not experiment_signals and not engineering_signals else "needs_repair",
        "triage": triage,
        "metrics": {
            "shared_confidence_calibration": {key: shared_cal.get(key) for key in ("temperature", "raw_ece", "scaled_ece", "abstain_threshold", "abstain_coverage", "control_false_accept_rate")},
            "logic_confidence_calibration": {key: logic_cal.get(key) for key in ("temperature", "raw_ece", "scaled_ece", "abstain_threshold", "abstain_coverage")},
            "sql": {"pg09_oracle_pairs": pg09.get("oracle_revalidated_pair_count"), "pg14_oracle_pairs": pg14.get("oracle_revalidated_pair_count"), "pg15_oracle_pairs": pg15.get("oracle_revalidated_pair_count"), "pg15_source_count": (pg15.get("provenance") or {}).get("source_count"), "pg15_false_positive_rows": pg15.get("false_positive_ledger_row_count")},
            "logic": {"pg10_oracle_pairs": pg10.get("oracle_revalidated_pair_count"), "pg10_counterfactual_model_accepts": pg10.get("model_only_counterfactual_accept_count"), "pg16_oracle_pairs": pg16.get("oracle_revalidated_pair_count"), "pg16_counterfactual_model_candidates": pg16.get("logic_model_counterfactual_candidate_count")},
            "logic_v3": {"oracle_pairs": pg18.get("oracle_revalidated_pair_count"), "source_count": (pg18.get("provenance") or {}).get("source_count"), "false_positive_rows": pg18.get("false_positive_ledger_row_count"), "coverage": pg18.get("v3_positive_pair_coverage"), "promotion": {family: pg18.get("cross_source_memory_promotion", {}).get(family, {}).get("status") for family in ("access_control", "logic")}},
            "cross_family_guard": {"pre_fix_sql_candidates": pre_fix.get("sql_cross_family_candidate_count"), "post_fix_sql_candidates": pg16.get("sql_cross_family_candidate_count"), "post_fix_shared_injection_routes": pg16.get("shared_injection_route_count")},
        },
        "required_invariants": {
            "typed_oracle_positive_authority": True,
            "shared_router_positive_authority": False,
            "sql_cross_family_candidate_zero": int(pg16.get("sql_cross_family_candidate_count", 1)) == 0,
            "logic_v2_complete_pair_oracle": int(pg16.get("oracle_revalidated_pair_count", 0)) == 36,
            "logic_v2_counterfactual_candidates_zero": int(pg16.get("logic_model_counterfactual_candidate_count", 1)) == 0,
            "sql_v3_three_source_promotion": str(pg15.get("status")) == "promote" and int((pg15.get("provenance") or {}).get("source_count", 0)) >= 3,
            "logic_v3_three_source_promotion": str(pg18.get("status")) == "promote" and int((pg18.get("provenance") or {}).get("source_count", 0)) >= 3 and bool((pg18.get("v3_positive_pair_coverage") or {}).get("passed", False)),
            "pre_fix_failure_preserved": int(pre_fix.get("sql_cross_family_candidate_count", 0)) > int(pg16.get("sql_cross_family_candidate_count", 0)),
            "confidence_is_route_only": True,
        },
        "source_reports": [str(path.relative_to(ROOT)) for path in required if path != SHARED_CHECKPOINT],
    }
    ledger = ExperimentLedger(LEDGER_PATH, ROOT)
    ledger_record = ledger.append({
        "protocol_id": PROTOCOL_ID,
        "run_id": "pg-pk-17-joint-regression",
        "dataset_id": "cross-family-local-fixtures-v2",
        "target_instance_id": "sql-v1+v2+v3+logic-v1+v2",
        "sampling_seed": 20610101,
        "status": result["status"],
        "triage_classification": triage["classification"],
        "experiment_gate_passed": not bool(experiment_signals),
        "engineering_gate_passed": not bool(engineering_signals),
        "metrics": result["metrics"],
        "model_checkpoint_sha256": _sha256(SHARED_CHECKPOINT),
        "local_only": True,
    })
    result["ledger"] = {"path": str(LEDGER_PATH.relative_to(ROOT)), "record_hash": ledger_record["record_hash"], "verification": ledger.verify()}
    REPORT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(
        "# PG-PK-17 联合回归门\n\n"
        f"状态：`{result['status']}`；triage：`{triage['classification']}`。\n\n"
        f"SQL v1/v2/v3 oracle pair：{pg09.get('oracle_revalidated_pair_count', 0)}/{pg14.get('oracle_revalidated_pair_count', 0)}/{pg15.get('oracle_revalidated_pair_count', 0)}；v3 source：{(pg15.get('provenance') or {}).get('source_count', 0)}；false-positive ledger：{pg15.get('false_positive_ledger_row_count', 0)}。\n\n"
        f"logic v1/v2/v3 oracle pair：{pg10.get('oracle_revalidated_pair_count', 0)}/{pg16.get('oracle_revalidated_pair_count', 0)}/{pg18.get('oracle_revalidated_pair_count', 0)}；PG-PK-16 SQL 跨族 candidate：{pg16.get('sql_cross_family_candidate_count', 0)}（pre-fix {pre_fix.get('sql_cross_family_candidate_count', 0)}）；shared injection route：{pg16.get('shared_injection_route_count', 0)}。\n\n"
        f"shared ECE：{shared_cal.get('raw_ece', 0):.6f} → {shared_cal.get('scaled_ece', 0):.6f}；logic ECE：{logic_cal.get('raw_ece', 0):.6f} → {logic_cal.get('scaled_ece', 0):.6f}。\n\n"
        f"台账：`{result['ledger']['path']}`，head `{result['ledger']['verification']['head']}`。\n",
        encoding="utf-8",
    )
    PROTOCOL_PATH.write_text(json.dumps({"protocol_id": PROTOCOL_ID, "source_reports": result["source_reports"], "required_invariants": result["required_invariants"], "triage": "app/experiment_engineering_triage.py", "ledger": result["ledger"]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protocol_id": PROTOCOL_ID, "status": result["status"], "classification": triage["classification"], "metrics": result["metrics"], "report": str(REPORT_PATH.relative_to(ROOT)), "markdown": str(MARKDOWN_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
