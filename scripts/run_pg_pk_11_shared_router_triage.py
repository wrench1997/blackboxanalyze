"""Audit PG-PK-11 and classify experiment vs engineering failure paths."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.experiment_engineering_triage import triage_failure  # noqa: E402
from app.logic_access_fixture import logic_access_fixture_source_sha256  # noqa: E402
from app.shared_family_representation import SHARED_FAMILY_CLASSES, SHARED_FEATURE_DIM  # noqa: E402


REPORT_PATH = ROOT / "artifacts" / "shared-family-router-pg-pk-11" / "report.json"
CHECKPOINT_PATH = ROOT / "artifacts" / "shared-family-router-pg-pk-11" / "shared_family_router.pt"
TRIAGE_PATH = ROOT / "research" / "pg_pk_11_shared_router_triage_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg_pk_11_shared_router_triage_v1.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    experiment_signals: list[str] = []
    engineering_signals: list[str] = []
    evidence: dict[str, Any] = {
        "report_exists": REPORT_PATH.exists(),
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
        "report_schema_valid": False,
        "checkpoint_schema_valid": False,
        "lineage_matches": False,
    }
    report: dict[str, Any] = {}
    if not REPORT_PATH.exists():
        engineering_signals.append("oom_timeout_or_checkpoint_failure")
    else:
        try:
            report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
            evidence["report_schema_valid"] = report.get("schema_version") == "sift-shared-family-router-report-v1"
        except (OSError, ValueError, TypeError):
            engineering_signals.append("data_hash_or_lineage_mismatch")
    checkpoint: dict[str, Any] = {}
    if not CHECKPOINT_PATH.exists():
        engineering_signals.append("oom_timeout_or_checkpoint_failure")
    else:
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
            evidence["checkpoint_schema_valid"] = checkpoint.get("schema_version") == "sift-shared-family-router-checkpoint-v1"
            if int(checkpoint.get("feature_dim", -1)) != SHARED_FEATURE_DIM or list(checkpoint.get("classes", [])) != list(SHARED_FAMILY_CLASSES):
                engineering_signals.append("data_hash_or_lineage_mismatch")
        except (OSError, ValueError, RuntimeError, TypeError):
            engineering_signals.append("oom_timeout_or_checkpoint_failure")
    expected_hash = logic_access_fixture_source_sha256()
    source_hashes = ((report.get("checkpoint") and report.get("feature_contract")) or {})
    # The concrete source hashes are stored on the checkpoint, not in the
    # visible model report; check that checkpoint lineage is present and valid.
    training_sources = checkpoint.get("training_sources") or {}
    evidence["lineage_matches"] = bool(
        training_sources.get("logic_access_fixture_source_sha256") == expected_hash
        and isinstance(training_sources.get("surface_fixture_source_sha256"), str)
        and isinstance(training_sources.get("sql_fixture_source_sha256"), str)
    )
    if not evidence["lineage_matches"]:
        engineering_signals.append("data_hash_or_lineage_mismatch")

    holdout = report.get("encoding_holdout_metrics") or {}
    fresh = report.get("fresh_logic_target_metrics") or {}
    acceptance = report.get("acceptance") or {}
    # Family routing is a diagnostic head.  A regression here is scientific;
    # a missing/corrupt artifact is engineering.  Do not combine the paths.
    if not bool(acceptance.get("passed", False)):
        experiment_signals.append("family_holdout_regression")
    if float(holdout.get("control_false_accept_rate", 0.0) or 0.0) > 0.05:
        experiment_signals.append("family_holdout_regression")
    if float(fresh.get("control_false_accept_rate", 0.0) or 0.0) > 0.05:
        experiment_signals.append("family_holdout_regression")

    experiment_gate_passed = not bool({*experiment_signals})
    engineering_gate_passed = not bool({*engineering_signals})
    triage = triage_failure(
        experiment_signals=experiment_signals,
        engineering_signals=engineering_signals,
        experiment_gate_passed=experiment_gate_passed,
        engineering_gate_passed=engineering_gate_passed,
        evidence=evidence,
    )
    result = {
        "schema_version": "sift-pg-pk-11-shared-router-triage-report-v1",
        "protocol_id": "pg-pk-11-shared-router-triage-v1",
        "status": "pass" if experiment_gate_passed and engineering_gate_passed else "needs_repair",
        "report": str(REPORT_PATH.relative_to(ROOT)),
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "triage": triage,
        "independent_gates": {
            "experiment_gate_passed": experiment_gate_passed,
            "engineering_gate_passed": engineering_gate_passed,
            "model_change_authorized": triage["model_change_authorized"],
            "infrastructure_scale_authorized": triage["infrastructure_scale_authorized"],
        },
    }
    TRIAGE_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(
        "# PG-PK-11 实验问题 / 工程能力问题双路径诊断\n\n"
        f"状态：`{result['status']}`；分类：`{triage['classification']}`。\n\n"
        f"实验路径：`{', '.join(triage['experiment_path']['signals']) or '无失败信号'}`；工程路径：`{', '.join(triage['engineering_path']['signals']) or '无失败信号'}`。\n\n"
        "两条路径独立验收：实验失败不得用扩容掩盖，工程失败不得通过修改科学假设掩盖。\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "protocol_id": result["protocol_id"],
        "status": result["status"],
        "classification": triage["classification"],
        "experiment_gate_passed": experiment_gate_passed,
        "engineering_gate_passed": engineering_gate_passed,
        "report": str(TRIAGE_PATH.relative_to(ROOT)),
        "markdown": str(MARKDOWN_PATH.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
