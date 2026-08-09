"""PG-245: train a larger frozen-XXL adapter on multi-step repair traces.

PG-244 contributes explicit counterfactual -> repair -> fresh-replay records.
The 24402 seed is held out as a route/seed process check; PG-242 XSS seed
24202 is an additional family/source holdout.  This keeps the experiment
focused on self-error and repair generalization rather than memorization.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG237 = _load("run_pg237_capacity_training.py")
ORIGINAL_LOAD_RECORDS = PG237._load_records

RESEARCH = ROOT / "research"
PG241_DATASET = RESEARCH / "pg241_pikachu_payload_acceptance_dataset_v1.json"
PG242_DATASET = RESEARCH / "pg242_pikachu_xss_dom_acceptance_dataset_v1.json"
PG244_DATASET = RESEARCH / "pg244_failure_repair_trajectory_dataset_v1.json"
REPORT = RESEARCH / "pg244_failure_repair_capacity_training_report_v1.json"
DATASET = RESEARCH / "pg244_failure_repair_capacity_training_dataset_v1.json"
TRACE = RESEARCH / "pg244_failure_repair_capacity_training_trace_v1.json"
PROTOCOL = RESEARCH / "pg244_failure_repair_capacity_training_protocol_v1.json"
MARKDOWN = RESEARCH / "pg244_failure_repair_capacity_training_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg244-failure-repair-capacity-v1"


def _load_records() -> tuple[list[dict[str, Any]], dict[str, int]]:
    original_source = PG237.FRESH_SOURCE
    PG237.FRESH_SOURCE = "pg237_pikachu_result_fixture_replay"
    try:
        base_rows, counts = ORIGINAL_LOAD_RECORDS()
    finally:
        PG237.FRESH_SOURCE = original_source
    added: list[dict[str, Any]] = []
    for path in (PG241_DATASET, PG242_DATASET, PG244_DATASET):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        added.extend(dict(row) for row in payload.get("records", []))
    rows = list(base_rows) + added
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, str]] = set()
    duplicate_count = 0
    for row in rows:
        key = (str(row.get("trajectory_hash", row.get("token_hash", ""))), int(row.get("seed", 0) or 0), str(row.get("source", "")), str(row.get("route_source_sha256", "")))
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        unique.append(row)
    return unique, {**counts, "fresh_pg241_records": sum(1 for row in added if row.get("source") == "pg241_pikachu_source_native"), "fresh_pg242_records": sum(1 for row in added if row.get("source") == "pg242_pikachu_source_native"), "fresh_pg244_records": sum(1 for row in added if row.get("source") == "pg244_pikachu_sql_repair"), "unique_records": len(unique), "duplicate_records": duplicate_count}


def main() -> int:
    PG237._load_records = _load_records
    PG237.FRESH_SOURCE = "pg244_pikachu_sql_repair"
    PG237.FRESH_HOLDOUT_SEEDS = (24402,)
    PG237.EXTRA_HOLDOUT_SOURCE = "pg242_pikachu_source_native"
    PG237.EXTRA_HOLDOUT_SEEDS = (24202,)
    PG237.ARTIFACT_DIR = ARTIFACT_DIR
    PG237.EXPERIMENT_ID = "pg244_failure_repair"
    PG237.REPORT = REPORT
    PG237.DATASET = DATASET
    PG237.TRACE = TRACE
    PG237.PROTOCOL = PROTOCOL
    PG237.MARKDOWN = MARKDOWN
    PG237.main()
    report = json.loads(REPORT.read_text(encoding="utf-8-sig"))
    dataset = json.loads(DATASET.read_text(encoding="utf-8-sig"))
    report.update({"protocol_id": "pg-pk-245-failure-repair-capacity-training-v1", "schema_version": "pg244-failure-repair-capacity-training-v1", "status": "completed_multistep_failure_repair_capacity_training", "source_datasets": [str(PG241_DATASET.relative_to(ROOT)), str(PG242_DATASET.relative_to(ROOT)), str(PG244_DATASET.relative_to(ROOT))], "holdout_contract": {"pg244_seed_24402_never_in_training": True, "pg242_xss_seed_24202_never_in_training": True, "holdout_contains_send_and_abstain": True, "long_context_repair_rows_present": True}, "honesty": {"frozen_xxl_body_not_updated": True, "adapter_only": True, "counterfactual_is_not_model_error": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "general_web_capability_not_established": True}})
    report["report_sha256"] = PG237.digest(report)
    dataset["schema_version"] = "pg244-failure-repair-capacity-training-dataset-v1"
    dataset["source_datasets"] = list(dataset.get("source_datasets") or []) + [str(PG241_DATASET.relative_to(ROOT)), str(PG242_DATASET.relative_to(ROOT)), str(PG244_DATASET.relative_to(ROOT))]
    dataset["contract"] = {**dict(dataset.get("contract") or {}), "pg244_seed_24402_never_in_training": True, "pg242_xss_seed_24202_never_in_training": True, "long_context_repair_rows_present": True, "next_token_loss_not_promotion_gate": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "vulnerability_claim_allowed": False}
    dataset["dataset_sha256"] = PG237.digest(dataset)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8-sig"))
    protocol.update({"protocol_id": "pg-pk-245-failure-repair-capacity-training-v1", "schema_version": "pg244-failure-repair-capacity-training-protocol-v1", "process_holdout": ["pg244_pikachu_sql_repair:24402", "pg242_pikachu_source_native:24202"], "must_include_steps": ["counterfactual", "repair", "replay"], "holdout_must_contain_send_and_abstain": True, "false_send_is_hard_failure": True, "next_token_loss_not_promotion_gate": True, "frozen_body_required": True, "promotion_blocked": True, "raw_payload_and_response_excluded": True})
    protocol["protocol_sha256"] = PG237.digest(protocol)
    PG237._write(REPORT, report)
    PG237._write(DATASET, dataset)
    PG237._write(PROTOCOL, protocol)
    trace = json.loads(TRACE.read_text(encoding="utf-8-sig"))
    trace.update({"schema_version": "pg244-failure-repair-capacity-training-trace-v1", "process_holdout": ["pg244:24402", "pg242:24202"], "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    PG237._write(TRACE, trace)
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "counts": report["counts"], "selected": report["selected"], "safety_abstain_gate": report.get("safety_abstain_gate_pass"), "capability_gate": report.get("capability_gate_pass"), "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
