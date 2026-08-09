"""PG-241 capacity training over grounded Pikachu payload-acceptance traces.

This reuses the frozen-XXL failure-conditioned trainer but adds PG-241's
real GET/POST result-oracle records.  The 24102 seed is held out; an older
23702 positive/abstain seed is held out as a second source check.  The large
body remains frozen, and the resulting adapter is a training candidate only.
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
REPORT = RESEARCH / "pg241_payload_capacity_training_report_v1.json"
DATASET = RESEARCH / "pg241_payload_capacity_training_dataset_v1.json"
TRACE = RESEARCH / "pg241_payload_capacity_training_trace_v1.json"
PROTOCOL = RESEARCH / "pg241_payload_capacity_training_protocol_v1.json"
MARKDOWN = RESEARCH / "pg241_payload_capacity_training_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg241-payload-capacity-training-v1"


def _load_records() -> tuple[list[dict[str, Any]], dict[str, int]]:
    # The original loader supplies PG-236 plus PG-237's non-trivial typed
    # replay.  It is called before changing FRESH_SOURCE so its source labels
    # remain truthful.
    original_source = PG237.FRESH_SOURCE
    PG237.FRESH_SOURCE = "pg237_pikachu_result_fixture_replay"
    try:
        base_rows, counts = ORIGINAL_LOAD_RECORDS()
    finally:
        PG237.FRESH_SOURCE = original_source
    payload = json.loads(PG241_DATASET.read_text(encoding="utf-8-sig"))
    fresh = [dict(row) for row in payload.get("records", [])]
    rows = list(base_rows) + fresh
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
    return unique, {**counts, "fresh_pg241_records": len(fresh), "unique_records": len(unique), "duplicate_records": duplicate_count}


def main() -> int:
    PG237._load_records = _load_records
    PG237.FRESH_SOURCE = "pg241_pikachu_source_native"
    PG237.FRESH_HOLDOUT_SEEDS = (24102,)
    PG237.EXTRA_HOLDOUT_SOURCE = "pg237_pikachu_result_fixture_replay"
    PG237.EXTRA_HOLDOUT_SEEDS = (23702,)
    PG237.ARTIFACT_DIR = ARTIFACT_DIR
    PG237.EXPERIMENT_ID = "pg241_payload"
    PG237.REPORT = REPORT
    PG237.DATASET = DATASET
    PG237.TRACE = TRACE
    PG237.PROTOCOL = PROTOCOL
    PG237.MARKDOWN = MARKDOWN
    PG237.main()
    report = json.loads(REPORT.read_text(encoding="utf-8-sig"))
    dataset = json.loads(DATASET.read_text(encoding="utf-8-sig"))
    report.update(
        {
            "protocol_id": "pg-pk-241-payload-capacity-training-v1",
            "schema_version": "pg241-payload-capacity-training-v1",
            "status": "completed_grounded_payload_trace_capacity_training",
            "source_dataset": str(PG241_DATASET.relative_to(ROOT)),
            "holdout_contract": {"pg241_seed_24102_never_in_training": True, "pg237_seed_23702_never_in_training": True, "holdout_contains_positive_and_abstain": True},
            "honesty": {"frozen_xxl_body_not_updated": True, "runtime_binder_trace_is_abstracted": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "general_web_capability_not_established": True},
        }
    )
    report["report_sha256"] = PG237.digest(report)
    dataset["schema_version"] = "pg241-payload-capacity-training-dataset-v1"
    dataset["source_datasets"] = list(dataset.get("source_datasets") or []) + [str(PG241_DATASET.relative_to(ROOT))]
    dataset["contract"] = {**dict(dataset.get("contract") or {}), "pg241_gold_process_rows_included": True, "pg241_seed_24102_never_in_training": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "vulnerability_claim_allowed": False}
    dataset["dataset_sha256"] = PG237.digest(dataset)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8-sig"))
    protocol.update({"protocol_id": "pg-pk-241-payload-capacity-training-v1", "schema_version": "pg241-payload-capacity-training-protocol-v1", "pg241_source_holdout": ["seed=24102"], "pg237_source_holdout": ["seed=23702"], "raw_payload_and_response_excluded": True, "training_artifact_promotion_allowed": False, "memory_promotion_allowed": False})
    protocol["protocol_sha256"] = PG237.digest(protocol)
    PG237._write(REPORT, report)
    PG237._write(DATASET, dataset)
    PG237._write(PROTOCOL, protocol)
    trace = json.loads(TRACE.read_text(encoding="utf-8-sig"))
    trace.update({"schema_version": "pg241-payload-capacity-training-trace-v1", "pg241_source_holdout": ["seed=24102"], "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    PG237._write(TRACE, trace)
    print(json.dumps({"status": report["status"], "device": report.get("device"), "counts": report.get("counts"), "selected": report.get("selected"), "safety_abstain_gate_pass": report.get("safety_abstain_gate_pass"), "capability_gate_pass": report.get("capability_gate_pass"), "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
