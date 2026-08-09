# -*- coding: utf-8 -*-
"""PG-263: train the mask-aware adapter with audited PG-262 fresh traces.

PG-262 is admitted only through its integrity sidecar.  The model receives
abstract tokens and labels; typed oracle facts stay in the report/audit lane
and are never concatenated to model input.  PG-261's historical records are
kept as the base representation, while even PG-262 seeds form a fresh holdout.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location("pg263_pg261_runner", ROOT / "scripts" / "run_pg261_masked_active_belief_capacity_training.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-261 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG261 = _load()
PG260 = PG261.PG260
RESEARCH = ROOT / "research"
PG261_DATASET = RESEARCH / "pg261_masked_active_belief_capacity_training_dataset_v1.json"
PG262_DATASET = RESEARCH / "pg262_targeted_paired_trace_collection_dataset_v1.json"
PG262_AUDIT = RESEARCH / "pg262_targeted_paired_trace_collection_audit_v1.json"
REPORT = RESEARCH / "pg263_pg262_augmented_masked_capacity_training_report_v1.json"
DATASET = RESEARCH / "pg263_pg262_augmented_masked_capacity_training_dataset_v1.json"
TRACE = RESEARCH / "pg263_pg262_augmented_masked_capacity_training_trace_v1.json"
PROTOCOL = RESEARCH / "pg263_pg262_augmented_masked_capacity_training_protocol_v1.json"
MARKDOWN = RESEARCH / "pg263_pg262_augmented_masked_capacity_training_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg263-pg262-augmented-masked-capacity-v1"
RUN_MARKER = RESEARCH / "pg263_training_running.json"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_records() -> list[dict[str, Any]]:
    audit = _load_json(PG262_AUDIT)
    if not bool(audit.get("all_required_fields_complete")) or int(audit.get("audited_record_count", 0) or 0) != 20:
        raise RuntimeError("PG-262 audit is incomplete; refusing to train")
    rows: list[dict[str, Any]] = []
    for path, lane in ((PG261_DATASET, "pg261_masked_base"), (PG262_DATASET, "pg262_audited_fresh")):
        payload = _load_json(path)
        for row in list(payload.get("records") or []):
            if str(row.get("lane", "")) in {"quarantine", "reject"}:
                continue
            rows.append(PG260._normalise(row, lane))
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, str, str]] = set()
    for row in rows:
        key = (str(row.get("source", "")), int(row.get("seed", 0) or 0), PG260._route(row), str(row.get("route_source_sha256", "")), str(row.get("trajectory_hash", row.get("token_hash", ""))))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


BASE_IS_HOLDOUT = PG260._is_holdout


def _is_holdout(row: dict[str, Any]) -> bool:
    source = str(row.get("source", ""))
    if source.startswith("pg262_"):
        # Even fresh seeds are entirely disjoint from the odd-seed training
        # rows and preserve every new family in the holdout.
        return int(row.get("seed", 0) or 0) % 2 == 0
    return bool(BASE_IS_HOLDOUT(row))


def _write_marker() -> None:
    RUN_MARKER.write_text(json.dumps({"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(), "report": str(REPORT.relative_to(ROOT)), "protocol_id": "pg-pk-263-pg262-augmented-masked-capacity-v1"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _clear_marker() -> None:
    try:
        RUN_MARKER.unlink(missing_ok=True)
    except OSError:
        pass


def main() -> int:
    _write_marker()
    try:
        return _run()
    finally:
        _clear_marker()


def _run() -> int:
    PG260._load_records = _load_records
    PG260._is_holdout = _is_holdout
    PG260.FRESH_SOURCE_PREFIXES = ("pg259_", "pg260_", "pg262_")
    # The first PG-263 run was intentionally full-batch for a clean baseline.
    # Future reruns default to micro-batching so a single 12 GB GPU can finish
    # the same three capacity branches without retaining all activations.
    raw_micro_batch = os.environ.get("PG263_MICRO_BATCH_SIZE", "16").strip()
    PG260.MICRO_BATCH_SIZE = int(raw_micro_batch) if raw_micro_batch.isdigit() and int(raw_micro_batch) > 0 else 0
    PG261.REPORT = REPORT
    PG261.DATASET = DATASET
    PG261.TRACE = TRACE
    PG261.PROTOCOL = PROTOCOL
    PG261.MARKDOWN = MARKDOWN
    PG261.ARTIFACT_DIR = ARTIFACT_DIR
    result = PG261.main()

    report = _load_json(REPORT)
    report.update({"protocol_id": "pg-pk-263-pg262-augmented-masked-capacity-v1", "schema_version": "pg263-pg262-augmented-masked-capacity-training-report-v1", "status": "completed_pg263_pg262_augmented_masked_capacity_training"})
    report["resource_profile"] = dict(report.get("resource_profile") or {}, pg263_micro_batch_size=int(PG260.MICRO_BATCH_SIZE or 0), capacity_variants_sequential=True)
    report["architecture_change"] = {"id": "pg263-pg262-augmented-mask-aware-pooling-v1", "base": "pg261-mask-aware-pooling-v1", "fresh_source": "pg262 audited 20-record local Pikachu collection", "pg262_audit_id": "pg262-fresh-replay-integrity-v1", "masked_mean_pool": True, "padding_invariant_classification": True, "oracle_target_off_input": True, "legacy_pg261_artifact_unchanged": True}
    report["counts"]["pg262_rows"] = sum(int(str(row.get("source", "")).startswith("pg262_")) for row in _load_records())
    report["counts"]["pg262_holdout_rows"] = sum(int(str(row.get("source", "")).startswith("pg262_") and _is_holdout(row)) for row in _load_records())
    report["counts"]["pg262_train_rows"] = report["counts"]["pg262_rows"] - report["counts"]["pg262_holdout_rows"]
    report["independent_final_judge"]["authority"] = list(report["independent_final_judge"].get("authority") or []) + ["PG-262 fresh replay integrity audit"]
    report["independent_final_judge"]["pg262_audit_required"] = True
    report["honesty"]["pg262_audit_complete"] = True
    report["honesty"]["raw_payload_strings_stored"] = False
    report["honesty"]["raw_response_bodies_stored"] = False
    report["promotion"] = {"training_promotion_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False, "blocked_by": list(report["independent_final_judge"].get("reasons") or [])}
    report["report_sha256"] = ""
    report["report_sha256"] = _digest(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    dataset = _load_json(DATASET)
    dataset["schema_version"] = "pg263-pg262-augmented-masked-capacity-training-dataset-v1"
    dataset["source_datasets"] = [str(PG261_DATASET.relative_to(ROOT)), str(PG262_DATASET.relative_to(ROOT))]
    dataset["contract"].update({"pg261_mask_aware_base": True, "pg262_audit_complete": True, "pg262_audit_file": str(PG262_AUDIT.relative_to(ROOT)), "oracle_target_off_input": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "training_promotion_allowed": False})
    dataset["dataset_sha256"] = ""
    dataset["dataset_sha256"] = _digest(dataset)
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    trace = _load_json(TRACE)
    trace.update({"schema_version": "pg263-pg262-augmented-masked-capacity-trace-v1", "pg262_audit_file": str(PG262_AUDIT.relative_to(ROOT)), "oracle_target_off_input": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = _load_json(PROTOCOL)
    protocol.update({"protocol_id": "pg-pk-263-pg262-augmented-masked-capacity-v1", "schema_version": "pg263-pg262-augmented-masked-capacity-training-protocol-v1", "training_sources": [str(PG261_DATASET.relative_to(ROOT)), str(PG262_DATASET.relative_to(ROOT))], "pg262_audit_file": str(PG262_AUDIT.relative_to(ROOT)), "fresh_source_prefixes": ["pg259_", "pg260_", "pg262_"], "pg262_even_seed_holdout": True, "oracle_target_off_input": True, "promotion_blocked": True})
    protocol["protocol_sha256"] = ""
    protocol["protocol_sha256"] = _digest(protocol)
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN.write_text(MARKDOWN.read_text(encoding="utf-8") + "\nPG-263 adds only the audited PG-262 abstract records; raw wire/response bodies remain excluded and all promotion gates remain blocked until independent review.\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT)), "selected": report.get("selected"), "judge": report.get("independent_final_judge"), "pg262_rows": report["counts"]["pg262_rows"], "pg262_holdout_rows": report["counts"]["pg262_holdout_rows"]}, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
