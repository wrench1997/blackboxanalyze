# -*- coding: utf-8 -*-
"""PG-265: train a larger adapter after the audited PG-264 growth tranche.

The frozen sequence body is unchanged.  PG-263's audited base is combined
with PG-264's new abstract records, while the even PG-264 seeds form a fresh
holdout.  Oracle/reference fields are labels only and never enter the model.
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


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG263 = _load(ROOT / "scripts" / "run_pg263_pg262_augmented_masked_capacity_training.py", "pg265_pg263_base")
PG261 = PG263.PG261
PG260 = PG263.PG260
RESEARCH = ROOT / "research"
PG263_DATASET = RESEARCH / "pg263_pg262_augmented_masked_capacity_training_dataset_v1.json"
PG264_DATASET = RESEARCH / "pg264_pikachu_growth_collection_dataset_v1.json"
PG264_AUDIT = RESEARCH / "pg264_pikachu_growth_collection_audit_v1.json"
REPORT = RESEARCH / "pg265_growth_augmented_large_capacity_training_report_v1.json"
DATASET = RESEARCH / "pg265_growth_augmented_large_capacity_training_dataset_v1.json"
TRACE = RESEARCH / "pg265_growth_augmented_large_capacity_training_trace_v1.json"
PROTOCOL = RESEARCH / "pg265_growth_augmented_large_capacity_training_protocol_v1.json"
MARKDOWN = RESEARCH / "pg265_growth_augmented_large_capacity_training_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg265-growth-augmented-large-capacity-v1"
RUN_MARKER = RESEARCH / "pg265_training_running.json"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_records() -> list[dict[str, Any]]:
    audit = _read(PG264_AUDIT)
    if not bool(audit.get("all_required_fields_complete")) or int(audit.get("audited_record_count", 0) or 0) != 32:
        raise RuntimeError("PG-264 independent audit is incomplete; refusing PG-265 training")
    rows: list[dict[str, Any]] = []
    for path, lane in ((PG263_DATASET, "pg263_audited_base"), (PG264_DATASET, "pg264_audited_growth")):
        payload = _read(path)
        for raw in list(payload.get("records") or []):
            if not isinstance(raw, dict) or str(raw.get("lane", "")) in {"quarantine", "reject"}:
                continue
            rows.append(PG260._normalise(raw, lane))
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, str, str]] = set()
    for row in rows:
        key = (str(row.get("source", "")), int(row.get("seed", 0) or 0), PG260._route(row), str(row.get("route_source_sha256", "")), str(row.get("trajectory_hash", row.get("token_hash", ""))))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _capacity_variants() -> tuple[int, ...]:
    raw = os.environ.get("PG265_CAPACITY_VARIANTS", "4096,8192,12288")
    values: list[int] = []
    for item in raw.split(","):
        try:
            value = int(item.strip())
        except ValueError:
            continue
        if value > 0 and value not in values:
            values.append(value)
    return tuple(values) or (4096, 8192, 12288)


BASE_IS_HOLDOUT = PG263._is_holdout


def _is_holdout(row: dict[str, Any]) -> bool:
    source = str(row.get("source", ""))
    if source.startswith("pg264_"):
        return int(row.get("seed", 0) or 0) % 2 == 0
    return bool(BASE_IS_HOLDOUT(row))


def _write_marker() -> None:
    RUN_MARKER.write_text(json.dumps({"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(), "report": str(REPORT.relative_to(ROOT)), "protocol_id": "pg-pk-265-growth-augmented-large-capacity-v1"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _clear_marker() -> None:
    RUN_MARKER.unlink(missing_ok=True)


def main() -> int:
    _write_marker()
    try:
        PG260._load_records = _load_records
        PG260._is_holdout = _is_holdout
        PG260.FRESH_SOURCE_PREFIXES = ("pg259_", "pg260_", "pg262_", "pg264_", "pg267_")
        PG260.CAPACITY_VARIANTS = _capacity_variants()
        PG260.TRAIN_STEPS = int(os.environ.get("PG265_TRAIN_STEPS", "170"))
        PG260.MICRO_BATCH_SIZE = max(int(os.environ.get("PG265_MICRO_BATCH_SIZE", "16")), 1)
        PG261.REPORT = REPORT
        PG261.DATASET = DATASET
        PG261.TRACE = TRACE
        PG261.PROTOCOL = PROTOCOL
        PG261.MARKDOWN = MARKDOWN
        PG261.ARTIFACT_DIR = ARTIFACT_DIR
        PG260.REPORT = REPORT
        PG260.DATASET = DATASET
        PG260.TRACE = TRACE
        PG260.PROTOCOL = PROTOCOL
        PG260.MARKDOWN = MARKDOWN
        PG260.ARTIFACT_DIR = ARTIFACT_DIR
        code = PG261.main()
        report = _read(REPORT)
        rows = _load_records()
        pg264_rows = [row for row in rows if str(row.get("source", "")).startswith("pg264_")]
        report.update({"protocol_id": "pg-pk-265-growth-augmented-large-capacity-v1", "schema_version": "pg265-growth-augmented-large-capacity-training-report-v1", "status": "completed_pg265_growth_augmented_large_capacity_training"})
        report["capacity_variants"] = list(PG260.CAPACITY_VARIANTS)
        report["architecture_change"] = {"id": "pg265-growth-augmented-large-capacity-v1", "base": "pg263-pg262-augmented-mask-aware-pooling-v1", "fresh_source": "PG-264 audited 32-record local Pikachu growth tranche", "pg264_audit_id": "pg264-fresh-replay-integrity-v1", "oracle_target_off_input": True, "legacy_artifacts_unchanged": True}
        report["growth_counts"] = {"combined_records": len(rows), "pg263_records": len(rows) - len(pg264_rows), "pg264_records": len(pg264_rows), "pg264_even_seed_holdout": sum(int(_is_holdout(row)) for row in pg264_rows)}
        report["promotion"] = {"training_promotion_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False, "blocked_by": list((report.get("independent_final_judge") or {}).get("reasons") or [])}
        report["evaluation_audit"] = {"audit_id": "pg265-final-report-audit-v1", "pg264_audit_sha256": str(audit_sha256 := _read(PG264_AUDIT).get("audit_sha256", "")), "weights_changed": False}
        report["report_sha256"] = ""
        report["report_sha256"] = _digest(report)
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        dataset = _read(DATASET)
        dataset["schema_version"] = "pg265-growth-augmented-large-capacity-training-dataset-v1"
        dataset["source_datasets"] = [str(PG263_DATASET.relative_to(ROOT)), str(PG264_DATASET.relative_to(ROOT))]
        dataset["contract"] = dict(dataset.get("contract") or {}, pg264_audit_complete=True, pg264_audit_file=str(PG264_AUDIT.relative_to(ROOT)), oracle_target_off_input=True, raw_payload_strings_stored=False, raw_response_bodies_stored=False, training_promotion_allowed=False)
        dataset["dataset_sha256"] = ""
        dataset["dataset_sha256"] = _digest(dataset)
        DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        protocol = _read(PROTOCOL)
        protocol.update({"protocol_id": report["protocol_id"], "schema_version": "pg265-growth-augmented-large-capacity-training-protocol-v1", "training_sources": [str(PG263_DATASET.relative_to(ROOT)), str(PG264_DATASET.relative_to(ROOT))], "pg264_audit_file": str(PG264_AUDIT.relative_to(ROOT)), "capacity_variants": list(PG260.CAPACITY_VARIANTS), "promotion_blocked": True})
        protocol["protocol_sha256"] = ""
        protocol["protocol_sha256"] = _digest(protocol)
        PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": report["status"], "combined_records": len(rows), "pg264_records": len(pg264_rows), "selected": report.get("selected"), "judge": report.get("independent_final_judge"), "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2), flush=True)
        return code
    finally:
        _clear_marker()


if __name__ == "__main__":
    raise SystemExit(main())
