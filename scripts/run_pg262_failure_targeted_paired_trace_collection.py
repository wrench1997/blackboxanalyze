# -*- coding: utf-8 -*-
"""PG-262: targeted fresh paired traces for PG-260 confusion classes.

This wrapper reuses the audited local Pikachu child runners but schedules new
seeds/routes.  It is intentionally separate from PG-260 so source/seed
holdout boundaries remain visible.  No network target or raw body is added.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location("pg262_collection", ROOT / "scripts" / "run_pg260_fresh_paired_trace_collection.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-260 collector")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG260 = _load()
RESEARCH = ROOT / "research"
PG260.RUN_DIR = RESEARCH / "pg262_child_runs"
PG260.REPORT = RESEARCH / "pg262_targeted_paired_trace_collection_report_v1.json"
PG260.DATASET = RESEARCH / "pg262_targeted_paired_trace_collection_dataset_v1.json"
PG260.TRACE = RESEARCH / "pg262_targeted_paired_trace_collection_trace_v1.json"
PG260.PROTOCOL = RESEARCH / "pg262_targeted_paired_trace_collection_protocol_v1.json"
PG260.MARKDOWN = RESEARCH / "pg262_targeted_paired_trace_collection_report_v1.md"
RUN_MARKER = RESEARCH / "pg262_training_running.json"
PG260.SQL_SEEDS = tuple(26201 + i for i in range(8))
PG260.SQL_ROUTE_PATHS = (
    "/vul/sqli/sqli_blind_b.php",
    "/vul/sqli/sqli_blind_t.php",
    "/vul/sqli/sqli_id.php",
    "/vul/sqli/sqli_search.php",
    "/vul/sqli/sqli_str.php",
    "/vul/sqli/sqli_x.php",
    "/vul/sqli/sqli_id.php",
    "/vul/sqli/sqli_search.php",
)
PG260.XSS_SEEDS = tuple(26211 + i for i in range(6))
PG260.XSS_CASE_IDS = ("xss_reflected_get", "xss_htmlspecialchars_02", "xss_href_03", "xss_js_04", "xss_dom_x", "xss_reflected_post")
PG260.BOOLEAN_SEEDS = tuple(26221 + i for i in range(3))
PG260.WIDEBYTE_SEEDS = tuple(26231 + i for i in range(3))


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _write_run_marker() -> None:
    RUN_MARKER.write_text(
        json.dumps(
            {"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(), "report": str(PG260.REPORT.relative_to(ROOT)), "protocol_id": "pg-pk-262-targeted-paired-trace-collection-v1"},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _clear_run_marker() -> None:
    try:
        RUN_MARKER.unlink(missing_ok=True)
    except OSError:
        pass


def main() -> int:
    _write_run_marker()
    try:
        return _run()
    finally:
        _clear_run_marker()


def _run() -> int:
    code = PG260.main()
    report = json.loads(PG260.REPORT.read_text(encoding="utf-8"))
    dataset = json.loads(PG260.DATASET.read_text(encoding="utf-8"))
    trace = json.loads(PG260.TRACE.read_text(encoding="utf-8"))
    protocol = json.loads(PG260.PROTOCOL.read_text(encoding="utf-8"))
    for payload in (report, dataset, trace, protocol):
        for key in ("protocol_id", "schema_version"):
            if key in payload:
                payload[key] = str(payload[key]).replace("260", "262")
    for row in list(dataset.get("records") or []):
        if isinstance(row, dict):
            row["source"] = str(row.get("source", "")).replace("pg260_", "pg262_")
            row["parent_record_id"] = str(row.get("parent_record_id", "")).replace("pg260:", "pg262:")
    for row in list(trace.get("records") or []):
        if isinstance(row, dict):
            row["source"] = str(row.get("source", "")).replace("pg260_", "pg262_")
            row["parent_record_id"] = str(row.get("parent_record_id", "")).replace("pg260:", "pg262:")
    if isinstance(dataset.get("counts"), dict):
        dataset["counts"]["source_counts"] = {str(key).replace("pg260_", "pg262_"): value for key, value in dict(dataset["counts"].get("source_counts") or {}).items()}
    report["targeted_for"] = ["sql_syntax route/seed confusion", "dom/sql family boundary", "fresh GET/POST pairing"]
    report["promotion"] = {"training_promotion_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}
    report["report_sha256"] = ""
    report["report_sha256"] = _digest(report)
    dataset["dataset_sha256"] = ""
    dataset["dataset_sha256"] = _digest(dataset)
    protocol["targeted_schedule"] = True
    protocol["promotion_blocked_until_pg263_training_judge"] = True
    protocol["protocol_sha256"] = ""
    protocol["protocol_sha256"] = _digest(protocol)
    PG260.REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PG260.DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PG260.TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PG260.PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PG260.MARKDOWN.write_text(PG260.MARKDOWN.read_text(encoding="utf-8").replace("PG-260", "PG-262") + "\n本轮是针对混淆矩阵的 targeted fresh route schedule，不进入训练直到 PG-263 独立容量判官通过。\n", encoding="utf-8")
    print(json.dumps({"status": report.get("status"), "counts": report.get("counts"), "report": str(PG260.REPORT.relative_to(ROOT)), "dataset": str(PG260.DATASET.relative_to(ROOT)), "targeted_for": report["targeted_for"]}, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
