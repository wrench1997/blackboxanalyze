# -*- coding: utf-8 -*-
"""PG-264: add a fresh, balanced Pikachu tranche for capacity training.

This is a data-growth experiment, not a public scanner.  The child runners
still perform the local AI/reference/negative/typed-oracle loop and only the
bounded projections are retained.  Raw request values and response bodies
remain ephemeral stdout-only material.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_base() -> Any:
    path = ROOT / "scripts" / "run_pg260_fresh_paired_trace_collection.py"
    spec = importlib.util.spec_from_file_location("pg264_growth_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-260 local collector")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base()
RESEARCH = ROOT / "research"
RUN_DIR = RESEARCH / "pg264_growth_child_runs"
REPORT = RESEARCH / "pg264_pikachu_growth_collection_report_v1.json"
DATASET = RESEARCH / "pg264_pikachu_growth_collection_dataset_v1.json"
TRACE = RESEARCH / "pg264_pikachu_growth_collection_trace_v1.json"
PROTOCOL = RESEARCH / "pg264_pikachu_growth_collection_protocol_v1.json"
MARKDOWN = RESEARCH / "pg264_pikachu_growth_collection_report_v1.md"
RUN_MARKER = RESEARCH / "pg264_collection_running.json"

# New seed cells are disjoint from PG-259/260/262.  The schedule keeps all
# four families and both transport methods represented in the fresh tranche.
BASE.RUN_DIR = RUN_DIR
BASE.REPORT = REPORT
BASE.DATASET = DATASET
BASE.TRACE = TRACE
BASE.PROTOCOL = PROTOCOL
BASE.MARKDOWN = MARKDOWN
BASE.SQL_SEEDS = tuple(26401 + i for i in range(8))
BASE.SQL_ROUTE_PATHS = (
    "/vul/sqli/sqli_blind_b.php",
    "/vul/sqli/sqli_id.php",
    "/vul/sqli/sqli_search.php",
    "/vul/sqli/sqli_str.php",
    "/vul/sqli/sqli_x.php",
    "/vul/sqli/sqli_widebyte.php",
    "/vul/sqli/sqli_blind_t.php",
    "/vul/sqli/sqli_id.php",
)
BASE.XSS_SEEDS = tuple(26411 + i for i in range(8))
BASE.XSS_CASE_IDS = (
    "xss_reflected_get",
    "xss_filter_01",
    "xss_htmlspecialchars_02",
    "xss_href_03",
    "xss_js_04",
    "xss_dom",
    "xss_dom_x",
    "xss_reflected_post",
)
BASE.BOOLEAN_SEEDS = tuple(26421 + i for i in range(8))
BASE.WIDEBYTE_SEEDS = tuple(26431 + i for i in range(8))


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _rename_record_fields(items: list[dict[str, Any]]) -> None:
    for item in items:
        if not isinstance(item, dict):
            continue
        if "source" in item:
            item["source"] = str(item["source"]).replace("pg260_", "pg264_")
        if "parent_record_id" in item:
            item["parent_record_id"] = str(item["parent_record_id"]).replace("pg260:", "pg264:")


def _write_growth_metadata() -> None:
    report = _read(REPORT)
    dataset = _read(DATASET)
    trace = _read(TRACE)
    protocol = _read(PROTOCOL)
    _rename_record_fields(list(dataset.get("records") or []))
    _rename_record_fields(list(trace.get("records") or []))
    report["protocol_id"] = "pg-pk-264-pikachu-growth-collection-v1"
    report["schema_version"] = "pg264-pikachu-growth-collection-report-v1"
    report["targeted_for"] = [
        "increase fresh route/seed support before larger adapter training",
        "preserve SQL/XSS/boolean/widebyte family balance",
        "keep GET/POST and typed-oracle evidence in the evaluation lane",
    ]
    report["growth_schedule"] = {
        "seed_ranges": {
            "sql": list(BASE.SQL_SEEDS),
            "xss": list(BASE.XSS_SEEDS),
            "boolean": list(BASE.BOOLEAN_SEEDS),
            "widebyte": list(BASE.WIDEBYTE_SEEDS),
        },
        "fresh_target": "127.0.0.1 Pikachu Docker only",
        "expected_episode_count": 32,
    }
    report["promotion"] = {
        "training_promotion_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    report["report_sha256"] = ""
    report["report_sha256"] = _digest(report)
    dataset["schema_version"] = "pg264-pikachu-growth-collection-dataset-v1"
    dataset["contract"] = dict(dataset.get("contract") or {}, fresh_seed_schedule=True, raw_payload_strings_stored=False, raw_response_bodies_stored=False, training_promotion_allowed=False, memory_promotion_allowed=False, vulnerability_claim_allowed=False)
    dataset["dataset_sha256"] = ""
    dataset["dataset_sha256"] = _digest(dataset)
    trace["schema_version"] = "pg264-pikachu-growth-collection-trace-v1"
    trace["raw_payload_strings_stored"] = False
    trace["raw_response_bodies_stored"] = False
    protocol["protocol_id"] = report["protocol_id"]
    protocol["schema_version"] = "pg264-pikachu-growth-collection-protocol-v1"
    protocol["fresh_seed_count_per_class"] = 8
    protocol["promotion_blocked_until_training_audit"] = True
    protocol["protocol_sha256"] = ""
    protocol["protocol_sha256"] = _digest(protocol)
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(TRACE, trace)
    _write(PROTOCOL, protocol)
    MARKDOWN.write_text(
        "\n".join(
            [
                "# PG-264 Pikachu growth collection",
                "",
                f"records={report.get('counts', {}).get('records', 0)}; fresh seeds per class=8",
                "目标仅为 127.0.0.1 Pikachu Docker；训练、长期记忆、Payload Catalog 晋级和公网漏洞声明均关闭。",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RUN_MARKER.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "report": str(REPORT.relative_to(ROOT)),
                "protocol_id": "pg-pk-264-pikachu-growth-collection-v1",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        code = BASE.main()
        _write_growth_metadata()
        print(json.dumps({"status": "completed_pg264_growth_collection", "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT)), "counts": _read(REPORT).get("counts", {})}, ensure_ascii=False, indent=2), flush=True)
        return code
    finally:
        RUN_MARKER.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())

