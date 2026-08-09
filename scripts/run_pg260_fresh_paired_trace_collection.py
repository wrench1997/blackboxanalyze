"""PG-260: collect fresh, route-paired local Pikachu traces.

PG-259 showed that the auxiliary belief/probe heads were learnable while the
Rule-IR route holdout was still weak.  This tranche adds *real* new seeds, but
keeps each seed on one deliberately chosen route so that route and seed are
not silently coupled.  Every child runner still owns the actual AI candidate,
independent reference, matched negative, fresh-container and typed-oracle
checks.  This coordinator only joins bounded projections; executable request
values and response bodies remain ephemeral stdout-only material.

The route schedule is balanced across GET/POST and syntax/boolean/widebyte /
DOM-marker/oracle-gap surfaces.  It is intentionally small enough to run as a
repeatable research tranche rather than hiding a long all-routes sweep behind
one aggregate score.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg231_feedback_trajectory import prepare_feedback_record  # noqa: E402


RESEARCH = ROOT / "research"
RUN_DIR = RESEARCH / "pg260_child_runs"
REPORT = RESEARCH / "pg260_fresh_paired_trace_collection_report_v1.json"
DATASET = RESEARCH / "pg260_fresh_paired_trace_collection_dataset_v1.json"
TRACE = RESEARCH / "pg260_fresh_paired_trace_collection_trace_v1.json"
PROTOCOL = RESEARCH / "pg260_fresh_paired_trace_collection_protocol_v1.json"
MARKDOWN = RESEARCH / "pg260_fresh_paired_trace_collection_report_v1.md"

# Eight independent seed cells per class.  The route schedules deliberately
# contain both GET and POST where the source exposes both channels.
SQL_SEEDS = tuple(26001 + i for i in range(8))
SQL_ROUTE_PATHS = (
    "/vul/sqli/sqli_blind_b.php",
    "/vul/sqli/sqli_id.php",
    "/vul/sqli/sqli_search.php",
    "/vul/sqli/sqli_str.php",
    "/vul/sqli/sqli_x.php",
    "/vul/sqli/sqli_widebyte.php",
    "/vul/sqli/sqli_blind_t.php",
    "/vul/sqli/sqli_id.php",
)
XSS_SEEDS = tuple(26011 + i for i in range(8))
XSS_CASE_IDS = (
    "xss_reflected_get",
    "xss_filter_01",
    "xss_htmlspecialchars_02",
    "xss_href_03",
    "xss_js_04",
    "xss_dom",
    "xss_dom_x",
    "xss_reflected_post",
)
BOOLEAN_SEEDS = tuple(26021 + i for i in range(8))
WIDEBYTE_SEEDS = tuple(26031 + i for i in range(8))

# Loading the frozen field/body adapter is much more expensive than one
# bounded route replay on the local GPU.  Reuse the already-frozen inference
# objects inside this coordinator; the target container is still recreated for
# every route/seed and no model state is updated during collection.
_SQL_RUNNER: Any | None = None
_SQL_CACHE: dict[str, Any] = {}
_XSS_RUNNER: Any | None = None
_XSS_CACHE: dict[str, Any] = {}


def _load(filename: str, unique_name: str) -> Any:
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(unique_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_sql(seed: int, route_path: str, index: int) -> dict[str, Any]:
    report_path = RUN_DIR / f"sql_{seed}_{index}_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8-sig"))
    global _SQL_RUNNER
    if _SQL_RUNNER is None:
        _SQL_RUNNER = _load("run_pg255_pikachu_fixed_sql_pg254_replay.py", "pg260_sql_shared")
        import torch

        _SQL_CACHE["routes"] = [dict(route) for route in _SQL_RUNNER.PG212._routes()]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        legacy, vocabulary = _SQL_RUNNER.ORIGINAL_LOAD_MODEL(device)
        gate = _SQL_RUNNER._load_gate(device)
        _SQL_CACHE.update({"legacy": legacy, "vocabulary": vocabulary, "gate": gate})
        _SQL_RUNNER.ORIGINAL_LOAD_MODEL = lambda _device: (legacy, vocabulary)
        _SQL_RUNNER._load_gate = lambda _device: gate
    runner = _SQL_RUNNER
    routes = [dict(route) for route in list(_SQL_CACHE.get("routes") or []) if str(route.get("path")) == route_path]
    if len(routes) != 1:
        raise RuntimeError(f"PG-260 SQL route not found: {route_path}")
    runner.SEEDS = (int(seed),)
    runner.PG212._routes = lambda routes=routes: [dict(routes[0])]
    runner.PG214.BASE_PORT = 12000 + index
    runner.WIRE_LOG.clear()
    runner.REPORT = report_path
    runner.TRACE = RUN_DIR / f"sql_{seed}_{index}_trace.json"
    runner.PROTOCOL = RUN_DIR / f"sql_{seed}_{index}_protocol.json"
    runner.MARKDOWN = RUN_DIR / f"sql_{seed}_{index}.md"
    code = runner.main()
    if code != 0:
        raise RuntimeError(f"PG-255 child failed for SQL seed {seed} route {route_path}: {code}")
    return json.loads(runner.REPORT.read_text(encoding="utf-8-sig"))


def _run_xss(seed: int, case_id: str, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    report_path = RUN_DIR / f"xss_{seed}_{index}_report.json"
    dataset_path = RUN_DIR / f"xss_{seed}_{index}_dataset.json"
    if report_path.exists() and dataset_path.exists():
        return (json.loads(report_path.read_text(encoding="utf-8-sig")), json.loads(dataset_path.read_text(encoding="utf-8-sig")))
    global _XSS_RUNNER
    if _XSS_RUNNER is None:
        _XSS_RUNNER = _load("run_pg242_pikachu_xss_dom_acceptance.py", "pg260_xss_shared")
        import torch

        _XSS_CACHE["cases"] = [dict(case) for case in _XSS_RUNNER._case_specs()]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, vocabulary = _XSS_RUNNER.PG208._load_model(device)
        _XSS_CACHE.update({"model": model, "vocabulary": vocabulary})
        _XSS_RUNNER.PG208._load_model = lambda _device: (model, vocabulary)
    runner = _XSS_RUNNER
    cases = [dict(case) for case in list(_XSS_CACHE.get("cases") or []) if str(case.get("case_id")) == case_id]
    if len(cases) != 1:
        raise RuntimeError(f"PG-260 XSS case not found: {case_id}")
    runner.SEEDS = (int(seed),)
    runner._case_specs = lambda cases=cases: [dict(cases[0])]
    runner.BASE_PORT = 13000 + index
    runner.PG214.BASE_PORT = runner.BASE_PORT
    runner.REPORT = report_path
    runner.DATASET = dataset_path
    runner.TRACE = RUN_DIR / f"xss_{seed}_{index}_trace.json"
    runner.PROTOCOL = RUN_DIR / f"xss_{seed}_{index}_protocol.json"
    runner.MARKDOWN = RUN_DIR / f"xss_{seed}_{index}.md"
    code = runner.main()
    if code != 0:
        raise RuntimeError(f"PG-242 child failed for XSS seed {seed} case {case_id}: {code}")
    return (
        json.loads(runner.REPORT.read_text(encoding="utf-8-sig")),
        json.loads(runner.DATASET.read_text(encoding="utf-8-sig")),
    )


def _run_boolean(seed: int, index: int) -> dict[str, Any]:
    runner = _load("run_pg221_pikachu_boolean_blind_oracle.py", f"pg260_boolean_{seed}_{index}")
    runner.SEEDS = (int(seed),)
    runner.PG214.BASE_PORT = 14000 + index
    runner.REPORT = RUN_DIR / f"boolean_{seed}_{index}_report.json"
    runner.DATASET = RUN_DIR / f"boolean_{seed}_{index}_dataset.json"
    runner.TRACE = RUN_DIR / f"boolean_{seed}_{index}_trace.json"
    runner.PROTOCOL = RUN_DIR / f"boolean_{seed}_{index}_protocol.json"
    runner.MARKDOWN = RUN_DIR / f"boolean_{seed}_{index}.md"
    if runner.REPORT.exists():
        return json.loads(runner.REPORT.read_text(encoding="utf-8-sig"))
    code = runner.main()
    if code != 0:
        raise RuntimeError(f"PG-221 child failed for boolean seed {seed}: {code}")
    return json.loads(runner.REPORT.read_text(encoding="utf-8-sig"))


def _run_widebyte(seed: int, index: int) -> dict[str, Any]:
    runner = _load("run_pg256_pikachu_widebyte_oracle.py", f"pg260_widebyte_{seed}_{index}")
    runner.SEEDS = (int(seed),)
    runner.PG214.BASE_PORT = 15000 + index
    runner.REPORT = RUN_DIR / f"widebyte_{seed}_{index}_report.json"
    runner.TRACE = RUN_DIR / f"widebyte_{seed}_{index}_trace.json"
    runner.PROTOCOL = RUN_DIR / f"widebyte_{seed}_{index}_protocol.json"
    runner.MARKDOWN = RUN_DIR / f"widebyte_{seed}_{index}.md"
    runner.WIRE_LOG.clear()
    if runner.REPORT.exists():
        return json.loads(runner.REPORT.read_text(encoding="utf-8-sig"))
    code = runner.main()
    if code != 0:
        raise RuntimeError(f"PG-256 child failed for widebyte seed {seed}: {code}")
    return json.loads(runner.REPORT.read_text(encoding="utf-8-sig"))


def _rename_source(rows: list[dict[str, Any]], source: str, prefix: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["source"] = source
        item["parent_record_id"] = f"pg260:{prefix}:{item.get('seed')}:{item.get('route') or item.get('parent_record_id', '')}"
        result.append(item)
    return result


def _join_sql(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    helper = _load("run_pg259_fresh_local_trace_collection.py", "pg260_collection_helpers_sql")
    rows: list[dict[str, Any]] = []
    for report in reports:
        rows.extend(helper._sql_feedback_rows(report))
    return _rename_source(rows, "pg260_pikachu_sql_paired", "sql")


def _join_xss(reports: list[tuple[dict[str, Any], dict[str, Any]]]) -> list[dict[str, Any]]:
    helper = _load("run_pg259_fresh_local_trace_collection.py", "pg260_collection_helpers_xss")
    rows: list[dict[str, Any]] = []
    for report, dataset in reports:
        rows.extend(helper._xss_feedback_rows(report, dataset))
    return _rename_source(rows, "pg260_pikachu_xss_paired", "xss")


def _join_boolean(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    helper = _load("run_pg259_fresh_local_trace_collection.py", "pg260_collection_helpers_boolean")
    rows: list[dict[str, Any]] = []
    for report in reports:
        rows.extend(helper._boolean_feedback_rows(report))
    return _rename_source(rows, "pg260_pikachu_boolean_paired", "boolean")


def _join_widebyte(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    helper = _load("run_pg259_fresh_local_trace_collection.py", "pg260_collection_helpers_widebyte")
    rows: list[dict[str, Any]] = []
    for report in reports:
        rows.extend(helper._widebyte_feedback_rows(report))
    return _rename_source(rows, "pg260_pikachu_widebyte_paired", "widebyte")


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if len(SQL_SEEDS) != len(SQL_ROUTE_PATHS) or len(XSS_SEEDS) != len(XSS_CASE_IDS):
        raise RuntimeError("PG-260 route schedule and seed schedule must have equal lengths")
    sql_reports: list[dict[str, Any]] = []
    xss_reports: list[tuple[dict[str, Any], dict[str, Any]]] = []
    boolean_reports: list[dict[str, Any]] = []
    widebyte_reports: list[dict[str, Any]] = []
    if os.environ.get("PG260_REBUILD_ONLY") == "1":
        for index, seed in enumerate(SQL_SEEDS):
            path = RUN_DIR / f"sql_{seed}_{index}_report.json"
            sql_reports.append(json.loads(path.read_text(encoding="utf-8-sig")))
        for index, seed in enumerate(XSS_SEEDS):
            report_path = RUN_DIR / f"xss_{seed}_{index}_report.json"
            dataset_path = RUN_DIR / f"xss_{seed}_{index}_dataset.json"
            xss_reports.append((json.loads(report_path.read_text(encoding="utf-8-sig")), json.loads(dataset_path.read_text(encoding="utf-8-sig"))))
        for index, seed in enumerate(BOOLEAN_SEEDS):
            boolean_reports.append(json.loads((RUN_DIR / f"boolean_{seed}_{index}_report.json").read_text(encoding="utf-8-sig")))
        for index, seed in enumerate(WIDEBYTE_SEEDS):
            widebyte_reports.append(json.loads((RUN_DIR / f"widebyte_{seed}_{index}_report.json").read_text(encoding="utf-8-sig")))
    else:
        for index, (seed, route) in enumerate(zip(SQL_SEEDS, SQL_ROUTE_PATHS)):
            print(f"PG260 SQL seed={seed} route={route}", flush=True)
            sql_reports.append(_run_sql(seed, route, index))
        for index, (seed, case_id) in enumerate(zip(XSS_SEEDS, XSS_CASE_IDS)):
            print(f"PG260 XSS seed={seed} case={case_id}", flush=True)
            xss_reports.append(_run_xss(seed, case_id, index))
        for index, seed in enumerate(BOOLEAN_SEEDS):
            print(f"PG260 BOOLEAN seed={seed}", flush=True)
            boolean_reports.append(_run_boolean(seed, index))
        for index, seed in enumerate(WIDEBYTE_SEEDS):
            print(f"PG260 WIDEBYTE seed={seed}", flush=True)
            widebyte_reports.append(_run_widebyte(seed, index))

    rows = _join_sql(sql_reports) + _join_xss(xss_reports) + _join_boolean(boolean_reports) + _join_widebyte(widebyte_reports)
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, str, str]] = set()
    for row in rows:
        key = (str(row.get("source", "")), int(row.get("seed", 0) or 0), str(row.get("route", "")), str(row.get("route_source_sha256", "")), str(row.get("trajectory_hash", row.get("token_hash", ""))))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    source_counts = dict(Counter(str(row.get("source", "")) for row in unique))
    lane_counts = dict(Counter(str(row.get("lane", "")) for row in unique))
    counts = {
        "records": len(unique),
        "source_counts": source_counts,
        "lane_counts": lane_counts,
        "gold_count": int(lane_counts.get("gold", 0)),
        "hard_negative_count": int(lane_counts.get("hard_negative", 0)),
        "silver_count": int(lane_counts.get("silver", 0)),
        "quarantine_count": int(lane_counts.get("quarantine", 0)),
        "seed_sets": {"sql": list(SQL_SEEDS), "xss": list(XSS_SEEDS), "boolean": list(BOOLEAN_SEEDS), "widebyte": list(WIDEBYTE_SEEDS)},
        "route_schedule": {"sql": list(SQL_ROUTE_PATHS), "xss": list(XSS_CASE_IDS)},
        "fresh_seed_count_by_class": {"sql": len(SQL_SEEDS), "xss": len(XSS_SEEDS), "boolean": len(BOOLEAN_SEEDS), "widebyte": len(WIDEBYTE_SEEDS)},
        "ai_send_count": sum(int((report.get("counts") or {}).get("ai_candidate_send_count", (report.get("counts") or {}).get("ai_send_count", (report.get("counts") or {}).get("ai_candidate_pair_send_count", 0))) or 0) for report in sql_reports + [item[0] for item in xss_reports] + boolean_reports + widebyte_reports),
        "confirmed_positive_count": sum(int((report.get("counts") or {}).get("confirmed_positive_count", 0) or 0) for report in sql_reports + [item[0] for item in xss_reports] + boolean_reports + widebyte_reports),
        "negative_or_abstain_count": sum(int(row.get("lane") in {"hard_negative", "silver", "quarantine"}) for row in unique),
    }
    child_reports: list[dict[str, Any]] = []
    for path in sorted(RUN_DIR.glob("*_report.json")):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        child_reports.append({"path": str(path.relative_to(ROOT)), "sha256": _digest(payload), "protocol_id": payload.get("protocol_id"), "status": payload.get("status")})
    contract = {
        "schema_version": "pg260-fresh-paired-trace-collection-v1",
        "loopback_only": True,
        "fresh_container_per_episode": True,
        "ai_reference_negative_typed_oracle": True,
        "route_and_seed_are_explicit": True,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "training_promotion_allowed": False,
        "memory_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "oracle_gap_is_silver_not_positive": True,
    }
    dataset = {"schema_version": "pg260-fresh-paired-trace-collection-dataset-v1", "child_reports": child_reports, "records": unique, "counts": counts, "contract": contract}
    dataset["dataset_sha256"] = _digest(dataset)
    report = {
        "protocol_id": "pg-pk-260-fresh-paired-trace-collection-v1",
        "schema_version": "pg260-fresh-paired-trace-collection-report-v1",
        "status": "completed_fresh_route_paired_local_trace_collection",
        "counts": counts,
        "child_reports": child_reports,
        "promotion": {"training_promotion_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "honesty": {"all_targets_authorized_loopback": True, "model_participated_in_send_path": True, "reference_and_negative_independent": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "general_web_capability_not_established": True, "class_quota_is_not_a_capability_claim": True},
    }
    report["report_sha256"] = _digest(report)
    protocol = {
        "protocol_id": report["protocol_id"],
        "schema_version": "pg260-fresh-paired-trace-collection-protocol-v1",
        "child_runners": ["PG-255 fixed SQL", "PG-242 controlled DOM", "PG-221 boolean differential", "PG-256 widebyte"],
        "fresh_seed_count_per_class": 8,
        "route_pairing_is_explicit": True,
        "get_post_required": True,
        "typed_oracle_and_matched_negative_required": True,
        "raw_wire_storage": "stdout-only ephemeral",
        "promotion_blocked_until_pg260_training_judge": True,
    }
    protocol["protocol_sha256"] = _digest(protocol)
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(TRACE, {"schema_version": "pg260-fresh-paired-trace-collection-trace-v1", "counts": counts, "records": unique, "child_reports": child_reports, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    _write(PROTOCOL, protocol)
    MARKDOWN.write_text("\n".join(["# PG-260 fresh paired local trace collection", "", f"records={counts['records']}; gold={counts['gold_count']}; hard_negative={counts['hard_negative_count']}; silver={counts['silver_count']}", f"sources={counts['source_counts']}", "八个独立 seed cell 分别映射到明确路由/表面；所有 wire 只在本地运行时 stdout 临时展示，数据集只保留抽象 token、响应投影和证据哈希。", "训练晋级、长期记忆和公网能力声明均保持关闭，下一步由 PG-260 capacity training 独立验收。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "counts": counts, "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
