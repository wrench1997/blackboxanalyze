"""PG-254: enlarge the frozen-XXL probe gate with audited Pikachu routes.

PG-252 proved the causal action target on a route/seed holdout.  PG-254 adds
the per-route payload catalog produced by PG-253, trains on a subset of its
abstract route facts, and holds out the remaining SQL/XSS route families.
The target is still only ``safe probe availability``; SQL/XSS effects remain
evaluator-only and no raw payload/response is persisted.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch

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


PG252 = _load("run_pg252_probe_gate_capacity_training.py")
PG249 = PG252.PG249
PG248 = PG252.PG248
PG237 = PG252.PG237
from app.pg252_probe_gate import SCHEMA_VERSION as PROBE_SCHEMA, build_probe_gate_record, build_probe_gate_rows  # noqa: E402


RESEARCH = ROOT / "research"
PG253_REPORT = RESEARCH / "pg253_pikachu_payload_catalog_report_v1.json"
PG244_DATASET = RESEARCH / "pg244_failure_repair_capacity_training_dataset_v1.json"
PG246_DATASET = RESEARCH / "pg246_vulnerableapp_independent_dom_holdout_dataset_v1.json"
PG230_DATASET = RESEARCH / "pg230_next_token_quality_funnel_dataset_v1.json"
REPORT = RESEARCH / "pg254_pikachu_payload_catalog_capacity_training_report_v1.json"
DATASET = RESEARCH / "pg254_pikachu_payload_catalog_capacity_training_dataset_v1.json"
TRACE = RESEARCH / "pg254_pikachu_payload_catalog_capacity_training_trace_v1.json"
PROTOCOL = RESEARCH / "pg254_pikachu_payload_catalog_capacity_training_protocol_v1.json"
MARKDOWN = RESEARCH / "pg254_pikachu_payload_catalog_capacity_training_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg254-pikachu-payload-catalog-capacity-v1"

CATALOG_TRAIN_SOURCE = "pg253_payload_catalog_train"
CATALOG_HOLDOUT_SOURCE = "pg253_payload_catalog_holdout"
_ORIGINAL_LOAD_RECORDS = PG252._load_records
_ORIGINAL_PIKA_HOLDOUT = PG252._pika_holdout


def _catalog_rows() -> tuple[list[dict[str, Any]], dict[str, int]]:
    catalog = json.loads(PG253_REPORT.read_text(encoding="utf-8-sig"))
    entries = [dict(row) for row in catalog.get("entries", [])]
    # Hold out complete route families rather than random rows.  The training
    # side still receives new audited SQL/XSS route facts, while the holdout
    # tests whether the abstract gate transfers to unseen connections.
    holdout_paths = {
        "/vul/sqli/sqli_blind_b.php",
        "/vul/sqli/sqli_blind_t.php",
        "/vul/sqli/sqli_widebyte.php",
        "/vul/xss/xss_03.php",
        "/vul/xss/xss_reflected_get.php",
        "/vul/dir/dir_list.php",
        "/vul/urlredirect/urlredirect.php",
    }
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        path = str(entry.get("route", ""))
        source = CATALOG_HOLDOUT_SOURCE if path in holdout_paths else CATALOG_TRAIN_SOURCE
        family = str(entry.get("family", ""))
        typed = bool(entry.get("typed_oracle") == "dom_nojs_dual")
        if family == "sql":
            typed = bool(int(entry.get("typed_effect_confirmed_count", 0) or 0) > 0)
            if entry.get("probe_class") == "blind_boolean":
                typed = bool(int(entry.get("boolean_effect_confirmed_count", 0) or 0) > 0)
        if family not in {"sql", "xss"}:
            typed = False
        rows.append(
            build_probe_gate_record(
                {
                    "source": source,
                    "split_source": source,
                    "record_id": f"pg253:{index}:{path}",
                    "seed": 25401 if source == CATALOG_TRAIN_SOURCE else 25402,
                    "surface_class": "dom_surface" if family == "xss" else "sql_surface" if family == "sql" else "generic_surface",
                    "method": str(entry.get("method", "GET")),
                    "field_count": len(entry.get("fields") or []),
                    "oracle_available": typed,
                    "fresh_reset_ok": True,
                    "reset_completed": True,
                    "binding_valid": True,
                    "source_evidence_hash": str(entry.get("evidence_hash", "")),
                }
            )
        )
    return rows, {"catalog_entries": len(entries), "catalog_train_rows": sum(int(row["split_source"] == CATALOG_TRAIN_SOURCE) for row in rows), "catalog_holdout_rows": sum(int(row["split_source"] == CATALOG_HOLDOUT_SOURCE) for row in rows)}


def _load_records() -> tuple[list[dict[str, Any]], dict[str, int]]:
    base_rows, base_counts = PG249._load_records()
    probe_rows = build_probe_gate_rows(base_rows)
    catalog_rows, catalog_counts = _catalog_rows()
    rows = list(base_rows) + probe_rows + catalog_rows
    return rows, {**base_counts, "probe_gate_schema": PROBE_SCHEMA, "probe_gate_records": len(probe_rows), **catalog_counts, "combined_input_records": len(rows)}


def _pika_holdout(row: dict[str, Any]) -> bool:
    return _ORIGINAL_PIKA_HOLDOUT(row) or (str(row.get("split_source", row.get("source", ""))) == CATALOG_HOLDOUT_SOURCE and int(row.get("seed", 0) or 0) == 25402)


def _configure() -> None:
    PG252._configure()
    PG237._load_records = _load_records
    PG252._load_records = _load_records
    PG252._pika_holdout = _pika_holdout
    # PG-252's finalizer reads its own module globals, not PG-237's mutable
    # trainer paths.  Redirect those globals before reusing the independent
    # evaluator so PG-254 cannot overwrite the PG-252 report.
    PG252.REPORT = REPORT
    PG252.DATASET = DATASET
    PG252.TRACE = TRACE
    PG252.PROTOCOL = PROTOCOL
    PG252.MARKDOWN = MARKDOWN
    PG237.HOLDOUT_SOURCE_SEED_PAIRS = (
        ("pg242_pikachu_source_native", (24202,)),
        ("pg244_pikachu_sql_repair", (24402,)),
        ("pg244_pikachu_xss_repair", (24402,)),
        (CATALOG_HOLDOUT_SOURCE, (25402,)),
    )
    PG237.CAPACITY_VARIANTS = (2048, 4096)
    PG237.TRAIN_STEPS = 120
    PG237.ARTIFACT_DIR = ARTIFACT_DIR
    PG237.EXPERIMENT_ID = "pg254_pikachu_payload_catalog"
    PG237.REPORT = REPORT
    PG237.DATASET = DATASET
    PG237.TRACE = TRACE
    PG237.PROTOCOL = PROTOCOL
    PG237.MARKDOWN = MARKDOWN


def _finalize() -> dict[str, Any]:
    # Reuse PG-252's independent split/evaluator, now fed by the extended
    # loader and route holdout.  Then rewrite provenance to this experiment.
    PG252._finalize()
    report = json.loads(REPORT.read_text(encoding="utf-8-sig"))
    dataset = json.loads(DATASET.read_text(encoding="utf-8-sig"))
    trace = json.loads(TRACE.read_text(encoding="utf-8-sig"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8-sig"))
    report.update(
        {
            "protocol_id": "pg-pk-254-pikachu-payload-catalog-capacity-training-v1",
            "schema_version": "pg254-pikachu-payload-catalog-capacity-training-v1",
            "status": "completed_pikachu_payload_catalog_route_holdout_capacity_training",
            "source_datasets": [str(PG253_REPORT.relative_to(ROOT)), str(PG244_DATASET.relative_to(ROOT)), str(PG246_DATASET.relative_to(ROOT)), str(PG230_DATASET.relative_to(ROOT))],
            "catalog_route_holdout": {"train_source": CATALOG_TRAIN_SOURCE, "holdout_source": CATALOG_HOLDOUT_SOURCE, "holdout_is_route_family_disjoint": True},
            "probe_gate_schema": PROBE_SCHEMA,
            "training_eligible": bool(report.get("independent_final_judge", {}).get("pass")),
            "promotion": {"training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "payload_catalog_promotion_allowed": False, "judge_decision": report.get("independent_final_judge", {}).get("decision", "blocked")},
            "honesty": {**dict(report.get("honesty") or {}), "catalog_route_holdout_disjoint": True, "catalog_payload_strings_not_tokenized": True, "sql_ast_results_not_model_input": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
        }
    )
    report["report_sha256"] = PG237.digest(report)
    dataset.update({"schema_version": "pg254-pikachu-payload-catalog-capacity-training-dataset-v1", "source_datasets": report["source_datasets"], "catalog_route_holdout": report["catalog_route_holdout"], "contract": {**dict(dataset.get("contract") or {}), "catalog_payload_strings_not_tokenized": True, "route_family_holdout": True, "sql_ast_results_off_input": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}})
    dataset["dataset_sha256"] = PG237.digest(dataset)
    protocol.update({"protocol_id": report["protocol_id"], "schema_version": "pg254-pikachu-payload-catalog-capacity-training-protocol-v1", "training_sources": report["source_datasets"], "catalog_route_holdout": report["catalog_route_holdout"], "probe_target": "safe probe availability only; eventual typed effect remains evaluator-only", "capacity_variants": list(PG237.CAPACITY_VARIANTS), "train_steps": PG237.TRAIN_STEPS, "promotion_blocked": True, "raw_payload_and_response_excluded": True})
    protocol["protocol_sha256"] = PG237.digest(protocol)
    trace.update({"schema_version": "pg254-pikachu-payload-catalog-capacity-training-trace-v1", "catalog_route_holdout": report["catalog_route_holdout"], "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    PG237._write(REPORT, report)
    PG237._write(DATASET, dataset)
    PG237._write(PROTOCOL, protocol)
    PG237._write(TRACE, trace)
    MARKDOWN.write_text("\n".join(["# PG-254 Pikachu payload-catalog capacity training", "", f"catalog train={report['counts'].get('catalog_train_rows', 0)}; catalog holdout={report['counts'].get('catalog_holdout_rows', 0)}; total train={report['counts'].get('train_rows')}; total holdout={report['counts'].get('holdout_rows')}", f"final_judge={report.get('independent_final_judge', {}).get('decision')}", "", "只训练抽象 safe-probe gate；payload wire 与 SQL/XSS oracle 结果不进入输入，真实发送仍由 PG-253/PG-250 回放验证。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "counts": report["counts"], "selected": report.get("selected"), "final_judge": report.get("independent_final_judge"), "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    _configure()
    PG237.main()
    _finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
