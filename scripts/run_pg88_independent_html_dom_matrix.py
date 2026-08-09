"""PG-88: independent fresh Docker replay of the HTML/DOM triplet matrix.

PG-88 deliberately reuses the audited PG-74 collector implementation, but it
uses a new seed set and writes a separate evaluation-only artifact namespace.
The objective is to test cross-seed stability of the *collection contract*,
not to silently add rows to training or memory.  Every case still gets a
neutral request, a matched negative probe, a typed positive probe, a fresh
container reset and bounded evidence hashes.

The wrapper keeps the collector code path identical so a pass cannot be
explained by a collector rewrite.  The report records the implementation
reuse and independent seed set explicitly.
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

PG74_SCRIPT = ROOT / "scripts" / "run_pg74_causal_triplet_collector.py"
SEEDS = (88101, 88107, 88111, 88117)
PROTOCOL_ID = "pg-pk-88-independent-html-dom-matrix-v1"
SCHEMA_VERSION = "sift-pg88-independent-html-dom-matrix-v1"
REPORT_PATH = ROOT / "research" / "pg88_independent_html_dom_matrix_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg88_independent_html_dom_matrix_protocol_v1.json"
CATALOG_PATH = ROOT / "research" / "pg88_independent_html_dom_matrix_catalog_v1.json"
TRACE_PATH = ROOT / "research" / "pg88_independent_html_dom_matrix_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg88_independent_html_dom_matrix_report_v1.md"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_paths(base: Any) -> None:
    """Redirect the audited collector's output without changing its logic."""

    base.PROTOCOL_ID = PROTOCOL_ID
    base.SCHEMA_VERSION = SCHEMA_VERSION
    base.SEEDS = SEEDS
    base.REPORT_PATH = REPORT_PATH
    base.PROTOCOL_PATH = PROTOCOL_PATH
    base.CATALOG_PATH = CATALOG_PATH
    base.TRACE_PATH = TRACE_PATH
    base.MARKDOWN_PATH = MARKDOWN_PATH


def _annotate(report: dict[str, Any], trace: dict[str, Any], catalog: dict[str, Any] | None) -> dict[str, Any]:
    report["protocol_id"] = PROTOCOL_ID
    report["schema_version"] = SCHEMA_VERSION
    source = report.setdefault("source", {})
    source["collector_implementation"] = "pg74_causal_triplet_collector_v1_reused"
    source["independent_seed_set"] = list(SEEDS)
    source["fresh_replay_round"] = "pg88"
    source["matrix_case_count_per_seed"] = 7
    source["matrix_case_families"] = ["xss", "injection", "url_redirect"]
    source["target_surface_scope"] = ["html", "dom", "sql_response", "redirect"]
    source["loopback_only"] = True
    source["external_network"] = False
    metrics = report.setdefault("metrics", {})
    metrics["independent_seed_count"] = len(SEEDS)
    metrics["expected_case_count"] = len(SEEDS) * 7
    metrics["fresh_replay_rounds"] = len(SEEDS)
    report["hard_gate"]["claim_allowed"] = False
    report["promotion"] = {
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "training_catalog_generated": False,
        "status": "evaluation_only_independent_matrix",
        "reason": "PG88 must pass frozen PG86 replay and cross-seed/implementation gates before any training or memory use",
    }
    report["artifacts"] = {
        "report": str(REPORT_PATH.relative_to(ROOT)),
        "protocol": str(PROTOCOL_PATH.relative_to(ROOT)),
        "trace": str(TRACE_PATH.relative_to(ROOT)),
        "catalog": str(CATALOG_PATH.relative_to(ROOT)) if catalog is not None else None,
    }
    trace["protocol_id"] = PROTOCOL_ID
    trace["schema_version"] = "sift-pg88-independent-html-dom-matrix-trace-v1"
    trace["evaluation_only"] = True
    trace["training_eligible"] = False
    trace["independent_seed_set"] = list(SEEDS)
    trace["fresh_replay_round"] = "pg88"
    trace["long_term_memory_write"] = False
    if catalog is not None:
        catalog["schema_version"] = "sift-pg88-independent-html-dom-matrix-catalog-v1"
        catalog["catalog_id"] = "pg88-independent-html-dom-matrix-evaluation-only"
        catalog["evaluation_only"] = True
        catalog["training_eligible"] = False
        catalog["long_term_memory_write"] = False
    return report


def run(*, skip_docker: bool = False) -> dict[str, Any]:
    base = _load(PG74_SCRIPT, "pg88_pg74_collector_runtime")
    _patch_paths(base)
    report = base.run(skip_docker=skip_docker, seeds=SEEDS)
    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8")) if CATALOG_PATH.exists() else None
    report = _annotate(report, trace, catalog)
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if catalog is not None:
        CATALOG_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "sift-pg88-independent-html-dom-matrix-protocol-v1",
        "collector_implementation": "pg74_causal_triplet_collector_v1_reused",
        "pinned_image": report.get("source", {}).get("pinned_image"),
        "seed_set": list(SEEDS),
        "fresh_container_per_triplet": True,
        "loopback_only": True,
        "external_network": False,
        "methods": ["GET", "POST"],
        "families": ["xss", "injection", "url_redirect"],
        "typed_oracle_after_target": True,
        "negative_control_required": True,
        "evidence_sha256_required": True,
        "raw_persistence_forbidden": True,
        "run_result": {"hard_gate": report["hard_gate"], "training_allowed": False, "memory_promotion_allowed": False},
        "next_experiment": "PG89 frozen PG86 replay on the independent matrix",
    }
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(
        "# PG-88 Independent HTML/DOM matrix\n\n"
        + f"seed_count={len(SEEDS)}；cases={report['metrics'].get('triplet_case_count', 0)}；"
        + f"GET/POST={report['metrics'].get('get_post_covered', {})}；"
        + f"hard_gate={report['hard_gate']['status']}。\n\n"
        + "本轮仅为 evaluation-only；training/memory promotion=false。\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-docker", action="store_true")
    args = parser.parse_args()
    result = run(skip_docker=bool(args.skip_docker))
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "status": result["hard_gate"]["status"],
        "triplet_case_count": result["metrics"]["triplet_case_count"],
        "typed_positive_count": result["metrics"]["typed_positive_count"],
        "typed_negative_oracle_count": result["metrics"]["typed_negative_oracle_count"],
        "get_post_covered": result["metrics"]["get_post_covered"],
        "training_allowed": False,
        "memory_promotion_allowed": False,
    }, ensure_ascii=False, indent=2))
