"""PG-91: fresh collection from the independent PG-35 HTTP implementation.

PG-35 is intentionally not Pikachu: it is an in-repo ``http.server`` target
with three route layouts, nine semantic surfaces, GET/POST transport and
identity/URL-percent encoding pairs.  This wrapper reuses the audited
collector implementation with a new seed set and quarantines the resulting
catalog/trace in a new evaluation-only namespace.
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

PG35_SCRIPT = ROOT / "scripts" / "run_pg35_independent_fixture_catalog.py"
SEEDS = (39101, 39107, 39111)
PROTOCOL_ID = "pg-pk-91-pg35-independent-collector-v1"
REPORT_PATH = ROOT / "research" / "pg91_pg35_independent_collector_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg91_pg35_independent_collector_protocol_v1.json"
CATALOG_PATH = ROOT / "research" / "pg91_pg35_independent_fixture_catalog_v1.json"
TRACE_PATH = ROOT / "research" / "pg91_pg35_independent_fixture_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg91_pg35_independent_collector_report_v1.md"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_paths(base: Any) -> None:
    base.SEEDS = SEEDS
    base.CATALOG_OUTPUT = CATALOG_PATH
    base.TRACE_OUTPUT = TRACE_PATH


def _hard_gate(catalog: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    samples = list(catalog.get("samples", []))
    positives = [row for row in samples if bool((row.get("oracle_projection") or {}).get("positive"))]
    negatives = [row for row in samples if not bool((row.get("oracle_projection") or {}).get("positive"))]
    methods = {str(row.get("method", "")).upper() for row in samples}
    encodings = {str(row.get("encoding", "")) for row in samples}
    target_ids = [str(row.get("target_instance_id", "")) for row in samples]
    checks = {
        "independent_target_implementation": catalog.get("independent_target_implementation") is True and trace.get("independent_target_implementation") is True,
        "sample_count": len(samples) == 648,
        "typed_positive_count": len(positives) == 288 and all(bool((row.get("oracle_projection") or {}).get("positive_authority")) for row in positives),
        "typed_negative_count": len(negatives) == 360 and all(bool((row.get("oracle_projection") or {}).get("positive")) is False for row in negatives),
        "fresh_target_per_sample": len(target_ids) == len(set(target_ids)) == 648 and all(bool((row.get("reset") or {}).get("fresh_target")) and bool((row.get("reset") or {}).get("completed")) for row in samples),
        "get_post_covered": methods == {"GET", "POST"},
        "encoding_pair_covered": encodings == {"identity", "url_percent"} and int(catalog.get("encoding_pair_count", 0)) == 324,
        "source_variants": int(catalog.get("source_count", 0)) == 3 and len({str(row.get("source_sha256", "")) for row in samples}) == 3,
        "evidence_hashes_unique": len({str((row.get("evidence") or {}).get("evidence_hash", "")) for row in samples}) == len(samples),
        "trace_get_post": trace.get("methods") == ["GET", "POST"] and len(trace.get("steps", [])) == len(samples),
        "no_raw_persistence": catalog.get("raw_probe_strings_stored") is False and catalog.get("raw_response_bodies_stored") is False and trace.get("raw_probe_strings_stored") is False and trace.get("raw_response_bodies_stored") is False,
        "training_memory_blocked": catalog.get("training_eligible") is False and trace.get("training_eligible") is False and all(not bool(step.get("long_term_memory_write")) for step in trace.get("steps", [])),
    }
    return {"status": "passed" if all(checks.values()) else "blocked", "checks": checks, "blocking_reasons": [key for key, value in checks.items() if not value], "claim_allowed": False}


def run() -> dict[str, Any]:
    base = _load(PG35_SCRIPT, "pg91_pg35_collector_runtime")
    _patch_paths(base)
    result = base.main()
    if int(result or 0) != 0:
        raise RuntimeError(f"PG-35 collector returned {result}")
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    gate = _hard_gate(catalog, trace)
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg91-pg35-independent-collector-report-v1",
        "status": "completed_evaluation",
        "source": {
            "implementation": "standalone_python_http_fixture_v3",
            "collector_implementation": "pg35_independent_fixture_catalog_v1_reused",
            "fixture_source": "app/pg35_independent_fixture.py",
            "seed_set": list(SEEDS),
            "source_count": int(catalog.get("source_count", 0)),
            "loopback_only": True,
            "external_network": False,
            "fresh_reset_per_observation": True,
        },
        "metrics": {
            "sample_count": len(catalog.get("samples", [])),
            "typed_positive_count": int(catalog.get("typed_positive_count", 0)),
            "typed_negative_count": int(catalog.get("negative_control_count", 0)),
            "fresh_reset_count": int(catalog.get("fresh_reset_count", 0)),
            "target_instance_count": int(catalog.get("target_instance_count", 0)),
            "encoding_pair_count": int(catalog.get("encoding_pair_count", 0)),
            "source_count": int(catalog.get("source_count", 0)),
            "episode_count": int(catalog.get("trace_episode_count", 0)),
            "accepted_evaluation_episode_count": int(catalog.get("accepted_evaluation_episode_count", 0)),
            "methods": {"GET": sum(int(str(row.get("method", "")).upper() == "GET") for row in catalog.get("samples", [])), "POST": sum(int(str(row.get("method", "")).upper() == "POST") for row in catalog.get("samples", []))},
            "encodings": {"identity": sum(int(row.get("encoding") == "identity") for row in catalog.get("samples", [])), "url_percent": sum(int(row.get("encoding") == "url_percent") for row in catalog.get("samples", []))},
        },
        "hard_gate": gate,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "training_catalog_generated": False, "status": "evaluation_only_independent_implementation", "reason": "PG91 must pass frozen model replay and cross-implementation review before any promotion"},
        "artifacts": {"report": str(REPORT_PATH.relative_to(ROOT)), "protocol": str(PROTOCOL_PATH.relative_to(ROOT)), "catalog": str(CATALOG_PATH.relative_to(ROOT)), "trace": str(TRACE_PATH.relative_to(ROOT))},
    }
    catalog["schema_version"] = "pg91-pg35-independent-fixture-catalog-v1"
    catalog["catalog_id"] = "pg91-pg35-independent-fixture-evaluation-only"
    catalog["pg91_seed_set"] = list(SEEDS)
    catalog["evaluation_only"] = True
    catalog["training_eligible"] = False
    catalog["long_term_memory_write"] = False
    trace["schema_version"] = "pg91-pg35-independent-fixture-trace-v1"
    trace["pg91_seed_set"] = list(SEEDS)
    trace["evaluation_only"] = True
    trace["training_eligible"] = False
    trace["long_term_memory_write"] = False
    catalog["pg91_report_protocol_id"] = PROTOCOL_ID
    trace["pg91_report_protocol_id"] = PROTOCOL_ID
    protocol = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg91-pg35-independent-collector-protocol-v1",
        "implementation": "standalone_python_http_fixture_v3",
        "seed_set": list(SEEDS),
        "transport": ["GET", "POST"],
        "encoding_set": ["identity", "url_percent"],
        "typed_oracle_after_observation": True,
        "fresh_reset_per_observation": True,
        "positive_and_negative_authority_required": True,
        "evidence_sha256_required": True,
        "raw_persistence_forbidden": True,
        "run_result": {"hard_gate": gate, "training_allowed": False, "memory_promotion_allowed": False},
        "next_experiment": "PG91 frozen PG86 replay on this independent implementation",
    }
    CATALOG_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("# PG-91 PG-35 independent collector\n\n" + f"samples={report['metrics']['sample_count']}；positive={report['metrics']['typed_positive_count']}；negative={report['metrics']['typed_negative_count']}；GET/POST={report['metrics']['methods']}；encoding={report['metrics']['encodings']}。\n\n硬门：`{gate['status']}`；training/memory promotion=`false`。\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"protocol_id": PROTOCOL_ID, "status": result["hard_gate"]["status"], "sample_count": result["metrics"]["sample_count"], "typed_positive_count": result["metrics"]["typed_positive_count"], "typed_negative_count": result["metrics"]["typed_negative_count"], "training_allowed": False, "memory_promotion_allowed": False}, ensure_ascii=False, indent=2))
