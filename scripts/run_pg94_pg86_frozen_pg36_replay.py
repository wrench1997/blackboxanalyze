"""Replay frozen PG-86 on PG-94's independent PG-36 maze.

This is a third target implementation, not another training run.  The
adapter is deliberately strict: every response must carry the shared bounded
surface and geometry channels, and there is no PG-35 shape-delta fallback.
Typed oracle fields are used only after scoring for evaluation labels.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PG91_SCRIPT = ROOT / "scripts" / "run_pg91_pg86_frozen_pg35_replay.py"
PG86_SCRIPT = ROOT / "scripts" / "train_pg86_surface_signal_composite.py"
PG77_SCRIPT = ROOT / "scripts" / "run_pg77_real_triplet_transformer.py"
PG84_SCRIPT = ROOT / "scripts" / "run_pg84_cross_dataset_frozen_replay.py"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg86-surface-signal-composite-transformer" / "model.pt"
REFERENCE_DATASET_PATH = ROOT / "research" / "pg86_surface_signal_composite_trace_dataset_v1.json"
INPUT_TRACE_PATH = ROOT / "research" / "pg94_pg36_surface_trace_v1.json"
INPUT_CATALOG_PATH = ROOT / "research" / "pg94_pg36_surface_catalog_v1.json"
REPORT_PATH = ROOT / "research" / "pg94_pg86_frozen_pg36_replay_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg94_pg86_frozen_pg36_replay_protocol_v1.json"
TRACE_PATH = ROOT / "research" / "pg94_pg86_frozen_pg36_replay_trace_v1.json"
DATASET_PATH = ROOT / "research" / "pg94_pg86_frozen_pg36_replay_dataset_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg94_pg86_frozen_pg36_replay_report_v1.md"
PROTOCOL_ID = "pg-pk-94-pg86-frozen-pg36-replay-v1"
PROJECTION_SCHEMA = "canonical_effect_projection_v3_surface_signal"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run() -> dict[str, Any]:
    pg91 = _load(PG91_SCRIPT, "pg94_pg91_runtime")
    pg86 = _load(PG86_SCRIPT, "pg94_pg86_runtime")
    pg77 = _load(PG77_SCRIPT, "pg94_pg77_runtime")
    pg84 = _load(PG84_SCRIPT, "pg94_pg84_runtime")
    sha256_json = __import__("app.trace_aligned_dataset", fromlist=["sha256_json"]).sha256_json
    trace = json.loads(INPUT_TRACE_PATH.read_text(encoding="utf-8"))
    catalog = json.loads(INPUT_CATALOG_PATH.read_text(encoding="utf-8"))
    reference = json.loads(REFERENCE_DATASET_PATH.read_text(encoding="utf-8"))
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    original_adapter = pg91._adapter

    def strict_adapter(pg84_module: Any, projection: dict[str, Any], hash_fn: Any, *, baseline_key_count: int) -> dict[str, Any]:
        surface = projection.get("effect_surface")
        geometry = projection.get("effect_geometry")
        if projection.get("projection_schema") != PROJECTION_SCHEMA or not isinstance(surface, dict) or not isinstance(geometry, dict):
            raise RuntimeError("PG-94 replay refuses projections without the shared semantic channels")
        # This is the frozen shared encoder used by PG-86.  It intentionally
        # does not inspect PG-36 family names, oracle values or body shape
        # deltas, and it has no target-specific fallback.
        return original_adapter(pg84_module, projection, hash_fn, baseline_key_count=baseline_key_count)

    pg91._adapter = strict_adapter
    rows, metadata = pg91._build_cases(pg86, pg77, pg84, trace, sha256_json)
    reference_rows = [row for row in reference.get("rows", []) if row.get("split") == "train"]
    metrics, details = pg91._evaluate(rows, reference_rows, checkpoint, pg77)
    source_steps = list(trace.get("steps", []))
    candidate_steps = [step for step in source_steps if bool((step.get("oracle_projection") or {}).get("positive"))]
    methods = {str((step.get("action_manifest") or {}).get("method", "")).upper() for step in candidate_steps}
    source_targets = [str(step.get("target_instance_id", "")) for step in source_steps]
    semantic_steps = [
        step for step in source_steps
        if (step.get("response_projection") or {}).get("projection_schema") == PROJECTION_SCHEMA
        and isinstance((step.get("response_projection") or {}).get("effect_surface"), dict)
        and isinstance((step.get("response_projection") or {}).get("effect_geometry"), dict)
    ]
    checks = {
        "independent_pg36_collection_passed": catalog.get("independent_target_implementation") is True and len(source_steps) == 960,
        "third_implementation": catalog.get("collector_profile", "").startswith("pg36_original_maze_runner"),
        "triplet_replay_complete": metadata["positive_case_count"] == 96 and metadata["negative_case_count"] == 96 and len(rows) == 192,
        "typed_oracle_counts": metrics["typed_positive_count"] == 96 and metrics["typed_negative_count"] == 96,
        "fresh_source_targets": len(source_targets) == len(set(source_targets)) == 960 and all(bool(step.get("fresh_reset", {}).get("fresh_target")) for step in source_steps),
        "semantic_projection_complete": len(semantic_steps) == len(source_steps) and catalog.get("projection_repair_post_hoc") is False,
        "get_post_covered": methods == {"GET", "POST"},
        "independent_implementations": int(catalog.get("source_count", 0)) == 2,
        "unknown_token_count_zero": metrics["unknown_token_count"] == 0 and metrics["reference_unknown_token_count"] == 0,
        "false_accept_zero": metrics["false_accept_count"] == 0,
        "known_recall_min": metrics["confirm_recall"] >= 0.80,
        "cross_seed_recall_min": metrics["seed_min_confirm_recall"] >= 0.75,
        "family_recall_min": metrics["family_min_confirm_recall"] >= 0.50,
        "not_all_abstain": metrics["confirm_recall"] > 0.0,
        "raw_free": all(not row["raw_probe_stored"] and not row["raw_response_stored"] for row in rows),
    }
    status = "passed" if all(checks.values()) else "blocked"
    input_hash = hashlib.sha256(INPUT_TRACE_PATH.read_bytes()).hexdigest()
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg94-pg86-frozen-pg36-replay-report-v1",
        "status": "completed_evaluation",
        "source": {
            "frozen_checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
            "input_trace": str(INPUT_TRACE_PATH.relative_to(ROOT)),
            "input_trace_sha256": input_hash,
            "input_catalog": str(INPUT_CATALOG_PATH.relative_to(ROOT)),
            "reference_dataset": str(REFERENCE_DATASET_PATH.relative_to(ROOT)),
            "independent_implementation": "pg36_independent_maze_two_layouts_four_phases",
            "adapter_profile": "canonical_effect_projection_v3_surface_signal_shared_encoder",
            "post_hoc_schema_alignment": False,
            "target_specific_shape_delta_fallback": False,
            "device": metrics["device"],
            "training": False,
            "memory_write": False,
            "oracle_after_target_only": True,
        },
        "dataset": {
            "source_step_count": len(source_steps),
            "replay_case_count": metadata["positive_case_count"],
            "row_count": len(rows),
            "get_post_counts": {
                "GET": sum(int(str((step.get("action_manifest") or {}).get("method", "")).upper() == "GET") for step in candidate_steps),
                "POST": sum(int(str((step.get("action_manifest") or {}).get("method", "")).upper() == "POST") for step in candidate_steps),
            },
            "family_set": metadata["family_set"],
            "seed_set": sorted({int(step.get("sampling_seed", -1)) for step in candidate_steps}),
        },
        "metrics": metrics,
        "details": details,
        "capability_gate": {"status": status, "checks": checks, "blocking_reasons": [key for key, value in checks.items() if not value], "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "status": "third_implementation_frozen_evaluation_only", "reason": "PG-94 is a no-training third-implementation replay; its result cannot promote a checkpoint or memory"},
        "artifacts": {"report": str(REPORT_PATH.relative_to(ROOT)), "protocol": str(PROTOCOL_PATH.relative_to(ROOT)), "trace": str(TRACE_PATH.relative_to(ROOT)), "dataset": str(DATASET_PATH.relative_to(ROOT))},
    }
    dataset = {"schema_version": "pg94-pg86-frozen-pg36-replay-dataset-v1", "dataset_id": "pg94-pg86-frozen-pg36", "evaluation_only": True, "training_eligible": False, "rows": rows, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "long_term_memory_write": False}
    trace_out = {"schema_version": "pg94-pg86-frozen-pg36-replay-trace-v1", "protocol_id": PROTOCOL_ID, "input_trace_sha256": input_hash, "evaluation_only": True, "rows": [{"trace_id": item["trace_id"], "seed": item["seed"], "family": item["family"], "role": item["role"], "expected": item["expected"], "decision": item["decision"], "raw_probe_stored": False, "raw_response_stored": False} for item in details], "online_weight_update": False, "long_term_memory_write": False}
    protocol = {"protocol_id": PROTOCOL_ID, "schema_version": "pg94-pg86-frozen-pg36-replay-protocol-v1", "frozen_checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "input_trace": str(INPUT_TRACE_PATH.relative_to(ROOT)), "input_trace_sha256": input_hash, "independent_implementation": "pg36_independent_maze_two_layouts_four_phases", "projection_schema": PROJECTION_SCHEMA, "post_hoc_schema_alignment": False, "target_specific_shape_delta_fallback": False, "oracle_after_target_only": True, "raw_persistence_forbidden": True, "run_result": {"capability_gate": report["capability_gate"], "training_allowed": False, "memory_promotion_allowed": False}, "next_experiment": "PG95 independent implementation review and intern handoff gate"}
    DATASET_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps(trace_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("# PG-94 frozen PG-86 replay on PG-36\n\n" + f"rows={len(rows)}；recall={metrics['confirm_recall']}；seed_min={metrics['seed_min_confirm_recall']}；family_min={metrics['family_min_confirm_recall']}；false_accept={metrics['false_accept_count']}；unknown_tokens={metrics['unknown_token_count']}。\n\n能力门：`{status}`；post-hoc adapter=`false`；training/memory promotion=`false`。\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"protocol_id": PROTOCOL_ID, "status": result["capability_gate"]["status"], "confirm_recall": result["metrics"]["confirm_recall"], "seed_min_confirm_recall": result["metrics"]["seed_min_confirm_recall"], "family_min_confirm_recall": result["metrics"]["family_min_confirm_recall"], "false_accept_count": result["metrics"]["false_accept_count"], "unknown_token_count": result["metrics"]["unknown_token_count"], "device": result["metrics"]["device"], "training_allowed": False}, ensure_ascii=False, indent=2))
