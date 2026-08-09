"""PG-92: blind third-source replay using the quarantined PG-91 adapter.

No PG-34 rows are used for training.  The purpose is to test whether the
PG-35 shape-delta compatibility profile transfers to a different fixture
whose positive/control responses have the same coarse JSON shape.
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

PG91_SCRIPT = ROOT / "scripts" / "run_pg91_pg86_frozen_pg35_replay.py"
PG86_SCRIPT = ROOT / "scripts" / "train_pg86_surface_signal_composite.py"
PG77_SCRIPT = ROOT / "scripts" / "run_pg77_real_triplet_transformer.py"
PG84_SCRIPT = ROOT / "scripts" / "run_pg84_cross_dataset_frozen_replay.py"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg86-surface-signal-composite-transformer" / "model.pt"
REFERENCE_DATASET_PATH = ROOT / "research" / "pg86_surface_signal_composite_trace_dataset_v1.json"
INPUT_TRACE_PATH = ROOT / "research" / "pg34_independent_fixture_trace_v1.json"
REPORT_PATH = ROOT / "research" / "pg92_blind_pg34_frozen_replay_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg92_blind_pg34_frozen_replay_protocol_v1.json"
TRACE_PATH = ROOT / "research" / "pg92_blind_pg34_frozen_replay_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg92_blind_pg34_frozen_replay_report_v1.md"
PROTOCOL_ID = "pg-pk-92-blind-pg34-frozen-replay-v1"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run() -> dict[str, Any]:
    pg91 = _load(PG91_SCRIPT, "pg92_pg91_runtime")
    pg86 = _load(PG86_SCRIPT, "pg92_pg86_runtime")
    pg77 = _load(PG77_SCRIPT, "pg92_pg77_runtime")
    pg84 = _load(PG84_SCRIPT, "pg92_pg84_runtime")
    sha256_json = __import__("app.trace_aligned_dataset", fromlist=["sha256_json"]).sha256_json
    trace = json.loads(INPUT_TRACE_PATH.read_text(encoding="utf-8"))
    reference = json.loads(REFERENCE_DATASET_PATH.read_text(encoding="utf-8"))
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    rows, metadata = pg91._build_cases(pg86, pg77, pg84, trace, sha256_json)
    reference_rows = [row for row in reference.get("rows", []) if row.get("split") == "train"]
    metrics, details = pg91._evaluate(rows, reference_rows, checkpoint, pg77)
    source_ids = [str(step.get("target_instance_id", "")) for step in trace.get("steps", [])]
    candidate_steps = [step for step in trace.get("steps", []) if bool((step.get("oracle_projection") or {}).get("positive"))]
    methods = {str((step.get("action_manifest") or {}).get("method", "")).upper() for step in candidate_steps}
    checks = {
        "independent_pg34_trace": trace.get("independent_target_implementation") is True and len(trace.get("steps", [])) == 108,
        "triplet_replay_complete": metadata["positive_case_count"] == 48 and metadata["negative_case_count"] == 48 and len(rows) == 96,
        "typed_counts": metrics["typed_positive_count"] == 48 and metrics["typed_negative_count"] == 48,
        "fresh_source_targets": len(source_ids) == len(set(source_ids)) == 108 and all(bool(step.get("fresh_reset", {}).get("fresh_target")) for step in trace.get("steps", [])),
        "get_post_covered": methods == {"GET", "POST"},
        "unknown_token_zero": metrics["unknown_token_count"] == 0 and metrics["reference_unknown_token_count"] == 0,
        "false_accept_zero": metrics["false_accept_count"] == 0,
        "not_all_abstain": metrics["confirm_recall"] > 0.0,
        "raw_free": all(not row["raw_probe_stored"] and not row["raw_response_stored"] for row in rows),
    }
    status = "passed" if all(checks.values()) else "blocked"
    input_hash = __import__("hashlib").sha256(INPUT_TRACE_PATH.read_bytes()).hexdigest()
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg92-blind-pg34-frozen-replay-report-v1",
        "status": "completed_evaluation",
        "source": {"frozen_checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "input_trace": str(INPUT_TRACE_PATH.relative_to(ROOT)), "input_trace_sha256": input_hash, "reference_dataset": str(REFERENCE_DATASET_PATH.relative_to(ROOT)), "adapter_profile_reused": "canonical_effect_projection_v4_pg35_shape_delta", "training": False, "memory_write": False, "oracle_after_target_only": True, "device": metrics["device"]},
        "dataset": {"source_step_count": len(trace.get("steps", [])), "replay_case_count": metadata["positive_case_count"], "row_count": len(rows), "get_post_counts": {"GET": sum(int(str((step.get("action_manifest") or {}).get("method", "")).upper() == "GET") for step in candidate_steps), "POST": sum(int(str((step.get("action_manifest") or {}).get("method", "")).upper() == "POST") for step in candidate_steps)}, "family_set": metadata["family_set"]},
        "metrics": metrics,
        "details": details,
        "capability_gate": {"status": status, "checks": checks, "blocking_reasons": [key for key, value in checks.items() if not value], "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "status": "blind_third_source_evaluation_only", "reason": "PG92 is a blind third-source replay; a blocked result is retained as a representation boundary"},
        "artifacts": {"report": str(REPORT_PATH.relative_to(ROOT)), "protocol": str(PROTOCOL_PATH.relative_to(ROOT)), "trace": str(TRACE_PATH.relative_to(ROOT))},
    }
    out_trace = {"schema_version": "pg92-blind-pg34-frozen-replay-trace-v1", "protocol_id": PROTOCOL_ID, "input_trace_sha256": input_hash, "evaluation_only": True, "rows": [{"trace_id": item["trace_id"], "expected": item["expected"], "decision": item["decision"], "family": item["family"], "role": item["role"], "raw_probe_stored": False, "raw_response_stored": False} for item in details], "online_weight_update": False, "long_term_memory_write": False}
    protocol = {"protocol_id": PROTOCOL_ID, "schema_version": "pg92-blind-pg34-frozen-replay-protocol-v1", "frozen_checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "input_trace": str(INPUT_TRACE_PATH.relative_to(ROOT)), "input_trace_sha256": input_hash, "adapter_profile_reused": "canonical_effect_projection_v4_pg35_shape_delta", "oracle_after_target_only": True, "raw_persistence_forbidden": True, "run_result": {"capability_gate": report["capability_gate"], "training_allowed": False, "memory_promotion_allowed": False}, "next_experiment": "PG93 representation-invariant adapter or independent third application"}
    TRACE_PATH.write_text(json.dumps(out_trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("# PG-92 blind PG-34 frozen replay\n\n" + f"rows={len(rows)}；recall={metrics['confirm_recall']}；false_accept={metrics['false_accept_count']}；unknown_tokens={metrics['unknown_token_count']}；adapter=`canonical_effect_projection_v4_pg35_shape_delta`。\n\n硬门：`{status}`；training/memory promotion=`false`。\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"protocol_id": PROTOCOL_ID, "status": result["capability_gate"]["status"], "confirm_recall": result["metrics"]["confirm_recall"], "false_accept_count": result["metrics"]["false_accept_count"], "unknown_token_count": result["metrics"]["unknown_token_count"], "device": result["metrics"]["device"], "training_allowed": False}, ensure_ascii=False, indent=2))
