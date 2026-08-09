"""Evaluate the PG-95 invariant candidate on frozen unseen PG-94 data.

The case builder fixes one proven replay bug: a positive confirm step is paired
with the negative control from the same method, encoding and phase, never with
an unrelated timeout/error response.  The PG-94 source itself remains frozen;
no trace row is used for PG-95 training.
"""

from __future__ import annotations

import collections
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
CHECKPOINT_PATH = ROOT / "artifacts" / "pg95-invariant-surface-transformer" / "model.pt"
REFERENCE_DATASET_PATH = ROOT / "research" / "pg95_invariant_surface_trace_dataset_v1.json"
INPUT_TRACE_PATH = ROOT / "research" / "pg94_pg36_surface_trace_v1.json"
REPORT_PATH = ROOT / "research" / "pg95_pg94_frozen_replay_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg95_pg94_frozen_replay_protocol_v1.json"
TRACE_PATH = ROOT / "research" / "pg95_pg94_frozen_replay_trace_v1.json"
DATASET_PATH = ROOT / "research" / "pg95_pg94_frozen_replay_dataset_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg95_pg94_frozen_replay_report_v1.md"
PROTOCOL_ID = "pg-pk-95-pg94-frozen-replay-v1"
PROJECTION_SCHEMA = "canonical_effect_projection_v3_surface_signal"

from app.invariant_token_funnel import SCHEMA_VERSION, canonicalize_tokens  # noqa: E402


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase(step: dict[str, Any]) -> str:
    probe_ref = str((step.get("action_manifest") or {}).get("probe_ref", ""))
    value = probe_ref.rsplit("-", 1)[-1]
    if value not in {"screen", "confirm", "error", "timeout"}:
        raise RuntimeError(f"PG-94 step lacks an allow-listed phase: {probe_ref}")
    return value


def _build_phase_aligned_cases(pg86: Any, pg77: Any, pg84: Any, trace: dict[str, Any], hash_fn: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for step in trace.get("steps", []):
        grouped[str(step["episode_id"])].append(step)
    cases: list[dict[str, Any]] = []
    for episode_id, steps in sorted(grouped.items()):
        controls: dict[tuple[str, tuple[str, ...], str], dict[str, Any]] = {}
        candidates: list[tuple[dict[str, Any], str]] = []
        for step in steps:
            action = dict(step.get("action_manifest") or {})
            method = str(action.get("method", "GET")).upper()
            encoding = tuple(str(value) for value in (action.get("encoding_chain") or ["identity"]))
            key = (method, encoding, _phase(step))
            if bool((step.get("oracle_projection") or {}).get("positive")):
                candidates.append((step, key[2]))
            else:
                controls[key] = step
        for candidate, phase in candidates:
            action = dict(candidate.get("action_manifest") or {})
            method = str(action.get("method", "GET")).upper()
            encoding = tuple(str(value) for value in (action.get("encoding_chain") or ["identity"]))
            neutral_step = controls.get((method, encoding, phase))
            if neutral_step is None:
                raise RuntimeError(f"missing same-phase PG-94 control for {candidate['step_id']}")
            neutral_raw = dict(neutral_step.get("response_projection") or {})
            if neutral_raw.get("projection_schema") != PROJECTION_SCHEMA or (candidate.get("response_projection") or {}).get("projection_schema") != PROJECTION_SCHEMA:
                raise RuntimeError("PG-94 replay requires the shared semantic projection")
            baseline_shape = neutral_raw.get("shape") if isinstance(neutral_raw.get("shape"), dict) else {}
            baseline_key_count = int(baseline_shape.get("key_count") or 0)
            neutral_projection = pg86_adapter(pg84, neutral_raw, hash_fn, baseline_key_count)
            candidate_projection = pg86_adapter(pg84, dict(candidate.get("response_projection") or {}), hash_fn, baseline_key_count)
            fake_step = {"action_manifest": action, "neutral_projection": neutral_projection, "negative_probe_projection": neutral_projection}
            for role, projection, oracle in (("positive", candidate_projection, dict(candidate.get("oracle_projection") or {})), ("negative", neutral_projection, dict(neutral_step.get("oracle_projection") or {}))):
                tokens, oracle_index = pg86._row_tokens(pg77, fake_step, projection, oracle)
                tokens = canonicalize_tokens(tokens)
                cases.append({"trace_id": f"{candidate['step_id']}-{role}", "episode_id": episode_id, "step_id": str(candidate["step_id"]), "seed": int(candidate.get("sampling_seed", 0)), "family": str(candidate.get("hypothesis", "unknown")), "surface": str(action.get("route_template_id", "unknown")), "method": method, "encoding": list(encoding), "role": role, "target_instance_id": str(candidate.get("target_instance_id", "")), "tokens": tokens, "oracle_index": oracle_index, "expected": "confirm" if bool(oracle.get("positive")) else "reject", "raw_probe_stored": False, "raw_response_stored": False})
    return cases, {"episode_count": len(grouped), "positive_case_count": sum(int(row["role"] == "positive") for row in cases), "negative_case_count": sum(int(row["role"] == "negative") for row in cases), "family_set": sorted({row["family"] for row in cases}), "method_set": sorted({row["method"] for row in cases})}


def pg86_adapter(pg84: Any, projection: dict[str, Any], hash_fn: Any, baseline_key_count: int) -> dict[str, Any]:
    if projection.get("projection_schema") != PROJECTION_SCHEMA:
        raise RuntimeError("projection schema mismatch")
    if not isinstance(projection.get("effect_surface"), dict) or not isinstance(projection.get("effect_geometry"), dict):
        raise RuntimeError("semantic projection is incomplete")
    # The fixed shared encoder is the same one used by PG-86; there is no
    # source-specific shape-delta reconstruction.
    adapter = _load(PG91_SCRIPT, "pg95_pg91_adapter_runtime")
    return adapter._adapter(pg84, projection, hash_fn, baseline_key_count=baseline_key_count)


def run() -> dict[str, Any]:
    pg91 = _load(PG91_SCRIPT, "pg95_pg91_eval_runtime")
    pg86 = _load(PG86_SCRIPT, "pg95_pg86_eval_runtime")
    pg77 = _load(PG77_SCRIPT, "pg95_pg77_eval_runtime")
    pg84 = _load(PG84_SCRIPT, "pg95_pg84_eval_runtime")
    hash_fn = __import__("app.trace_aligned_dataset", fromlist=["sha256_json"]).sha256_json
    trace = json.loads(INPUT_TRACE_PATH.read_text(encoding="utf-8"))
    reference = json.loads(REFERENCE_DATASET_PATH.read_text(encoding="utf-8"))
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    rows, metadata = _build_phase_aligned_cases(pg86, pg77, pg84, trace, hash_fn)
    reference_rows = [row for row in reference.get("rows", []) if row.get("split") == "train"]
    metrics, details = pg91._evaluate(rows, reference_rows, checkpoint, pg77)
    checks = {
        "phase_aligned_case_count": metadata["positive_case_count"] == 96 and metadata["negative_case_count"] == 96 and len(rows) == 192,
        "typed_counts": metrics["typed_positive_count"] == 96 and metrics["typed_negative_count"] == 96,
        "get_post_covered": metadata["method_set"] == ["GET", "POST"],
        "unknown_token_zero": metrics["unknown_token_count"] == 0 and metrics["reference_unknown_token_count"] == 0,
        "false_accept_zero": metrics["false_accept_count"] == 0,
        "known_recall_min": metrics["confirm_recall"] >= 0.80,
        "seed_recall_min": metrics["seed_min_confirm_recall"] >= 0.75,
        "family_recall_min": metrics["family_min_confirm_recall"] >= 0.50,
        "not_all_abstain": metrics["confirm_recall"] > 0.0,
        "raw_free": all(not row["raw_probe_stored"] and not row["raw_response_stored"] for row in rows),
    }
    status = "passed" if all(checks.values()) else "blocked"
    input_hash = hashlib.sha256(INPUT_TRACE_PATH.read_bytes()).hexdigest()
    report = {"protocol_id": PROTOCOL_ID, "schema_version": "pg95-pg94-frozen-replay-report-v1", "status": "completed_evaluation", "source": {"frozen_checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "input_trace": str(INPUT_TRACE_PATH.relative_to(ROOT)), "input_trace_sha256": input_hash, "reference_dataset": str(REFERENCE_DATASET_PATH.relative_to(ROOT)), "token_funnel_schema": SCHEMA_VERSION, "phase_aligned_controls": True, "training": False, "memory_write": False, "oracle_after_target_only": True, "device": metrics["device"]}, "dataset": {"source_step_count": len(trace.get("steps", [])), "replay_case_count": metadata["positive_case_count"], "row_count": len(rows), "get_post_counts": {"GET": sum(int(row["method"] == "GET" and row["role"] == "positive") for row in rows), "POST": sum(int(row["method"] == "POST" and row["role"] == "positive") for row in rows)}, "family_set": metadata["family_set"]}, "metrics": metrics, "details": details, "capability_gate": {"status": status, "checks": checks, "blocking_reasons": [key for key, value in checks.items() if not value], "claim_allowed": False}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "status": "candidate_frozen_pg94_evaluation_only", "reason": "PG95 is an offline candidate and PG94 is held out; no active checkpoint or memory update is allowed"}, "artifacts": {"report": str(REPORT_PATH.relative_to(ROOT)), "protocol": str(PROTOCOL_PATH.relative_to(ROOT)), "trace": str(TRACE_PATH.relative_to(ROOT)), "dataset": str(DATASET_PATH.relative_to(ROOT))}}
    dataset = {"schema_version": "pg95-pg94-frozen-replay-dataset-v1", "dataset_id": "pg95-pg94-frozen", "evaluation_only": True, "training_eligible": False, "rows": rows, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "long_term_memory_write": False}
    out_trace = {"schema_version": "pg95-pg94-frozen-replay-trace-v1", "protocol_id": PROTOCOL_ID, "input_trace_sha256": input_hash, "evaluation_only": True, "rows": [{"trace_id": item["trace_id"], "expected": item["expected"], "decision": item["decision"], "family": item["family"], "role": item["role"], "raw_probe_stored": False, "raw_response_stored": False} for item in details], "online_weight_update": False, "long_term_memory_write": False}
    protocol = {"protocol_id": PROTOCOL_ID, "schema_version": "pg95-pg94-frozen-replay-protocol-v1", "frozen_checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "input_trace": str(INPUT_TRACE_PATH.relative_to(ROOT)), "input_trace_sha256": input_hash, "token_funnel_schema": SCHEMA_VERSION, "phase_aligned_controls": True, "pg94_excluded_from_training": True, "oracle_after_target_only": True, "raw_persistence_forbidden": True, "run_result": {"capability_gate": report["capability_gate"], "training_allowed": False, "memory_promotion_allowed": False}, "next_experiment": "PG96 representation review before any promotion"}
    DATASET_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps(out_trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("# PG-95 PG-94 frozen replay\n\n" + f"rows={len(rows)}；recall={metrics['confirm_recall']}；seed_min={metrics['seed_min_confirm_recall']}；family_min={metrics['family_min_confirm_recall']}；false_accept={metrics['false_accept_count']}；unknown_tokens={metrics['unknown_token_count']}；phase_aligned=true。\n\n能力门：`{status}`；training/memory promotion=`false`。\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"protocol_id": PROTOCOL_ID, "status": result["capability_gate"]["status"], "confirm_recall": result["metrics"]["confirm_recall"], "seed_min_confirm_recall": result["metrics"]["seed_min_confirm_recall"], "family_min_confirm_recall": result["metrics"]["family_min_confirm_recall"], "false_accept_count": result["metrics"]["false_accept_count"], "unknown_token_count": result["metrics"]["unknown_token_count"], "device": result["metrics"]["device"], "training_allowed": False}, ensure_ascii=False, indent=2))
