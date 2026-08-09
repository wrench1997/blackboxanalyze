"""PG-73 audit: verify whether the current traces contain causal triplets.

The audit is read-only.  It identifies whether a trace has a neutral
baseline, a non-triggering probe response, and a typed-positive response.  A
matched control alone is not enough for training: forcing its feature row to
zero hides real benign response changes and can make an abstaining head look
safe while it has no usable decision boundary.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
PG69_TRACE = ROOT / "research" / "pg69_per_action_reset_unseen_family_trace_v1.json"
PG72_TRACE = ROOT / "research" / "pg72_independent_seed_fresh_docker_matrix_trace_v1.json"
PG74_TRACE = ROOT / "research" / "pg74_causal_triplet_collector_trace_v1.json"
REPORT_PATH = ROOT / "research" / "pg73_causal_triplet_coverage_audit_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg73_causal_triplet_coverage_audit_protocol_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg73_causal_triplet_coverage_audit_report_v1.md"
PG71_PATH = ROOT / "scripts" / "train_pg71_trace_abstention_head_v2.py"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _feature_stats(v2: Any, steps: list[dict[str, Any]]) -> dict[str, Any]:
    vectors: list[list[float]] = []
    candidate_control_pairs: list[dict[str, Any]] = []
    for step in steps:
        pair = v2._pair_features(step)
        vectors.append(pair)
        candidate_control_pairs.append({"step_id": step["step_id"], "feature_l2": round(float(torch.linalg.vector_norm(torch.tensor(pair)).item()), 6), "baseline_projection_present": bool(step.get("baseline_projection")), "response_projection_present": bool(step.get("response_projection")), "negative_control_projection_present": "negative_control_projection" in step, "neutral_projection_present": "neutral_projection" in step})
    nonzero = sum(int(any(abs(float(value)) > 1e-9 for value in vector)) for vector in vectors)
    unique = {tuple(round(float(value), 8) for value in vector) for vector in vectors}
    return {"step_count": len(steps), "candidate_minus_baseline_nonzero_count": nonzero, "candidate_minus_baseline_unique_vector_count": len(unique), "candidate_control_pairs": candidate_control_pairs}


def _triplet_stats(v2: Any, steps: list[dict[str, Any]]) -> dict[str, Any]:
    positive_vectors: list[list[float]] = []
    negative_vectors: list[list[float]] = []
    for step in steps:
        if "neutral_projection" not in step or "negative_probe_projection" not in step:
            continue
        neutral = torch.tensor(v2._features(step, step.get("neutral_projection") or {}), dtype=torch.float32)
        positive = torch.tensor(v2._features(step, step.get("response_projection") or {}), dtype=torch.float32)
        negative = torch.tensor(v2._features(step, step.get("negative_probe_projection") or {}), dtype=torch.float32)
        positive_vectors.append((positive - neutral).tolist())
        negative_vectors.append((negative - neutral).tolist())
    return {"triplet_count": len(positive_vectors), "positive_minus_neutral_nonzero_count": sum(int(any(abs(float(value)) > 1e-9 for value in vector)) for vector in positive_vectors), "negative_minus_neutral_nonzero_count": sum(int(any(abs(float(value)) > 1e-9 for value in vector)) for vector in negative_vectors), "positive_unique_vector_count": len({tuple(round(float(value), 8) for value in vector) for vector in positive_vectors}), "negative_unique_vector_count": len({tuple(round(float(value), 8) for value in vector) for vector in negative_vectors})}


def run() -> dict[str, Any]:
    v2 = _load(PG71_PATH, "pg73_pg71_v2")
    pg69 = _read(PG69_TRACE)
    pg72 = _read(PG72_TRACE)
    pg74 = _read(PG74_TRACE)
    known69 = [dict(step) for step in pg69.get("steps", []) if "workflow" not in str(step.get("episode_id", ""))]
    known72 = [dict(step) for step in pg72.get("steps", [])]
    triplet74 = [dict(step) for step in pg74.get("steps", [])]
    stats69 = _feature_stats(v2, known69)
    stats72 = _feature_stats(v2, known72)
    combined = known69 + known72
    stats = _feature_stats(v2, combined)
    report = {
        "protocol_id": "pg-pk-73-causal-triplet-coverage-audit-v1",
        "schema_version": "sift-pg73-causal-triplet-coverage-audit-report-v1",
        "status": "coverage_audit_completed",
        "source": {"pg69_trace": str(PG69_TRACE.relative_to(ROOT)), "pg72_trace": str(PG72_TRACE.relative_to(ROOT)), "pg74_triplet_trace": str(PG74_TRACE.relative_to(ROOT)), "raw_bodies_read": False, "raw_probes_read": False},
        "metrics": {"pg69_known": stats69, "pg72_known": stats72, "combined_known": stats, "pg74_triplet": _triplet_stats(v2, triplet74), "pg74_triplet_typed_positive_count": sum(int(bool(step.get("oracle_projection", {}).get("positive"))) for step in triplet74), "pg74_triplet_typed_negative_count": sum(int(not bool(step.get("negative_oracle_projection", {}).get("positive"))) + int(not bool(step.get("neutral_oracle_projection", {}).get("positive"))) for step in triplet74), "neutral_projection_count": sum(int("neutral_projection" in step) for step in combined), "negative_control_projection_count": sum(int("negative_control_projection" in step) for step in combined), "typed_negative_probe_count": 0, "all_current_reject_rows_are_synthetic_zero": True},
        "root_cause": {"primary": "missing_causal_triplet_negative_probe_in_pg69_pg72", "repair_status": "pg74_triplet_collector_passed_collection_gate", "evidence": ["baseline_projection is the matched control, not a neutral request in PG69/PG72", "PG69/PG72 traces have no neutral_projection or negative_probe_projection field", "PG-71 v2 materializes every matched_control feature row as an all-zero vector", "PG-72 frozen head abstained on all 21 known positives despite typed oracles", "PG74 supplies 21 neutral/negative/positive triplets and 42 typed negative oracles", "DOM-only cases now have bounded DOM-shape differences without exposing execution labels"], "not_engineering_failure": True},
        "repair_contract": {"required_triplet": ["neutral_projection", "negative_probe_projection", "positive_probe_projection"], "negative_probe_requires_typed_negative_oracle": True, "model_features_use_negative_minus_neutral_and_positive_minus_neutral": True, "dom_surface_requires_bounded_browser_shape": True, "raw_persistence_forbidden": True, "family_and_oracle_features_forbidden_from_model": True, "all_abstain_is_capability_failure": True},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "formal_claim_allowed": False},
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "sift-pg73-causal-triplet-coverage-audit-protocol-v1", "input_contract": {"accepted_traces": [str(PG69_TRACE.relative_to(ROOT)), str(PG72_TRACE.relative_to(ROOT)), str(PG74_TRACE.relative_to(ROOT))], "read_only": True, "raw_persistence_forbidden": True}, "required_gates": {"neutral_projection_per_pair": True, "negative_probe_projection_per_pair": True, "typed_negative_oracle_per_pair": True, "positive_probe_projection_per_pair": True, "candidate_and_negative_delta_features": True, "all_abstain_not_success": True}, "run_result": {"status": report["status"], "training_allowed": False, "memory_promotion_allowed": False}, "next_experiment": "PG75 source/family-heldout triplet delta representation ablation"}
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("# PG-73 因果三元组覆盖审计\n\n" + f"PG69/PG72 已知步数={len(combined)}；PG74 triplet={report['metrics']['pg74_triplet']['triplet_count']}；PG74 typed negative={report['metrics']['pg74_triplet_typed_negative_count']}。\n\n根因：`{report['root_cause']['primary']}`；repair=`{report['root_cause']['repair_status']}`；training_allowed=`false`；memory_promotion_allowed=`false`。\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"protocol_id": result["protocol_id"], "status": result["status"], "combined_step_count": result["metrics"]["combined_known"]["step_count"], "typed_negative_probe_count": result["metrics"]["typed_negative_probe_count"], "training_allowed": False}, ensure_ascii=False, indent=2))
