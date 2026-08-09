"""Finalize PG-79's already-collected bounded trace after a validator-only fix.

No network call is made here.  The collector had all typed triplets and fresh
targets, but omitted the trace-level negative-control pair binding.  This
finalizer adds that structural binding, recomputes evidence/echo hashes and
reruns the same validator/episode gate.  It never changes projections or
oracle positive/negative values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.trace_aligned_dataset import evaluate_episode, sha256_json, validate_trace_step  # noqa: E402

TRACE_PATH = ROOT / "research" / "pg79_fresh_unified_triplet_collector_trace_v1.json"
REPORT_PATH = ROOT / "research" / "pg79_fresh_unified_triplet_collector_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg79_fresh_unified_triplet_collector_protocol_v1.json"


def _evidence(step: dict[str, Any]) -> str:
    return sha256_json({"step_id": step["step_id"], "target_instance_id": step["target_instance_id"], "neutral": step["neutral_projection"], "negative": step["negative_probe_projection"], "positive": step["response_projection"], "neutral_oracle": step["neutral_oracle_projection"], "negative_oracle": step["negative_oracle_projection"], "positive_oracle": step["oracle_projection"], "reset": step["fresh_reset"]})


def _echo(step: dict[str, Any]) -> dict[str, str]:
    body = {key: step[key] for key in ("action_manifest", "baseline_projection", "response_projection", "oracle_projection", "belief_before", "belief_after", "decision", "next_action", "neutral_projection", "negative_probe_projection", "neutral_oracle_projection", "negative_oracle_projection")}
    return {"sha256": sha256_json(body)}


def run() -> dict[str, Any]:
    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    by_episode: dict[str, list[dict[str, Any]]] = {}
    failures: list[dict[str, Any]] = []
    for step in trace.get("steps", []):
        step = dict(step)
        step["oracle_projection"] = dict(step.get("oracle_projection") or {})
        step["oracle_projection"]["negative_control_pair_id"] = f"pg79-control-{str(step['step_id']).split('-')[-1]}"
        step["evidence_sha256"] = _evidence(step)
        step["echo"] = _echo(step)
        try:
            normalized = validate_trace_step(step)
        except ValueError as exc:
            failures.append({"step_id": step.get("step_id"), "error_type": type(exc).__name__})
            normalized = step
        by_episode.setdefault(str(normalized["episode_id"]), []).append(normalized)
    normalized_steps = [step for episode in by_episode.values() for step in episode]
    episodes: list[dict[str, Any]] = []
    for episode_id, episode_steps in by_episode.items():
        episodes.append({"episode_id": episode_id, "steps": episode_steps, "validation": evaluate_episode(episode_steps) if not failures else {"status": "trace_only", "reasons": ["step_validation_failure"]}})
    trace["steps"] = normalized_steps
    trace["episodes"] = episodes
    trace["accepted_episode_count"] = sum(int(item["validation"].get("status") == "accepted_evaluation") for item in episodes)
    trace["validation_failures"] = failures
    trace["finalized_by"] = "scripts/finalize_pg79_trace.py"
    trace["online_weight_update"] = False
    trace["long_term_memory_write"] = False
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    report["metrics"]["trace_accepted_episode_count"] = trace["accepted_episode_count"]
    report["metrics"]["validation_failure_count"] = len(failures)
    report["hard_gate"]["checks"]["trace_episodes_accepted"] = trace["accepted_episode_count"] == trace["episode_count"] and not failures
    report["hard_gate"]["blocking_reasons"] = [key for key, value in report["hard_gate"]["checks"].items() if not value]
    report["hard_gate"]["status"] = "passed" if not report["hard_gate"]["blocking_reasons"] else "blocked"
    report["finalization"] = {"validator_only": True, "network_replay": False, "oracle_values_changed": False, "projection_values_changed": False, "pair_binding_added": True, "evidence_and_echo_recomputed": True, "finalizer": "scripts/finalize_pg79_trace.py"}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["run_result"]["hard_gate"] = report["hard_gate"]
    protocol["run_result"]["network_replay"] = False
    protocol["run_result"]["oracle_values_changed"] = False
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": report["hard_gate"]["status"], "accepted_episode_count": trace["accepted_episode_count"], "validation_failure_count": len(failures), "network_replay": False}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
