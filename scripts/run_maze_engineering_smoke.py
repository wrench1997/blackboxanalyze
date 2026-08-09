"""One-command, no-network smoke run for the maze engineering pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.detection_payload import build_detection_payload
from app.dom_oracle import run_dom_oracle
from app.juice_shop_adapter import agent_observation
from app.maze_engine import MazeRunRecorder
from app.maze_solver import assess_rule_exit
from app.sql_ast_oracle import run_sql_ast_oracle


ROOT = Path(__file__).resolve().parents[1]


def observation(path: str, status: int, semantic: str) -> dict:
    return agent_observation(
        action={"method": "GET", "path": path},
        status_code=status,
        response_headers={"content-type": "application/json"},
        response_summary={
            "body_length": len(semantic),
            "semantic_body_sha256": semantic,
            "json_shape": {"data": "list"},
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local no-network maze engineering smoke test")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    recorder = MazeRunRecorder(workspace_root=ROOT, run_id=args.run_id, target_kind="synthetic_local")
    recorder.mark_reset(reset_kind="synthetic_fresh", evaluator_state_hidden=True)

    dom_payload = build_detection_payload(
        path="/playground",
        marker="sift-dom-01",
        probe='<span data-sift-marker="sift-dom-01">sift-dom-01</span>',
        probe_kind="inert_dom_markup",
        expected={"browser_sink_observed": True, "dom_change": True},
    )
    dom_evidence = run_dom_oracle('<span data-sift-marker="sift-dom-01">inert</span>', marker="sift-dom-01").to_dict()
    dom_goal = assess_rule_exit("xss", visible_evidence=dom_evidence, rechecks=[dom_evidence])
    recorder.record_transition(
        observation("/dom/baseline", 200, "empty"),
        {"method": "GET", "path": "/dom/probe"},
        observation("/dom/probe", 200, "dom-delta"),
        evidence=dom_evidence,
        goal=dom_goal,
        detection_payload=dom_payload,
    )

    sql_evidence = run_sql_ast_oracle("time_delay").to_dict()["evidence"]
    sql_goal = assess_rule_exit("injection", visible_evidence=sql_evidence, rechecks=[sql_evidence])
    sql_payload = build_detection_payload(
        path="/api/search",
        marker="sift-sql-01",
        probe="time_delay",
        probe_kind="sql_channel_class",
        expected={"channel": "bounded_timing", "timing_bucket": "above_budget", "requires_recheck": True},
    )
    recorder.record_transition(
        observation("/sql/baseline", 200, "fast"),
        {"method": "GET", "path": "/sql/probe"},
        observation("/sql/probe", 200, "timeout-bucket"),
        evidence=sql_evidence,
        goal=sql_goal,
        detection_payload=sql_payload,
    )

    manifest = recorder.finalize(notes=[
        "synthetic only; no network call was made",
        "detection payloads are manifests, not an execution tool",
    ])
    print(json.dumps({
        "run_id": manifest["run_id"],
        "status": manifest["status"],
        "step_count": manifest["step_count"],
        "manifest": str((ROOT / "artifacts" / "maze-runs" / manifest["run_id"] / "manifest.json").relative_to(ROOT)),
        "safety": manifest["safety"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
