from pathlib import Path

import pytest

from app.dom_oracle import run_dom_oracle
from app.juice_shop_adapter import agent_observation
from app.maze_engine import MazeRunRecorder, load_manifest, validate_evidence, verify_ledger


def obs(path: str, status: int, semantic: str):
    return agent_observation(
        action={"method": "GET", "path": path},
        status_code=status,
        response_headers={"content-type": "application/json"},
        response_summary={"body_length": len(semantic), "semantic_body_sha256": semantic, "json_shape": {"data": "list"}},
    )


def test_maze_run_manifest_is_replayable_and_hash_chained(tmp_path: Path):
    recorder = MazeRunRecorder(workspace_root=tmp_path, run_id="smoke-001")
    recorder.mark_reset()
    evidence = run_dom_oracle('<span data-sift-marker="sift-marker">inert</span>').to_dict()
    recorder.record_transition(
        obs("/before", 200, "empty"),
        {"method": "GET", "path": "/after"},
        obs("/after", 200, "node"),
        evidence=evidence,
        goal={"status": "candidate"},
    )
    manifest = recorder.finalize(notes=["engineering smoke"])
    loaded = load_manifest(tmp_path / "artifacts" / "maze-runs" / "smoke-001" / "manifest.json")
    assert loaded["schema_version"] == "sift-maze-run-v1"
    assert loaded["step_count"] == 1
    assert manifest["artifacts"]["ledger_head"] == verify_ledger(tmp_path / "artifacts" / "maze-runs" / "smoke-001" / "evidence.jsonl")["head"]
    assert Path(tmp_path / "artifacts" / "maze-runs" / "smoke-001" / "graph.json").exists()


def test_maze_run_rejects_unsafe_evidence_and_forbidden_raw_content():
    with pytest.raises(ValueError):
        validate_evidence({"script_execution": True})
    with pytest.raises(ValueError):
        validate_evidence({"body_preview": "secret-looking raw response"})

