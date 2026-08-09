import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg80_frozen_replay_keeps_zero_recall_failure_visible():
    report = _read("pg80_frozen_trace_transformer_replay_report_v1.json")
    assert report["dataset"]["step_count"] == 270
    assert report["metrics"]["confirm_recall"] == 0.0
    assert report["metrics"]["false_accept_count"] == 0
    assert report["metrics"]["unknown_token_count"] == 2100
    assert report["capability_gate"]["status"] == "blocked"
    assert report["capability_gate"]["checks"]["known_recall_min"] is False
    assert report["promotion"]["training_allowed"] is False
