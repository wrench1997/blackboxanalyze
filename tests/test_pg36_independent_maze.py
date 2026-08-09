import json
import threading
from collections import Counter
from pathlib import Path

import httpx

from app.pg36_independent_maze_fixture import make_pg36_server


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg36_fixture_exposes_delayed_confirm_and_deterministic_negative_controls():
    server = make_pg36_server(0, "north")
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{server.server_port}", timeout=3.0, follow_redirects=False) as client:
            screen = client.get("/maze/north/surface-01", params={"branch": "surface-01", "signal": "markup_candidate", "phase": "screen"})
            confirm = client.get("/maze/north/surface-01", params={"branch": "surface-01", "signal": "markup_candidate", "phase": "confirm"})
            error = client.get("/maze/north/surface-01", params={"branch": "surface-01", "signal": "markup_candidate", "phase": "error"})
            timeout = client.get("/maze/north/surface-01", params={"branch": "surface-01", "signal": "markup_candidate", "phase": "timeout"})
        assert screen.status_code == 200
        assert screen.json()["ambiguous"] is True
        assert screen.json().get("typed_effect_ready") is None
        assert confirm.status_code == 200
        assert confirm.json()["typed_effect_ready"] is True
        assert confirm.json()["dom_change"] is True
        assert error.status_code == 400
        assert timeout.status_code == 504
        for response in (screen, confirm, error, timeout):
            body = response.json()
            assert body.get("state_mutated", False) is False
            assert body.get("external_network", False) is False
            assert body.get("script_execution", False) is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_pg36_catalog_has_delayed_phases_two_methods_and_unknown_negative():
    catalog = _load("pg36_independent_maze_catalog_v1.json")
    rows = catalog["samples"]
    assert catalog["independent_target_implementation"] is True
    assert catalog["training_eligible"] is False
    assert catalog["methods"] == ["GET", "POST"]
    assert set(catalog["phases"]) == {"screen", "confirm", "error", "timeout"}
    assert len(rows) == 960
    assert catalog["typed_positive_count"] == 96
    assert catalog["negative_control_count"] == 864
    assert catalog["delayed_confirm_count"] == 240
    assert catalog["ambiguous_screen_count"] == 240
    assert catalog["deterministic_error_count"] == 240
    assert catalog["deterministic_timeout_count"] == 240
    assert catalog["source_count"] == 2
    assert all(row["reset"]["kind"] == "fresh_pg36_http_server" for row in rows)
    assert all(row["reset"]["fresh_target"] and row["reset"]["completed"] for row in rows)
    assert len({row["evidence"]["evidence_hash"] for row in rows}) == len(rows)
    assert "unknown_surface" in {row["family"] for row in rows}
    serialized = json.dumps(catalog, ensure_ascii=False).casefold()
    assert "<script" not in serialized
    assert "onerror" not in serialized
    assert "union select" not in serialized


def test_pg36_trace_is_step_aligned_and_only_known_positive_episodes_are_accepted():
    trace = _load("pg36_independent_maze_trace_v1.json")
    assert trace["independent_target_implementation"] is True
    assert trace["training_eligible"] is False
    assert trace["episode_count"] == 60
    assert trace["accepted_evaluation_episode_count"] == 48
    assert len(trace["steps"]) == 960
    assert set(step["action_manifest"]["method"] for step in trace["steps"]) == {"GET", "POST"}
    assert all(step["online_weight_update"] is False for step in trace["steps"])
    assert all(step["long_term_memory_write"] is False for step in trace["steps"])
    for episode_id in {step["episode_id"] for step in trace["steps"]}:
        steps = [step for step in trace["steps"] if step["episode_id"] == episode_id]
        assert len({step["step_id"] for step in steps}) == len(steps)
        assert any(step["decision"] == "confirmed_positive" for step in steps) == (next(item for item in trace["episodes"] if item["episode_id"] == episode_id)["status"] == "accepted_evaluation")
    positive_steps = [step for step in trace["steps"] if step["decision"] == "confirmed_positive"]
    assert len(positive_steps) == 96
    assert all(step["oracle_projection"].get("negative_control_pair_id") for step in positive_steps)
    assert Counter(item["status"] for item in trace["episodes"]) == Counter({"accepted_evaluation": 48, "trace_only": 12})
