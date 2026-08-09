import json
import threading
from collections import Counter, defaultdict
from pathlib import Path

import httpx

from app.pg37_counterfactual_fixture import make_pg37_server


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg37_fixture_exposes_three_surfaces_and_both_channels_safely():
    server = make_pg37_server(0, "atlas")
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}/counterfactual/atlas/surface-01"
        with httpx.Client(timeout=3.0) as client:
            responses = [
                client.get(base, params={"branch": "surface-01", "signal": "markup_candidate", "surface": variant, "phase": phase})
                for variant in ("compact", "nested", "headerized")
                for phase in ("screen", "confirm")
            ]
            post = client.post(base, json={"branch": "surface-01", "signal": "markup_candidate", "surface": "nested", "phase": "confirm"})
            error = client.get(base, params={"branch": "surface-01", "signal": "markup_candidate", "surface": "compact", "phase": "error"})
            timeout = client.get(base, params={"branch": "surface-01", "signal": "markup_candidate", "surface": "compact", "phase": "timeout"})
        assert all(response.status_code == 200 for response in responses)
        assert sum(int(response.json().get("typed_effect_ready", False)) for response in responses) == 3
        assert post.status_code == 200 and post.json()["typed_effect_ready"] is True
        assert error.status_code == 400 and timeout.status_code == 504
        assert len({response.json().get("view", response.json().get("meta", {}).get("view")) for response in responses}) == 3
        for response in [*responses, post, error, timeout]:
            body = response.json()
            assert body.get("state_mutated", False) is False
            assert body.get("external_network", False) is False
            assert body.get("script_execution", False) is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_pg37_catalog_has_true_counterfactual_counts_and_no_raw_probe_text():
    catalog = _load("pg37_counterfactual_catalog_v1.json")
    rows = catalog["samples"]
    assert catalog["independent_target_implementation"] is True
    assert catalog["training_eligible"] is True
    assert catalog["methods"] == ["GET", "POST"]
    assert set(catalog["surface_variants"]) == {"compact", "nested", "headerized"}
    assert len(rows) == 2880
    assert catalog["typed_positive_count"] == 288
    assert catalog["negative_control_count"] == 2592
    assert catalog["counterfactual_pair_count"] == 480
    assert catalog["trace_episode_count"] == 180
    assert catalog["accepted_evaluation_episode_count"] == 144
    assert Counter(row["surface_variant"] for row in rows) == Counter({"compact": 960, "nested": 960, "headerized": 960})
    assert Counter(row["method"] for row in rows) == Counter({"GET": 1440, "POST": 1440})
    assert all(row["reset"]["fresh_target"] and row["reset"]["completed"] for row in rows)
    assert len({row["evidence"]["evidence_hash"] for row in rows}) == len(rows)
    grouped = defaultdict(list)
    for row in rows:
        if row["oracle_projection"]["positive"] and row["phase"] == "confirm":
            grouped[(row["implementation"], row["surface_id"], row["sampling_seed"], row["method"])].append(row)
    assert all({row["surface_variant"] for row in group} == {"compact", "nested", "headerized"} for group in grouped.values())
    serialized = json.dumps(catalog, ensure_ascii=False).casefold()
    assert "<script" not in serialized
    assert "onerror" not in serialized
    assert "union select" not in serialized


def test_pg37_trace_has_unique_steps_and_typed_positive_negative_pairs():
    trace = _load("pg37_counterfactual_trace_v1.json")
    assert trace["episode_count"] == 180
    assert trace["accepted_evaluation_episode_count"] == 144
    assert len(trace["steps"]) == 2880
    assert len({step["step_id"] for step in trace["steps"]}) == len(trace["steps"])
    assert {step["action_manifest"]["method"] for step in trace["steps"]} == {"GET", "POST"}
    assert all(step["online_weight_update"] is False for step in trace["steps"])
    assert all(step["long_term_memory_write"] is False for step in trace["steps"])
    positive = [step for step in trace["steps"] if step["decision"] == "confirmed_positive"]
    assert len(positive) == 288
    assert all(step["oracle_projection"].get("negative_control_pair_id") for step in positive)
    assert Counter(item["status"] for item in trace["episodes"]) == Counter({"accepted_evaluation": 144, "trace_only": 36})
