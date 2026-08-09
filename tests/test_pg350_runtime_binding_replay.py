from __future__ import annotations

import json
from pathlib import Path

from app.pg348_dynamic_runtime import load_registry
from scripts.run_pg350_runtime_binding_replay import replay


ROOT = Path(__file__).resolve().parents[1]


def test_pg350_replay_binds_real_get_and_post_but_persists_only_abstract_projection() -> None:
    registry = load_registry(ROOT / "fixtures" / "pg348" / "registry_v1.json")
    report, sidecars, shown = replay(registry, seeds=(35001,), max_routes=2, show_wire=False)
    assert report["status"] == "completed_evaluator_only"
    assert report["counts"] == {
        "episodes": 2,
        "routes": 2,
        "seeds": 1,
        "get_episodes": 1,
        "post_episodes": 1,
        "confirmed_positive": 2,
        "candidate_typed": 2,
        "reference_typed": 2,
        "negative_clean": 2,
        "replay_consistent": 2,
        "failure_action_change": 2,
    }
    assert shown == []
    serialized = json.dumps({"report": report, "sidecars": sidecars}, ensure_ascii=False)
    assert "PG350S" not in serialized
    assert "http://127.0.0.1:" not in serialized
    assert report["raw_wire_policy"]["raw_wire_stored"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["scientific_scope"]["binder_and_neural_capability_separate"] is True


def test_pg350_replay_ephemeral_wire_is_only_returned_when_explicitly_shown() -> None:
    registry = load_registry(ROOT / "fixtures" / "pg348" / "registry_v1.json")
    report, sidecars, shown = replay(registry, seeds=(35002,), max_routes=2, show_wire=True)
    assert report["status"] == "completed_evaluator_only"
    assert len(shown) == 8  # four role bindings × GET/POST
    assert any(wire.startswith("GET http://127.0.0.1:") for wire in shown)
    assert any(wire.startswith("POST http://127.0.0.1:") for wire in shown)
    assert any("%27" in wire for wire in shown)
    # The returned sidecar still cannot contain the ephemeral marker/wire.
    assert "PG350S" not in json.dumps(sidecars, ensure_ascii=False)
