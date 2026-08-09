from __future__ import annotations

import json

from scripts.run_pg387_ctf_context_process_replay import run_process_replay


def test_pg387_process_replay_sends_only_reviewed_context_and_asks_elsewhere() -> None:
    report, wires = run_process_replay(show_wire=False)
    assert report["status"] == "completed_process_only_context_diagnostic"
    assert report["counts"] == {
        "rows": 16,
        "cases": 4,
        "roles": 4,
        "typed_effect": 3,
        "ask_context": 12,
        "action_changed": 4,
        "negative_violation": 0,
        "fresh_reset": 4,
    }
    assert wires == []
    assert report["execution"]["docker_started"] is False
    assert report["training_eligible"] == 0


def test_pg387_process_replay_wire_is_ephemeral_and_bounded() -> None:
    report, wires = run_process_replay(show_wire=True)
    assert len(wires) == 4
    assert all("127.0.0.1" not in wire for wire in wires)
    assert all("%25253A" in wire for wire in wires)
    persisted = json.dumps(report, ensure_ascii=False)
    for marker in ("PG387_CAND", "PG387_REFERENCE", "PG387_REPLAY", "PG387_NEG", "GET /local-filter", "http://", "https://"):
        assert marker not in persisted


def test_pg387_process_replay_marks_unsafe_js_contexts_as_ask() -> None:
    report, _ = run_process_replay(show_wire=False)
    unsafe = [row for row in report["rows"] if row["case_ref"] in {"script_loader_policy", "storage_policy_guard", "dynamic_code_guard"}]
    assert len(unsafe) == 12
    assert all(row["status"] == "ask_context" for row in unsafe)
    assert all(row["model_decision"]["safe_to_send"] == "0" for row in unsafe)
