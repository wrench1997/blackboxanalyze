from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8-sig"))


def _audit_module():
    path = ROOT / "scripts" / "audit_pg327b_paired_fresh_replay.py"
    spec = importlib.util.spec_from_file_location("pg327b_audit_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pg327b_has_two_phase_fresh_paired_replay_and_strict_evidence():
    report = _load("pg327b_paired_fresh_replay_report_v1.json")
    trace = _load("pg327b_paired_fresh_replay_trace_v1.json")
    assert report["status"] == "completed_local_docker_pg327b_paired_replay"
    assert report["counts"]["phase_count"] == 2
    assert report["counts"]["route_count_per_phase"] == 9
    assert report["counts"]["total_phase_routes"] == 18
    assert report["forgetting"]["paired_replay_present"] is True
    assert report["forgetting"]["same_canary_route_set"] is True
    assert all(report["checks"][key] is True for key in ("fresh_reset_before_all", "fresh_reset_after_all", "distinct_container_pairs", "typed_evidence_before", "typed_evidence_after", "failure_action_changed_before", "failure_action_changed_after", "role_bound_belief_before", "role_bound_belief_after", "context_firewall_before", "context_firewall_after", "raw_payload_excluded", "raw_response_excluded", "before_after_checkpoint_distinct"))
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert trace["raw_payload_stored"] is False
    assert trace["raw_response_body_stored"] is False
    assert trace["training_eligible"] is False


def test_pg327b_read_only_audit_passes_without_contacting_target():
    result = _audit_module().audit()
    assert result["status"] == "passed"
    assert result["promotion_allowed"] is False
    assert result["target_contacted"] is False

