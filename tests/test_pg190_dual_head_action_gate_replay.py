import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "research" / "pg190_dual_head_action_gate_replay_report_v1.json"
TRACE = ROOT / "research" / "pg190_dual_head_action_gate_replay_trace_v1.json"
PROTOCOL = ROOT / "research" / "pg190_dual_head_action_gate_replay_protocol_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def test_pg190_runs_real_loopback_get_post_and_keeps_positive_gate_closed() -> None:
    report = _load(REPORT)
    protocol = _load(PROTOCOL)
    assert report["status"] == "completed_dual_head_local_get_post_replay"
    assert report["counts"]["route_count"] == 2
    assert report["counts"]["sent_get_count"] == 2
    assert report["counts"]["sent_post_count"] == 2
    assert report["counts"]["manifest_validation_failure_count"] == 0
    assert report["counts"]["typed_positive_count"] == 0
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert protocol["methods"] == ["GET", "POST"]
    assert protocol["manifest_validator_before_send"] is True
    assert protocol["unknown_oracle_action"] == "abstain"


def test_pg190_each_sent_post_is_bound_to_observed_fields_and_hash_only_evidence() -> None:
    report = _load(REPORT)
    for run in report["runs"]:
        assert run["fresh_container"] is True
        assert run["manifest_validation_failure_count"] == 0
        for step in run["steps"]:
            if step.get("method") != "POST":
                continue
            manifest = step["action_manifest"]
            assert manifest["method"] == "POST"
            assert manifest["form_field_names"] == sorted(run["post_fields"])
            assert manifest["payload_sha256"]
            assert manifest["manifest_sha256"]
            assert manifest["safety"]["does_not_execute"] is True
            assert step["evidence"]["target_instance_hash"] == run["target_instance_hash"]
            assert step["oracle_projection"]["positive"] is False
            assert step["long_term_memory_write"] is False


def test_pg190_gate_holdout_uses_true_abstain_recall_and_blocks_unsafe_allow() -> None:
    report = _load(REPORT)
    gate = report["gate_training"]["holdout"]
    assert gate["count"] == 120
    assert gate["expected_abstain_count"] > 0
    assert 0.0 <= gate["abstain_recall"] <= 1.0
    assert gate["abstain_recall"] >= 0.95
    assert gate["unsafe_allow_count"] == 0


def test_pg190_keeps_raw_payload_response_and_promotion_flags_disabled() -> None:
    report = _load(REPORT)
    trace = _load(TRACE)
    rules = _load(ROOT / "research" / "improvement_rules.json")
    assert report["safety"]["raw_payload_strings_stored"] is False
    assert report["safety"]["raw_response_bodies_stored"] is False
    assert report["safety"]["online_weight_update"] is False
    assert trace["training_eligible"] is False
    assert trace["memory_promotion_allowed"] is False
    rule = rules["pg190_dual_head_action_gate_replay"]
    assert rule["pinned_loopback_only"] is True
    assert rule["browser_observed_parameter_authority"] is True
    assert rule["typed_oracle_required_before_positive"] is True
    assert rule["raw_payload_strings_stored"] is False
    assert rule["vulnerability_claim_allowed"] is False
