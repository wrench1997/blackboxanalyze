import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "research" / "pg188_xxl_replay_action_training_report_v1.json"
TRACE = ROOT / "research" / "pg188_xxl_replay_action_training_trace_v1.json"
PROTOCOL = ROOT / "research" / "pg188_xxl_replay_action_training_protocol_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_pg188_trains_large_body_with_replay_and_reports_negative_gate() -> None:
    report = _load(REPORT)
    trace = _load(TRACE)
    protocol = _load(PROTOCOL)
    assert report["status"] == "completed_xxl_replay_action_training"
    assert report["body_config"]["d_model"] == 1024
    assert report["body_config"]["layers"] == 8
    assert report["source"]["lm_replay_rows"] == 4096
    assert report["source"]["target_trace_used_for_training"] is False
    assert {row["variant"] for row in report["variants"]} == {"frozen_xxl_head", "replay_xxl_low_lr", "replay_xxl_strong"}
    assert all(row["parameter_count"] > 100_000_000 for row in report["variants"])
    assert report["selection"]["selected_variant"] is None
    assert report["selection"]["training_artifact_promotion_allowed"] is False
    assert report["selection"]["memory_promotion_allowed"] is False
    assert trace["target_trace_used_for_training"] is False
    assert protocol["parameter_target"] == 101380329
    assert protocol["forgetting_gate"]["catastrophic_forgetting_blocks_selection"] is True


def test_pg188_keeps_raw_payloads_and_responses_out_of_training_artifacts() -> None:
    report = _load(REPORT)
    trace = _load(TRACE)
    protocol = _load(PROTOCOL)
    assert report["safety"]["raw_payloads_in_model"] is False
    assert report["safety"]["raw_responses_in_model"] is False
    assert report["safety"]["target_trace_in_training"] is False
    assert trace["training_artifact_promotion_allowed"] is False
    assert trace["memory_promotion_allowed"] is False
    assert protocol["selection_gate"]["target_trace_training_forbidden"] is True
