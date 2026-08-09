import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def test_pg207_replays_get_post_on_two_fresh_sources_and_abstains_unknown_oracle() -> None:
    report = _load("research/pg207_cross_lab_xxl_replay_report_v1.json")
    assert report["status"] == "completed_cross_lab_three_seed_xxl_replay"
    counts = report["counts"]
    assert counts["source_count"] == 2
    assert counts["seed_count"] == 3
    assert counts["fresh_container_count"] == 6
    assert counts["route_replay_count"] == 12
    assert counts["get_count"] == 6
    assert counts["post_count"] == 6
    assert counts["unknown_oracle_abstain_count"] == 12
    assert counts["candidate_send_count"] == 0
    assert counts["field_fault_count"] == 48
    assert counts["network_allowed_on_fault_count"] == 0
    assert counts["false_positive_count"] == 0
    assert report["model"]["base_parameter_count"] > 100_000_000
    assert len({row["target_instance_hash"] for row in report["targets"]}) == 6
    assert all(row["candidate_generated"] for row in report["route_runs"])
    assert all(row["candidate_sent"] is False for row in report["route_runs"])
    assert all(row["model_decision"]["effective_action"] == "abstain" for row in report["route_runs"])


def test_pg207_fault_gate_and_promotion_are_fail_closed() -> None:
    report = _load("research/pg207_cross_lab_xxl_replay_report_v1.json")
    protocol = _load("research/pg207_cross_lab_xxl_replay_protocol_v1.json")
    rules = _load("research/improvement_rules.json")
    serialized = json.dumps(report, ensure_ascii=False)
    assert "<span" not in serialized
    assert "response_body" not in serialized
    assert report["promotion"]["training_eligible"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert protocol["raw_payload_and_response_excluded"] is True
    assert protocol["training_promotion_allowed"] is False
    assert rules["pg207_cross_lab_xxl_replay"]["network_allowed_on_fault_count"] == 0
    assert rules["pg207_cross_lab_xxl_replay"]["candidate_send_count"] == 0

