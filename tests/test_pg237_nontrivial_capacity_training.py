import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pg237_fresh_replay_has_typed_positive_and_negative_pairs() -> None:
    report = json.loads((ROOT / "research" / "pg237_pikachu_result_fixture_replay_report_v1.json").read_text(encoding="utf-8-sig"))
    assert report["status"] == "completed_fresh_typed_positive_negative_result_fixture_replay"
    assert report["seeds"] == [23701, 23702]
    assert report["counts"]["fresh_container_count"] == 14
    assert report["counts"]["ai_send_count"] == 14
    assert report["counts"]["reference_send_count"] == 14
    assert report["counts"]["negative_send_count"] == 14
    assert report["counts"]["typed_effect_confirmed_count"] == 8
    assert report["counts"]["result_fixture_verified_count"] == 8
    assert report["counts"]["false_positive_count"] == 0
    assert report["promotion"]["training_eligible"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert report["promotion"]["raw_payload_strings_stored"] is False
    assert report["promotion"]["raw_response_bodies_stored"] is False


def test_pg237_capacity_uses_nontrivial_seed_holdout_and_freezes_body() -> None:
    report = json.loads((ROOT / "research" / "pg237_capacity_training_report_v1.json").read_text(encoding="utf-8-sig"))
    dataset = json.loads((ROOT / "research" / "pg237_capacity_training_dataset_v1.json").read_text(encoding="utf-8-sig"))
    rule = json.loads((ROOT / "research" / "improvement_rules.json").read_text(encoding="utf-8-sig"))["pg237_pikachu_nontrivial_capacity_training"]
    metrics = report["selected"]["metrics"]["seed_holdout"]
    artifact = ROOT / report["selected"]["artifact"]

    assert report["status"] == "completed_nontrivial_seed_holdout_capacity_training"
    assert report["counts"]["unique_records"] == 163
    assert report["counts"]["duplicate_records"] == 0
    assert report["counts"]["holdout_rows"] == 21
    assert report["counts"]["holdout_action_counts"] == {"abstain": 17, "send_candidate": 4}
    assert report["honesty"]["seed23702_is_never_in_training"] is True
    assert report["honesty"]["seed23632_is_never_in_training"] is True
    assert report["safety_abstain_gate_pass"] is True
    assert report["capability_gate_pass"] is True
    assert metrics["next_token_accuracy"] == 0.89030612
    assert metrics["positive_send_recall"] == 1.0
    assert metrics["abstain_recall"] == 1.0
    assert metrics["false_send_count"] == 0
    assert metrics["missed_send_count"] == 0
    assert report["frozen_body_changed"] is False
    assert artifact.exists()
    assert _sha256(artifact) == report["selected"]["artifact_sha256"]
    assert dataset["contract"]["holdout_contains_positive_and_abstain"] is True
    assert dataset["contract"]["raw_payload_strings_stored"] is False
    assert dataset["contract"]["raw_response_bodies_stored"] is False
    assert report["promotion"]["training_promotion_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert rule["all_abstain_holdout_forbidden"] is True
    assert rule["selected_hidden_dim"] == 2048
    assert rule["capability_gate_pass"] is True

