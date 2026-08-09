import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _digest_without_hash(value: dict, field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def test_pg240_source_replay_has_real_get_post_oracle_pairs() -> None:
    report = json.loads((ROOT / "research" / "pg240_pikachu_source_replay_report_v1.json").read_text(encoding="utf-8-sig"))
    dataset = json.loads((ROOT / "research" / "pg240_pikachu_source_replay_dataset_v1.json").read_text(encoding="utf-8-sig"))
    protocol = json.loads((ROOT / "research" / "pg240_pikachu_source_replay_protocol_v1.json").read_text(encoding="utf-8-sig"))
    assert report["status"] == "completed_cross_application_source_replay"
    assert report["source_repository"]["commit"] == "5e1e8d9d14a3ba61d62f28cf35531c4df4dd24fc"
    assert report["counts"] == {
        "fresh_container_count": 14,
        "get_episode_count": 10,
        "post_episode_count": 4,
        "database_health_gate_count": 14,
        "ai_send_count": 14,
        "reference_send_count": 14,
        "negative_send_count": 14,
        "known_positive_fixture_record_count": 12,
        "negative_fixture_clean_count": 14,
        "result_fixture_verified_count": 8,
        "typed_effect_confirmed_count": 8,
        "false_positive_count": 0,
        "docker_restart_used_count": 0,
    }
    assert report["promotion"]["training_eligible"] is False
    assert report["honesty"]["source_holdout_not_full_backend_independence"] is True
    assert protocol["positive_negative_pair_required"] is True
    assert protocol["training_promotion_allowed"] is False
    assert all(record["training_eligible"] is False for record in dataset["records"])
    assert all(record["raw_payload_strings_stored"] is False for record in dataset["records"])
    assert all(record["raw_response_bodies_stored"] is False for record in dataset["records"])
    assert _digest_without_hash(report, "report_sha256") == report["report_sha256"]
    assert _digest_without_hash(dataset, "dataset_sha256") == dataset["dataset_sha256"]


def test_pg240_source_holdout_training_is_nontrivial_and_frozen() -> None:
    report = json.loads((ROOT / "research" / "pg240_source_holdout_capacity_report_v1.json").read_text(encoding="utf-8-sig"))
    dataset = json.loads((ROOT / "research" / "pg240_source_holdout_capacity_dataset_v1.json").read_text(encoding="utf-8-sig"))
    rule = json.loads((ROOT / "research" / "improvement_rules.json").read_text(encoding="utf-8-sig"))["pg240_pikachu_source_holdout_replay"]
    artifact = ROOT / report["selected"]["artifact"]
    metrics = report["selected"]["metrics"]["seed_holdout"]
    assert report["status"] == "completed_source_heldout_capacity_training"
    assert report["holdout_source"] == "pg240_pikachu_source_replay"
    assert report["holdout_seeds"] == [24002, 23632]
    assert report["counts"]["holdout_rows"] == 21
    assert report["counts"]["holdout_action_counts"] == {"abstain": 17, "send_candidate": 4}
    assert report["safety_abstain_gate_pass"] is True
    assert report["capability_gate_pass"] is True
    assert metrics["positive_send_recall"] == 1.0
    assert metrics["abstain_recall"] == 1.0
    assert metrics["false_send_count"] == 0
    assert metrics["missed_send_count"] == 0
    assert report["frozen_body_changed"] is False
    assert artifact.exists()
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == report["selected"]["artifact_sha256"]
    assert dataset["contract"]["fresh_source_holdout_seeds_never_in_training"] is True
    assert dataset["contract"]["raw_payload_strings_stored"] is False
    assert dataset["contract"]["raw_response_bodies_stored"] is False
    assert rule["training_eligible"] is False
    assert rule["capability_gate_pass"] is True


def test_pg240_registry_and_rule_counts_are_consistent() -> None:
    registry = json.loads((ROOT / "research" / "pg_pk_24_cross_lab_registry_v1.json").read_text(encoding="utf-8-sig"))
    rule = json.loads((ROOT / "research" / "improvement_rules.json").read_text(encoding="utf-8-sig"))["pg240_pikachu_source_holdout_replay"]
    targets = {target["target_id"]: target for target in registry["targets"]}
    assert registry["evaluation_only_target_count"] == 116
    assert targets["pg240_pikachu_source_replay"]["training_eligible"] is False
    assert targets["pg240_source_holdout_capacity_training"]["holdout_rows"] == 21
    assert targets["pg240_source_holdout_capacity_training"]["holdout_false_send_count"] == 0
    assert rule["source_commit"] == "5e1e8d9d14a3ba61d62f28cf35531c4df4dd24fc"
