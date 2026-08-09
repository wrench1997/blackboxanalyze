import json
from pathlib import Path


def test_fast_local_docker_replay_policy_is_scoped_and_quarantines_failures() -> None:
    rules = json.loads(Path("research/improvement_rules.json").read_text(encoding="utf-8"))
    policy = rules["fast_local_docker_replay_policy"]

    assert policy["scope"]["owned_containers_only"] is True
    assert policy["scope"]["pinned_image_or_digest_required"] is True
    assert policy["scope"]["published_ports_must_bind_loopback"] is True
    assert policy["fast_recovery"]["max_retries_per_episode"] == 1
    assert policy["fast_recovery"]["daemon_restart_allowed"] is False
    assert policy["fast_recovery"]["restart_all_containers_forbidden"] is True
    assert policy["failure_handling"]["failed_replay_rows_enter_training"] is False
    assert policy["failure_handling"]["quarantine_after_retry_exhausted"] is True
    assert policy["acceptance_after_recovery"]["fresh_reset_required"] is True
    assert policy["acceptance_after_recovery"]["typed_oracle_and_matched_negative_required"] is True


def test_weekend_remote_a800_is_gpu0_only_and_keeps_data_gates() -> None:
    rules = json.loads(Path("research/improvement_rules.json").read_text(encoding="utf-8"))
    schedule = rules["execution_location_policy"]["training_schedule"]["weekend_remote_a800"]
    assert schedule["days"] == ["Saturday", "Sunday"]
    assert schedule["host"] == "112.111.7.91:60228"
    assert schedule["gpu"] == "NVIDIA A800 GPU0 only"
    assert schedule["gpu_index"] == 0
    assert schedule["cuda_visible_devices"] == "0"
    assert schedule["other_gpus_touched"] is False
    assert schedule["requires_data_code_audit_hash_lock"] is True
    assert schedule["requires_information_preservation_gate"] is True
    assert schedule["promotion_still_blocked_until_fresh_holdout"] is True
