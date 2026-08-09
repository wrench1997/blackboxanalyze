import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def _sha256_file(relative_path: str) -> str:
    return hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()


def test_pg118_report_passes_both_blind_gates_without_promotion():
    report = _load("pg118_transition_training_report_v1.json")
    assert report["status"] == "completed_pg118_transition_slot_training"
    assert report["scope"]["real_vulnerability_scanner_claim_allowed"] is False
    assert report["scope"]["device"] == "cuda"
    assert report["scope"]["feature_dim"] == 44
    assert report["scope"]["parameter_count"] == 4804
    assert report["training"]["dev_metrics"]["accuracy"] == 1.0

    pg114 = report["blind_pg114"]
    assert pg114["family_holdout_confirm_recall"] == 1.0
    assert pg114["decoy_false_accept_count"] == 0
    assert pg114["withheld_oracle_abstain_rate"] == 1.0
    assert pg114["step_metrics"]["accuracy"] == 1.0

    pg117 = report["blind_pg117"]
    assert pg117["route_positive_recall"] == 1.0
    assert pg117["decoy_false_accept_count"] == 0
    assert pg117["blind_oracle_abstain_rate"] == 1.0
    assert pg117["step_metrics"]["accuracy"] == 0.958333

    assert all(report["checks"].values())
    assert report["promotion"]["checkpoint_written"] is True
    assert report["promotion"]["training_artifact_promotion_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg118_collection_is_double_encoded_get_post_and_slot_bound():
    report = _load("pg118_transition_training_report_v1.json")
    collection = report["collection"]
    assert collection["training_source_set"] == ["pg116_identity", "pg118_delta_double_encoding"]
    assert collection["holdout_source_set"] == ["pg117_gamma_double_encoding"]
    assert collection["delta_target_instance_count"] == 6
    assert collection["delta_episode_count"] == 24
    assert collection["delta_step_count"] == 96
    assert collection["delta_get_step_count"] == 48
    assert collection["delta_post_step_count"] == 48
    assert collection["delta_encoding_chain"] == ["html_entity", "url_percent"]
    assert collection["gamma_holdout_encoding_chain"] == ["url_percent", "html_entity"]
    assert collection["fresh_reset_per_step"] is True
    assert collection["evidence_hash_valid"] is True
    assert collection["rule_ir_slot_binding_count"] == 24
    assert collection["train_unique_row_count"] == 312
    assert collection["dev_unique_row_count"] == 312
    assert collection["train_rows_after_balance"] == 396
    assert collection["dev_rows_after_balance"] == 396

    trace = _load("pg118_transition_training_trace_v1.json")
    assert trace["evaluation_only"] is False
    assert trace["training_eligible"] is True
    assert trace["memory_promotion_allowed"] is False
    assert trace["rule_ir_slot_binding_count"] == 24
    episodes = [episode for target in trace["sources"] for episode in target["episodes"]]
    assert len(episodes) == 24
    for episode in episodes:
        assert episode["episode_report"]["status"] == "accepted_evaluation"
        assert episode["negative_control_pair_clear"] is True
        assert [step["action_manifest"]["method"] for step in episode["steps"]] == ["GET", "GET", "POST", "POST"]
        assert len({step["fresh_reset"]["reset_epoch"] for step in episode["steps"]}) == 4
        step_hashes = {step["evidence_sha256"] for step in episode["steps"]}
        binding = episode["rule_ir_slot_binding"]
        assert binding["evidence_sha256"] in step_hashes
        assert set(binding["shadow_probe_evidence_sha256"]).issubset(step_hashes)
        for step in episode["steps"]:
            model_input = step["model_input"]
            assert "location_changed" in model_input["response_projection"]
            model_text = json.dumps(model_input, ensure_ascii=False).casefold()
            for forbidden in ("oracle", "positive_authority", "family", "target_instance_id", "probe_ref", "probe_sha256"):
                assert forbidden not in model_text
            assert step["online_weight_update"] is False
            assert step["long_term_memory_write"] is False


def test_pg118_dataset_protocol_rules_and_registry_are_consistent():
    dataset = _load("pg118_transition_training_dataset_v1.json")
    assert dataset["training_eligible"] is True
    assert dataset["memory_promotion_allowed"] is False
    assert dataset["transition_delta_slot"] == "response_projection.location_changed + transition_delta"
    assert dataset["pg117_gamma_excluded_from_training"] is True
    assert len(dataset["train_rows"]) == 396
    assert len(dataset["dev_rows"]) == 396
    assert all(row["training_eligible"] is True for row in dataset["train_rows"] + dataset["dev_rows"])
    assert all(row["memory_promotion_allowed"] is False for row in dataset["train_rows"] + dataset["dev_rows"])

    protocol = _load("pg118_transition_rule_ir_training_protocol_v1.json")
    assert protocol["model_contract"]["feature_dim"] == 44
    assert protocol["model_contract"]["previous_checkpoint_reuse_forbidden"] is True
    assert protocol["encoding_split"]["gamma_rows_excluded_from_training"] is True
    assert protocol["action_contract"]["fresh_reset_per_action"] is True
    assert protocol["promotion"]["memory_promotion_allowed"] is False

    rules = _load("improvement_rules.json")
    policy = rules["pg118_transition_slot_training_policy"]
    assert policy["feature_dim"] == 44
    assert policy["parameter_count"] == 4804
    assert policy["pg117_route_positive_recall"] == 1.0
    assert policy["pg117_decoy_false_accept_count"] == 0
    assert policy["pg117_unknown_abstain_rate"] == 1.0
    assert policy["training_artifact_promotion_allowed"] is False
    assert policy["memory_promotion_allowed"] is False

    registry = _load("pg_pk_24_cross_lab_registry_v1.json")
    assert registry["training_eligible_target_count"] == 40
    assert registry["evaluation_only_target_count"] == 116
    entry = next(item for item in registry["targets"] if item["target_id"] == "pg118_transition_delta_slot_training")
    assert entry["training_eligible"] is True
    assert entry["target_instance_count"] == 6
    assert entry["step_count"] == 96
    assert entry["get_step_count"] == 48
    assert entry["post_step_count"] == 48
    assert entry["training_artifact_promotion_allowed"] is False
    assert entry["memory_promotion_allowed"] is False


def test_pg118_report_source_hashes_and_checkpoint_exist():
    report = _load("pg118_transition_training_report_v1.json")
    for key, relative_path in {
        "delta_target": "app/pg118_transition_training_target.py",
        "delta_bridge": "app/pg118_transition_replay.py",
        "decoder": "app/pg118_transition_rule_ir_decoder.py",
        "runner": "scripts/run_pg118_transition_rule_ir_training.py",
        "pg116_dataset": "research/pg116_multisource_training_dataset_v1.json",
        "pg117_report": "research/pg117_double_holdout_report_v1.json",
        "pg114_report": "research/pg114_family_holdout_replay_report_v1.json",
    }.items():
        assert _sha256_file(relative_path) == report["source"][key]
    assert (ROOT / "artifacts/pg118-transition-rule-ir-decoder-v1/model.pt").exists()
