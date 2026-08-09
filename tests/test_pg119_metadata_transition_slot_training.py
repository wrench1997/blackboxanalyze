import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def _sha256_file(relative_path: str) -> str:
    return hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def test_pg119_report_passes_seed_holdout_and_metadata_slot_ablation():
    report = _load("pg119_metadata_training_report_v1.json")
    assert report["status"] == "completed_pg119_metadata_transition_slot_training"
    assert report["scope"]["real_vulnerability_scanner_claim_allowed"] is False
    assert report["scope"]["device"] == "cuda"
    assert report["scope"]["feature_dim"] == 48
    assert report["scope"]["parameter_count"] == 4996
    assert report["training"]["dev_metrics"]["accuracy"] == 0.998106

    pg114 = report["blind_pg114"]
    assert pg114["family_holdout_confirm_recall"] == 1.0
    assert pg114["decoy_false_accept_count"] == 0
    assert pg114["withheld_oracle_abstain_rate"] == 1.0

    pg117 = report["blind_pg117"]
    assert pg117["metadata_positive_recall"] == 1.0
    assert pg117["decoy_false_accept_count"] == 0
    assert pg117["blind_oracle_abstain_rate"] == 1.0
    assert pg117["cross_seed"]["positive_recall_variance"] == 0.0

    pg119 = report["blind_pg119"]
    assert pg119["metadata_positive_recall"] == 1.0
    assert pg119["decoy_false_accept_count"] == 0
    assert pg119["blind_oracle_abstain_rate"] == 1.0
    assert pg119["cross_seed"]["positive_recall_variance"] == 0.0
    ablation = report["slot_ablation_pg119"]
    assert ablation["metadata_positive_recall"] == 0.0
    assert report["checks"]["metadata_slot_ablation_changes_prediction"] is True
    assert report["checks"]["full_recall_above_metadata_slot_ablation"] is True
    assert all(report["checks"].values())
    assert report["promotion"]["training_artifact_promotion_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg119_trace_keeps_third_encoding_get_post_reset_and_evidence_contracts():
    report = _load("pg119_metadata_training_report_v1.json")
    collection = report["collection"]
    assert collection["training_source_set"] == ["pg116_identity", "pg118_delta_double_encoding", "pg119_metadata_triple_encoding"]
    assert collection["metadata_training_target_instance_count"] == 6
    assert collection["metadata_holdout_target_instance_count"] == 3
    assert collection["metadata_training_episode_count"] == 24
    assert collection["metadata_training_step_count"] == 96
    assert collection["metadata_training_get_step_count"] == 48
    assert collection["metadata_training_post_step_count"] == 48
    assert collection["metadata_encoding_chain"] == ["unicode_escape", "html_entity", "url_percent"]
    assert collection["fresh_reset_per_step"] is True
    assert collection["evidence_hash_valid"] is True
    assert collection["rule_ir_slot_binding_count"] == 24

    trace = _load("pg119_metadata_training_trace_v1.json")
    assert trace["evaluation_only"] is False
    assert trace["training_eligible"] is True
    assert trace["memory_promotion_allowed"] is False
    assert len(trace["sources"]) == 6
    assert len(trace["holdout_sources"]) == 3
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
            assert "metadata_changed" in step["model_input"]["response_projection"]
            model_text = json.dumps(step["model_input"], ensure_ascii=False).casefold()
            for forbidden in ("oracle", "positive_authority", "family", "target_instance_id", "probe_ref", "probe_sha256"):
                assert forbidden not in model_text
            assert step["online_weight_update"] is False
            assert step["long_term_memory_write"] is False
        for record in episode["evidence_records"]:
            body = dict(record)
            declared = body.pop("evidence_hash")
            assert declared == _sha256_json(body)


def test_pg119_dataset_protocol_and_rules_are_consistent():
    dataset = _load("pg119_metadata_training_dataset_v1.json")
    assert dataset["training_eligible"] is True
    assert dataset["memory_promotion_allowed"] is False
    assert dataset["transition_slots"] == ["response_projection.location_changed + transition_delta", "response_projection.metadata_changed + transition_delta"]
    assert dataset["pg119_holdout_excluded_from_training"] is True
    assert dataset["pg117_gamma_excluded_from_training"] is True
    assert dataset["train_unique_row_count"] == 444
    assert dataset["dev_unique_row_count"] == 444
    assert len(dataset["train_rows"]) == 528
    assert len(dataset["dev_rows"]) == 528
    assert {row["source"] for row in dataset["train_rows"]} == {"pg116_identity", "pg118_delta_double_encoding", "pg119_metadata_triple_encoding"}
    assert all(row["training_eligible"] is True for row in dataset["train_rows"] + dataset["dev_rows"])
    assert all(row["memory_promotion_allowed"] is False for row in dataset["train_rows"] + dataset["dev_rows"])

    protocol = _load("pg119_metadata_transition_slot_training_protocol_v1.json")
    assert protocol["model_contract"]["feature_dim"] == 48
    assert protocol["model_contract"]["previous_checkpoint_reuse_forbidden"] is True
    assert protocol["encoding_split"]["metadata_holdout_seeds_excluded_from_training"] is True
    assert protocol["action_contract"]["fresh_reset_per_action"] is True
    assert protocol["ablation"]["full_recall_must_exceed_ablation"] is True
    assert protocol["maze_distance"]["model_input_allowed"] is False
    assert protocol["maze_distance"]["confirmation_allowed"] is False
    assert protocol["promotion"]["memory_promotion_allowed"] is False

    rules = _load("improvement_rules.json")
    policy = rules["pg119_metadata_transition_slot_training_policy"]
    assert policy["feature_dim"] == 48
    assert policy["parameter_count"] == 4996
    assert policy["pg119_positive_recall"] == 1.0
    assert policy["pg119_slot_ablation_positive_recall"] == 0.0
    assert policy["pg117_cross_seed_positive_recall_variance"] == 0.0
    assert policy["pg119_cross_seed_positive_recall_variance"] == 0.0
    assert policy["metadata_slot_ablation_changes_prediction"] is True
    assert policy["training_artifact_promotion_allowed"] is False
    assert policy["memory_promotion_allowed"] is False
    assert policy["maze_distance"]["display_scalar_ordering_only"] is True


def test_pg119_source_hashes_and_checkpoint_are_persisted():
    report = _load("pg119_metadata_training_report_v1.json")
    for key, relative_path in {
        "metadata_target": "app/pg119_metadata_training_target.py",
        "metadata_bridge": "app/pg119_metadata_replay.py",
        "decoder": "app/pg119_metadata_rule_ir_decoder.py",
        "runner": "scripts/run_pg119_metadata_rule_ir_training.py",
        "pg116_dataset": "research/pg116_multisource_training_dataset_v1.json",
        "pg118_dataset": "research/pg118_transition_training_dataset_v1.json",
        "pg117_report": "research/pg117_double_holdout_report_v1.json",
        "pg114_report": "research/pg114_family_holdout_replay_report_v1.json",
    }.items():
        assert _sha256_file(relative_path) == report["source"][key]
    assert (ROOT / "artifacts/pg119-metadata-rule-ir-decoder-v1/model.pt").exists()
