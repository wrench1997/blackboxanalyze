import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def _sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def test_pg116_multisource_training_and_blind_gate_passes():
    report = _load("pg116_multisource_trace_training_report_v1.json")
    assert report["status"] == "completed_pg116_multisource_trace_training"
    assert report["scope"]["parameter_count"] == 4612
    assert report["scope"]["device"] in {"cuda", "cpu"}
    collection = report["collection"]
    assert collection["source_set"] == ["alpha", "beta"]
    assert collection["target_instance_count"] == 12
    assert collection["episode_count"] == 48
    assert collection["step_count"] == 192
    assert collection["get_step_count"] == 96
    assert collection["post_step_count"] == 96
    assert collection["fresh_reset_per_step"] is True
    assert collection["evidence_hash_valid"] is True
    blind = report["blind_pg114"]
    assert blind["step_metrics"]["accuracy"] >= 0.90
    assert blind["family_holdout_confirm_recall"] == 1.0
    assert blind["decoy_false_accept_count"] == 0
    assert blind["withheld_oracle_abstain_rate"] == 1.0
    assert all(report["checks"].values())
    assert report["promotion"]["training_artifact_promotion_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert (ROOT / "artifacts" / "pg116-multisource-rule-ir-decoder-v1" / "model.pt").exists()


def test_pg116_trace_is_get_post_paired_fresh_and_model_input_blind():
    trace = _load("pg116_multisource_trace_v1.json")
    assert trace["training_eligible"] is True
    assert trace["memory_promotion_allowed"] is False
    assert trace["raw_probe_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
    episodes = [episode for source in trace["sources"] for episode in source["episodes"]]
    assert len(episodes) == 48
    for episode in episodes:
        assert episode["episode_report"]["status"] == "accepted_evaluation"
        assert episode["negative_control_pair_clear"] is True
        assert [step["action_manifest"]["method"] for step in episode["steps"]] == ["GET", "GET", "POST", "POST"]
        assert len({step["fresh_reset"]["reset_epoch"] for step in episode["steps"]}) == 4
        for step in episode["steps"]:
            model_text = json.dumps(step["model_input"], ensure_ascii=False).casefold()
            for forbidden in ("oracle", "positive_authority", "family", "target_instance_id", "probe_ref", "probe_sha256"):
                assert forbidden not in model_text
            assert step["online_weight_update"] is False
            assert step["long_term_memory_write"] is False
    for episode in episodes:
        for record in episode["evidence_records"]:
            body = dict(record)
            declared = body.pop("evidence_hash")
            assert declared == _sha256_json(body)
            target_evidence = record["oracle_projection"]["source_evidence_sha256"]
            assert len(target_evidence) == 64


def test_pg116_training_dataset_is_source_and_seed_disjoint_with_balanced_replay():
    dataset = _load("pg116_multisource_training_dataset_v1.json")
    assert dataset["training_eligible"] is True
    assert dataset["memory_promotion_allowed"] is False
    assert dataset["pg114_excluded_from_training"] is True
    assert dataset["model_input_family_free"] is True
    assert dataset["model_input_oracle_blind"] is True
    assert dataset["train_unique_row_count"] == 96
    assert dataset["dev_unique_row_count"] == 96
    assert len(dataset["train_rows"]) == len(dataset["dev_rows"]) == 264
    assert dataset["train_class_unique"] == {"confirmed_positive": 6, "confirmed_negative": 66, "candidate": 18, "abstain": 6}
    assert dataset["dev_class_unique"] == {"confirmed_positive": 6, "confirmed_negative": 66, "candidate": 18, "abstain": 6}
    train_keys = {(row["source"], row["target_seed"]) for row in dataset["train_rows"]}
    dev_keys = {(row["source"], row["target_seed"]) for row in dataset["dev_rows"]}
    assert train_keys.isdisjoint(dev_keys)
    for row in dataset["train_rows"] + dataset["dev_rows"]:
        model_text = json.dumps(row["model_input"], ensure_ascii=False).casefold()
        assert "oracle" not in model_text
        assert "family" not in model_text
        assert "target_instance_id" not in model_text
        assert row["memory_promotion_allowed"] is False


def test_pg116_protocol_and_source_hashes_are_persisted():
    protocol = _load("pg116_multisource_trace_training_protocol_v1.json")
    assert protocol["action_contract"]["methods"] == ["GET", "POST"]
    assert protocol["action_contract"]["fresh_reset_per_action"] is True
    assert protocol["model_contract"]["previous_checkpoint_reuse_forbidden"] is True
    assert protocol["model_contract"]["pg114_excluded_from_training"] is True
    assert protocol["promotion"]["memory_promotion_allowed"] is False
    report = _load("pg116_multisource_trace_training_report_v1.json")
    for key, relative_path in {
        "target": "app/pg116_multisource_training_target.py",
        "bridge": "app/pg116_multisource_replay.py",
        "runner": "scripts/run_pg116_multisource_trace_training.py",
        "pg114_report": "research/pg114_family_holdout_replay_report_v1.json",
    }.items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == report["source"][key]

    rules = _load("improvement_rules.json")
    policy = rules["pg116_multisource_trace_training_policy"]
    assert policy["source_count"] == 2
    assert policy["target_instance_count"] == 12
    assert policy["step_count"] == 192
    assert policy["fresh_reset_per_step"] is True
    assert policy["model_input_oracle_blind"] is True
    assert policy["model_input_family_free"] is True
    assert policy["training_artifact_promotion_allowed"] is False
    assert policy["memory_promotion_allowed"] is False
