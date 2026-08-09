import hashlib
import json
from pathlib import Path

from app.pg121_shape_sanitized_rule_ir_decoder import FEATURE_DIM, model_input_feature_vector, shape_hash_slots_zeroed


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def _sha256_file(relative_path: str) -> str:
    return hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()


def test_pg121_repairs_pg120_unknown_abstain_without_capacity_growth():
    report = _load("pg121_shape_sanitized_training_report_v1.json")
    assert report["status"] == "completed_pg121_shape_sanitized_training"
    assert report["scope"]["feature_dim"] == 48
    assert report["scope"]["parameter_count"] == 4996
    assert report["scope"]["capacity_unchanged"] is True
    assert report["training"]["dev_metrics"]["accuracy"] == 0.992424
    pg120 = report["blind_pg120"]
    assert pg120["metadata_positive_recall"] == 1.0
    assert pg120["decoy_false_accept_count"] == 0
    assert pg120["blind_oracle_abstain_rate"] == 1.0
    assert pg120["cross_seed"]["positive_recall_variance"] == 0.0
    assert report["previous_pg120_unknown_abstain_rate"] == 0.0
    assert all(report["checks"].values())
    assert report["promotion"]["training_artifact_promotion_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg121_feature_projection_zeroes_shape_hash_buckets():
    model_input = {
        "action_manifest": {"method": "GET", "placement": "query", "encoding_chain": ["url_percent"], "safety": {}},
        "baseline_projection": {"status_class": "2xx", "body_length_bucket": "256-4095"},
        "response_projection": {"candidate_signal": True, "shape_changed": False, "policy_header_changed": False, "location_changed": False, "metadata_changed": False, "transition_delta": "none", "shape_class": "transition-v4", "status_class": "2xx", "noise_bucket": 2},
        "belief_before": {"effect": 0.2, "input_only": 0.3, "no_effect": 0.3, "unknown": 0.2},
    }
    vector = model_input_feature_vector(model_input)
    assert len(vector) == FEATURE_DIM
    assert shape_hash_slots_zeroed(vector) is True
    assert vector[36:40] == [0.0, 0.0, 0.0, 0.0]


def test_pg121_dataset_protocol_rules_and_sources_are_consistent():
    dataset = _load("pg121_shape_sanitized_training_dataset_v1.json")
    assert dataset["source_dataset"] == "pg119_metadata_training_dataset_v1.json"
    assert dataset["shape_hash_slots_zeroed"] is True
    assert dataset["feature_dim"] == 48
    assert dataset["memory_promotion_allowed"] is False
    assert len(dataset["train_rows"]) == 528
    assert len(dataset["dev_rows"]) == 528
    protocol = _load("pg121_shape_sanitized_rule_ir_training_protocol_v1.json")
    assert protocol["model_contract"]["capacity_unchanged"] is True
    assert protocol["model_contract"]["shape_hash_slots_zeroed"] is True
    assert protocol["action_contract"]["fresh_reset_per_action"] is True
    rules = _load("improvement_rules.json")
    policy = rules["pg121_shape_sanitized_rule_ir_training_policy"]
    assert policy["shape_hash_slots_zeroed"] is True
    assert policy["capacity_unchanged"] is True
    assert policy["pg120_unknown_abstain_rate"] == 1.0
    assert policy["training_artifact_promotion_allowed"] is False
    report = _load("pg121_shape_sanitized_training_report_v1.json")
    for key, relative_path in {
        "decoder": "app/pg121_shape_sanitized_rule_ir_decoder.py",
        "runner": "scripts/run_pg121_shape_sanitized_training.py",
        "training_dataset": "research/pg119_metadata_training_dataset_v1.json",
        "pg120_report": "research/pg120_cross_impl_holdout_report_v1.json",
    }.items():
        assert _sha256_file(relative_path) == report["source"][key]
    assert (ROOT / "artifacts/pg121-shape-sanitized-rule-ir-decoder-v1/model.pt").exists()
