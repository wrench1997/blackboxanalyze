from __future__ import annotations

import json
from pathlib import Path

from app.pg123_authorization_rule_ir_decoder import FEATURE_DIM, model_input_feature_vector


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg123_authorization_slot_repairs_pg122_without_decoy_or_old_family_regression() -> None:
    report = _load("pg123_authorization_slot_training_report_v1.json")
    assert report["status"] == "completed_pg123_authorization_slot_training"
    assert report["scope"]["feature_dim"] == 52
    assert report["scope"]["parameter_count"] == 5188
    assert report["blind_pg122_authorization"]["positive_recall"] == 1.0
    assert report["blind_pg122_authorization"]["decoy_false_accept_count"] == 0
    assert report["blind_pg122_authorization"]["unknown_abstain_rate"] == 0.333333
    assert report["blind_pg122_authorization_slot_ablation"]["positive_recall"] == 0.0
    assert report["blind_pg120_metadata"]["positive_recall"] == 1.0
    assert report["blind_pg117_route"]["positive_recall"] == 1.0
    assert report["blind_pg114"]["family_holdout_confirm_recall"] == 1.0
    assert all(report["checks"].values())
    assert report["promotion"]["training_artifact_promotion_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg123_training_dataset_excludes_pg122_holdout_seeds() -> None:
    dataset = _load("pg123_authorization_slot_training_dataset_v1.json")
    assert dataset["feature_dim"] == FEATURE_DIM == 52
    assert dataset["training_eligible"] is True
    assert dataset["memory_promotion_allowed"] is False
    assert set(dataset["holdout_seeds_excluded"]) == {12201, 12203, 12205}
    seeds = {row.get("target_seed") for row in dataset["train_rows"] + dataset["dev_rows"] if row.get("target_seed") is not None}
    assert not seeds.intersection(dataset["holdout_seeds_excluded"])


def test_pg123_vector_has_four_authorization_slots() -> None:
    model_input = {
        "action_manifest": {"method": "POST", "placement": "json", "encoding_chain": ["html_entity"], "safety": {}},
        "baseline_projection": {"status_class": "2xx", "body_length_bucket": "256-4095"},
        "response_projection": {"candidate_signal": True, "shape_changed": False, "policy_header_changed": False, "location_changed": False, "metadata_changed": False, "authorization_changed": True, "transition_delta": "authorization", "shape_class": "decision-v5", "status_class": "2xx", "noise_bucket": 2},
        "belief_before": {"effect": 0.2, "input_only": 0.3, "no_effect": 0.3, "unknown": 0.2},
    }
    vector = model_input_feature_vector(model_input)
    assert len(vector) == FEATURE_DIM
    assert vector[-4:] == [1.0, 0.0, 0.0, 1.0]
