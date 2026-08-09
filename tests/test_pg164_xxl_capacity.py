import json
from pathlib import Path


def _load(name: str):
    return json.loads((Path("research") / name).read_text(encoding="utf-8"))


def test_pg164_trains_a_100m_class_model_on_the_frozen_pg163_mix():
    report = _load("pg164_xxl_capacity_report_v1.json")
    result = report["result"]
    assert report["status"] == "completed_pg164_xxl_capacity"
    assert result["parameter_count"] == 101380329
    assert result["config"]["d_model"] == 1024
    assert result["config"]["layers"] == 8
    assert result["config"]["gradient_accumulation_steps"] == 2
    assert result["base_holdout"]["perplexity"] == 2.5717967
    assert result["typed_holdout"]["perplexity"] == 1.45351808
    assert result["typed_holdout"]["next_token_accuracy"] == 0.83348866
    assert report["promotion"]["training_artifact_promotion_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg164_registry_is_capacity_evidence_not_capability_promotion():
    registry = _load("pg_pk_24_cross_lab_registry_v1.json")
    entry = next(item for item in registry["targets"] if item["target_id"] == "pg164_xxl_capacity")
    assert registry["training_eligible_target_count"] == 40
    assert registry["evaluation_only_target_count"] == 116
    assert entry["training_eligible"] is True
    assert entry["parameter_count"] == 101380329
    assert entry["training_artifact_promotion_allowed"] is False
    assert entry["memory_promotion_allowed"] is False
