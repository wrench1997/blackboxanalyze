import json
from pathlib import Path


def _load(name: str):
    return json.loads((Path("research") / name).read_text(encoding="utf-8"))


def test_pg163_mixes_bounded_typed_tokens_without_raw_oracle_fields():
    report = _load("pg163_large_typed_mix_report_v1.json")
    dataset = _load("pg163_large_typed_mix_dataset_v1.json")
    collection = report["fresh_typed_collection"]
    assert report["status"] == "completed_pg163_large_typed_mix"
    assert collection["evidence_hash_valid"] is True
    assert collection["evidence_hash_count"] == 720
    assert collection["get_step_count"] == collection["post_step_count"] == 360
    assert collection["typed_train_unique_count"] == 326
    assert collection["typed_holdout_unique_count"] == 203
    assert report["corpus"]["mixed_train_count"] == 9999
    assert report["corpus"]["vocabulary_size"] == 233
    assert dataset["raw_probe_strings_stored"] is False
    assert dataset["raw_response_bodies_stored"] is False
    assert dataset["oracle_labels_in_tokens"] is False
    assert dataset["target_identity_in_tokens"] is False
    assert dataset["family_labels_in_tokens"] is False
    token_text = json.dumps(dataset["train_rows"][:30] + dataset["typed_holdout_rows"][:30], ensure_ascii=False).lower()
    assert "evaluator" not in token_text
    assert "payload" not in token_text


def test_pg163_large_and_xl_are_real_capacity_runs_and_not_promoted():
    report = _load("pg163_large_typed_mix_report_v1.json")
    variants = report["variants"]
    assert set(variants) == {"large_typed_mix", "xl_typed_mix"}
    assert variants["large_typed_mix"]["parameter_count"] == 19219689
    assert variants["xl_typed_mix"]["parameter_count"] == 57160937
    assert variants["large_typed_mix"]["typed_holdout"]["next_token_accuracy"] == 0.81811121
    assert variants["xl_typed_mix"]["typed_holdout"]["next_token_accuracy"] == 0.8286735
    assert variants["large_typed_mix"]["base_holdout"]["perplexity"] < 2.7
    assert variants["xl_typed_mix"]["base_holdout"]["perplexity"] < 2.7
    assert report["selection"]["promotion_allowed"] is False


def test_pg163_registry_keeps_large_training_artifacts_evaluation_only():
    registry = _load("pg_pk_24_cross_lab_registry_v1.json")
    entry = next(item for item in registry["targets"] if item["target_id"] == "pg163_large_typed_mix")
    assert registry["training_eligible_target_count"] == 40
    assert registry["evaluation_only_target_count"] == 116
    assert entry["training_eligible"] is True
    assert entry["training_artifact_promotion_allowed"] is False
    assert entry["memory_promotion_allowed"] is False
    assert entry["parameter_counts"]["xl_typed_mix"] == 57160937
