import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pg233_capacity_sweep_does_not_mask_double_holdout_failure() -> None:
    report = json.loads((ROOT / "research" / "pg233_cross_family_capacity_training_report_v1.json").read_text(encoding="utf-8-sig"))
    dataset = json.loads((ROOT / "research" / "pg233_cross_family_capacity_dataset_v1.json").read_text(encoding="utf-8-sig"))
    assert report["status"] == "completed_cross_family_capacity_double_holdout"
    assert report["counts"]["raw_new_samples"] == 28
    assert report["counts"]["unique_records"] == 115
    assert report["counts"]["usable_records"] == 73
    assert report["counts"]["train_rows"] == 59
    assert report["counts"]["source_holdout_rows"] == 13
    assert report["counts"]["family_holdout_rows"] == 5
    assert report["counts"]["double_holdout_rows"] == 4
    assert [variant["hidden_dim"] for variant in report["variants"]] == [64, 128, 256, 512]
    assert report["strict_source_family_double_holdout_pass"] is False
    assert report["selected"]["metrics"]["double_holdout"]["lane_accuracy"] == 0.0
    assert report["selected"]["metrics"]["double_holdout"]["repair_accuracy"] == 0.0
    assert report["frozen_body_changed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert dataset["contract"]["raw_payload_strings_stored"] is False
    assert dataset["contract"]["raw_response_bodies_stored"] is False
    assert dataset["contract"]["diagnosis_targets_not_features"] is True

