import json
from pathlib import Path


def _load(name: str):
    return json.loads((Path("research") / name).read_text(encoding="utf-8"))


def test_pg166_replay_anchor_reduces_surface_adaptation_forgetting():
    report = _load("pg166_surface_replay_adaptation_report_v1.json")
    assert report["status"] == "completed_pg166_surface_replay_adaptation"
    assert report["dataset"]["base_train_count"] == 9999
    assert report["dataset"]["surface_row_count"] == 28
    assert report["scope"]["real_vulnerability_scanner_claim_allowed"] is False
    assert report["baseline"]["base_holdout"]["perplexity"] == 2.5717967
    assert report["baseline"]["typed_holdout"]["perplexity"] == 1.45351808
    replay = report["variants"]["replay_anchored"]
    surface_only = report["variants"]["surface_only"]
    assert replay["base_holdout"]["perplexity"] == 2.55618482
    assert replay["typed_holdout"]["perplexity"] == 1.43844417
    assert replay["surface_rows"]["perplexity"] == 1.56505161
    assert surface_only["base_holdout"]["perplexity"] == 2.64894756
    assert surface_only["typed_holdout"]["perplexity"] == 1.71369749
    assert surface_only["surface_rows"]["perplexity"] == 3.11116081
    assert replay["base_holdout"]["perplexity"] < surface_only["base_holdout"]["perplexity"]
    assert replay["typed_holdout"]["perplexity"] < surface_only["typed_holdout"]["perplexity"]
    assert report["interpretation"]["promotion_allowed"] is False


def test_pg166_registry_is_a_forgetting_diagnostic_only():
    registry = _load("pg_pk_24_cross_lab_registry_v1.json")
    entry = next(item for item in registry["targets"] if item["target_id"] == "pg166_surface_replay_adaptation")
    assert registry["training_eligible_target_count"] == 40
    assert registry["evaluation_only_target_count"] == 116
    assert entry["training_eligible"] is True
    assert entry["training_role"] == "replay_forgetting_diagnostic"
    assert entry["vulnerability_claim_allowed"] is False
    assert entry["training_artifact_promotion_allowed"] is False
    assert entry["memory_promotion_allowed"] is False
