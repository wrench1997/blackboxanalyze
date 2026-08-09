import json
from pathlib import Path


def _load(name: str):
    return json.loads((Path("research") / name).read_text(encoding="utf-8"))


def test_pg167_runs_three_seeds_and_blocks_collided_ood_claim():
    report = _load("pg167_multiseed_surface_ood_report_v1.json")
    assert report["status"] == "completed_pg167_multiseed_surface_ood"
    assert len(report["seed_results"]) == 3
    assert report["split"]["seen_surface_row_count"] == 20
    assert report["split"]["unseen_surface_row_count"] == 8
    assert report["split"]["projection_overlap_count"] == 2
    assert report["split"]["unseen_novel_projection_count"] == 3
    assert report["split"]["projection_ood_informative"] is False
    assert report["interpretation"]["vulnerability_claim_allowed"] is False
    assert report["interpretation"]["promotion_allowed"] is False
    assert report["aggregate"]["base_holdout_perplexity"]["std"] < 0.01


def test_pg167_registry_preserves_the_projection_collision_as_a_negative_result():
    registry = _load("pg_pk_24_cross_lab_registry_v1.json")
    entry = next(item for item in registry["targets"] if item["target_id"] == "pg167_multiseed_surface_ood")
    assert registry["training_eligible_target_count"] == 40
    assert registry["evaluation_only_target_count"] == 116
    assert entry["training_eligible"] is True
    assert entry["training_role"] == "multiseed_ood_diagnostic"
    assert entry["projection_ood_informative"] is False
    assert entry["vulnerability_claim_allowed"] is False
    assert entry["training_artifact_promotion_allowed"] is False
    assert entry["memory_promotion_allowed"] is False
