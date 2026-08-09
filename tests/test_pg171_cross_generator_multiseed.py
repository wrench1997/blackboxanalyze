import json
from pathlib import Path


def _load(name: str):
    return json.loads((Path("research") / name).read_text(encoding="utf-8"))


def test_pg171_three_seed_cross_generator_stability_passes():
    report = _load("pg171_cross_generator_multiseed_report_v1.json")
    assert report["status"] == "completed_pg171_cross_generator_multiseed"
    assert len(report["seed_results"]) == 3
    assert report["dataset"]["projection_overlap_dev_ood"] == 0
    assert report["dataset"]["projection_overlap_generator_prior"] == 0
    assert report["stability"]["old_capability_thresholds_pass"] is True
    assert report["stability"]["ood_stability_pass"] is True
    assert report["aggregate"]["generator_ood_perplexity"]["mean"] == 2.48585746
    assert report["aggregate"]["generator_ood_perplexity"]["std"] == 0.02675335
    assert report["interpretation"]["vulnerability_claim_allowed"] is False
    assert report["interpretation"]["promotion_allowed"] is False


def test_pg171_registry_keeps_multiseed_result_diagnostic_only():
    registry = _load("pg_pk_24_cross_lab_registry_v1.json")
    entry = next(item for item in registry["targets"] if item["target_id"] == "pg171_cross_generator_multiseed")
    assert registry["training_eligible_target_count"] == 40
    assert registry["evaluation_only_target_count"] == 116
    assert entry["training_eligible"] is True
    assert entry["training_role"] == "cross_generator_multiseed_stability_diagnostic"
    assert entry["old_capability_thresholds_pass"] is True
    assert entry["ood_stability_pass"] is True
    assert entry["vulnerability_claim_allowed"] is False
    assert entry["training_artifact_promotion_allowed"] is False
    assert entry["memory_promotion_allowed"] is False
