import json
from pathlib import Path


def _load(name: str):
    return json.loads((Path("research") / name).read_text(encoding="utf-8"))


def test_pg174_uses_full_holdout_and_does_not_promote_worse_routing():
    report = _load("pg174_full_holdout_routing_report_v1.json")
    assert report["status"] == "completed_pg174_full_holdout_routing"
    assert report["dataset"]["full_eval_counts"] == {"base_holdout": 1210, "typed_holdout": 203, "pg168_ood": 1000, "pg170_ood": 1000, "pg172_ood": 1000}
    assert report["selection"]["baseline_160m_aggregate_ppl"] == 2.51077335
    assert report["selection"]["best_variant"] == "source_routed"
    assert report["selection"]["best_variant_beats_160m_baseline"] is False
    assert report["selection"]["selected_variant"] is None
    assert report["selection"]["promotion_allowed"] is False
    assert report["scope"]["real_vulnerability_scanner_claim_allowed"] is False


def test_pg174_registry_records_the_negative_full_holdout_result():
    registry = _load("pg_pk_24_cross_lab_registry_v1.json")
    entry = next(item for item in registry["targets"] if item["target_id"] == "pg174_full_holdout_routing")
    assert registry["training_eligible_target_count"] == 40
    assert registry["evaluation_only_target_count"] == 116
    assert entry["training_eligible"] is True
    assert entry["training_role"] == "full_holdout_routing_negative_diagnostic"
    assert entry["selected_variant"] is None
    assert entry["full_holdout_used"] is True
    assert entry["vulnerability_claim_allowed"] is False
    assert entry["training_artifact_promotion_allowed"] is False
    assert entry["memory_promotion_allowed"] is False
