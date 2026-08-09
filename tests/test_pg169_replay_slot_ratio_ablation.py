import json
from pathlib import Path


def _load(name: str):
    return json.loads((Path("research") / name).read_text(encoding="utf-8"))


def test_pg169_selects_only_ratio_that_preserves_old_capability_thresholds():
    report = _load("pg169_replay_slot_ratio_report_v1.json")
    assert report["status"] == "completed_pg169_replay_slot_ratio_ablation"
    assert report["selection"]["feasible_variants"] == ["ratio_1_1"]
    assert report["selection"]["selected_variant"] == "ratio_1_1"
    assert report["variants"]["ratio_1_1"]["base_holdout"]["perplexity"] == 2.59412937
    assert report["variants"]["ratio_1_1"]["typed_holdout"]["perplexity"] == 1.52237979
    assert report["variants"]["ratio_1_1"]["slot_ood"]["perplexity"] == 2.30065363
    assert report["variants"]["ratio_1_4"]["typed_holdout"]["perplexity"] > 1.45351808 * 1.10
    assert report["selection"]["promotion_allowed"] is False


def test_pg169_registry_keeps_ratio_choice_as_diagnostic_only():
    registry = _load("pg_pk_24_cross_lab_registry_v1.json")
    entry = next(item for item in registry["targets"] if item["target_id"] == "pg169_replay_slot_ratio_ablation")
    assert registry["training_eligible_target_count"] == 40
    assert registry["evaluation_only_target_count"] == 116
    assert entry["training_eligible"] is True
    assert entry["training_role"] == "replay_slot_ratio_ablation"
    assert entry["selected_variant"] == "ratio_1_1"
    assert entry["vulnerability_claim_allowed"] is False
    assert entry["training_artifact_promotion_allowed"] is False
    assert entry["memory_promotion_allowed"] is False
