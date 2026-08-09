import json
from pathlib import Path

from app.dataset_utility_audit import audit_dataset


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg31_is_explicitly_schema_only_and_not_training_data():
    report = audit_dataset(_load("pg_pk_31_ood_rule_ir_evaluation_manifest_v1.json"), dataset_id="pg31")
    assert report["utility_class"] == "schema_only"
    assert report["methods"] == []
    assert report["typed_positive_count"] == 0
    assert report["training_allowed"] is False
    assert "no_runtime_replay" in report["blockers"]


def test_pg28_has_real_get_post_context_but_is_negative_only():
    report = audit_dataset(_load("pg_pk_28_get_post_dual_catalog_v1.json"), dataset_id="pg28")
    assert report["get_post_complete"] is True
    assert report["runtime_replay"] is True
    assert report["typed_positive_count"] == 0
    assert report["utility_class"] == "negative_only_replay_evaluation"
    assert report["training_allowed"] is False


def test_pg33_contains_typed_dual_channel_replay_but_stays_behind_capability_gate():
    report = audit_dataset(
        _load("pg_pk_33_get_post_typed_replay_catalog_v1.json"), dataset_id="pg33"
    )
    assert report["row_count"] == 84
    assert report["runtime_replay"] is True
    assert report["get_post_complete"] is True
    assert report["typed_positive_count"] == 36
    assert report["negative_control_pair_count"] == 36
    assert report["complete_context_rows"] == 84
    assert report["fresh_reset_count"] == 84
    assert report["evidence_hash_count"] == 84
    assert report["utility_class"] == "replay_training_candidate_pending_capability_gate"
    assert report["training_allowed"] is False
    assert report["memory_promotion_allowed"] is False
    assert "dataset_marked_evaluation_only" in report["blockers"]


def test_pg34_independent_fixture_is_replay_candidate_not_schema_or_negative_only():
    report = audit_dataset(
        _load("pg34_independent_fixture_catalog_v1.json"), dataset_id="pg34_independent"
    )
    assert report["row_count"] == 108
    assert report["runtime_replay"] is True
    assert report["get_post_complete"] is True
    assert report["typed_positive_count"] == 48
    assert report["negative_control_pair_count"] == 48
    assert report["complete_context_rows"] == 108
    assert report["utility_class"] == "replay_training_candidate_pending_capability_gate"
    assert report["training_allowed"] is False
    assert report["memory_promotion_allowed"] is False


def test_pg35_encoding_pair_fixture_has_full_context_but_stays_behind_gate():
    report = audit_dataset(
        _load("pg35_independent_fixture_catalog_v1.json"), dataset_id="pg35_independent"
    )
    assert report["row_count"] == 648
    assert report["runtime_replay"] is True
    assert report["get_post_complete"] is True
    assert report["typed_positive_count"] == 288
    assert report["negative_control_pair_count"] == 288
    assert report["complete_context_rows"] == 648
    assert report["fresh_reset_count"] == 648
    assert report["evidence_hash_count"] == 648
    assert report["utility_class"] == "replay_training_candidate_pending_capability_gate"
    assert report["training_allowed"] is False
    assert report["memory_promotion_allowed"] is False
    assert "dataset_marked_evaluation_only" in report["blockers"]


def test_pg36_delayed_maze_has_full_get_post_context_but_stays_behind_gate():
    report = audit_dataset(
        _load("pg36_independent_maze_catalog_v1.json"), dataset_id="pg36_independent_maze"
    )
    assert report["row_count"] == 960
    assert report["runtime_replay"] is True
    assert report["get_post_complete"] is True
    assert report["typed_positive_count"] == 96
    assert report["negative_control_pair_count"] == 96
    assert report["complete_context_rows"] == 960
    assert report["fresh_reset_count"] == 960
    assert report["evidence_hash_count"] == 960
    assert report["utility_class"] == "replay_training_candidate_pending_capability_gate"


def test_pg37_counterfactual_has_full_get_post_context_but_stays_behind_gate():
    report = audit_dataset(
        _load("pg37_counterfactual_catalog_v1.json"), dataset_id="pg37_counterfactual"
    )
    assert report["row_count"] == 2880
    assert report["runtime_replay"] is True
    assert report["get_post_complete"] is True
    assert report["typed_positive_count"] == 288
    assert report["negative_control_pair_count"] == 288
    assert report["fresh_reset_count"] == 2880
    assert report["evidence_hash_count"] == 2880
    assert report["training_allowed"] is False
    assert report["memory_promotion_allowed"] is False
    assert report["utility_class"] == "replay_training_candidate_pending_capability_gate"


def test_pg40_semantic_router_has_full_get_post_context_but_stays_behind_gate():
    report = audit_dataset(
        _load("pg40_semantic_router_catalog_v1.json"), dataset_id="pg40_semantic_router"
    )
    assert report["row_count"] == 960
    assert report["runtime_replay"] is True
    assert report["get_post_complete"] is True
    assert report["typed_positive_count"] == 96
    assert report["negative_control_pair_count"] == 96
    assert report["fresh_reset_count"] == 960
    assert report["evidence_hash_count"] == 960
    assert report["training_allowed"] is False
    assert report["memory_promotion_allowed"] is False


def test_pg42_independent_semantic_has_full_context_but_stays_behind_ood_gate():
    report = audit_dataset(
        _load("pg42_independent_semantic_catalog_v1.json"), dataset_id="pg42_independent_semantic"
    )
    assert report["row_count"] == 2880
    assert report["runtime_replay"] is True
    assert report["get_post_complete"] is True
    assert report["typed_positive_count"] == 324
    assert report["negative_control_pair_count"] == 324
    assert report["complete_context_rows"] == 2880
    assert report["fresh_reset_count"] == 2880
    assert report["evidence_hash_count"] == 2880
    assert report["training_allowed"] is False
    assert report["memory_promotion_allowed"] is False
    assert report["utility_class"] == "replay_training_candidate_pending_capability_gate"
    assert report["utility_class"] == "replay_training_candidate_pending_capability_gate"
    assert report["training_allowed"] is False
    assert report["memory_promotion_allowed"] is False


def test_pg48_compositional_preprobe_has_dual_channel_context_but_stays_quarantined():
    report = audit_dataset(
        _load("pg48_compositional_preprobe_catalog_v1.json"), dataset_id="pg48_compositional_preprobe"
    )
    assert report["row_count"] == 1536
    assert report["runtime_replay"] is True
    assert report["get_post_complete"] is True
    assert report["typed_positive_count"] == 84
    assert report["negative_control_pair_count"] == 84
    assert report["complete_context_rows"] == 1536
    assert report["fresh_reset_count"] == 1536
    assert report["evidence_hash_count"] == 1536
    assert report["utility_class"] == "replay_training_candidate_pending_capability_gate"
    assert report["training_allowed"] is False
    assert report["memory_promotion_allowed"] is False
    assert "dataset_marked_evaluation_only" in report["blockers"]


def test_pg50_stability_matrix_has_full_context_but_stays_behind_matrix_gate():
    report = audit_dataset(
        _load("pg50_stability_matrix_catalog_v1.json"), dataset_id="pg50_stability_matrix"
    )
    assert report["row_count"] == 7200
    assert report["runtime_replay"] is True
    assert report["get_post_complete"] is True
    assert report["typed_positive_count"] == 405
    assert report["negative_control_pair_count"] == 405
    assert report["complete_context_rows"] == 7200
    assert report["fresh_reset_count"] == 7200
    assert report["evidence_hash_count"] == 7200
    assert report["utility_class"] == "replay_training_candidate_pending_capability_gate"
    assert report["training_allowed"] is False
    assert report["memory_promotion_allowed"] is False
    assert "dataset_marked_evaluation_only" in report["blockers"]


def test_pg51_real_docker_shadow_is_negative_only_until_authoritative_oracle_exists():
    report = audit_dataset(
        _load("pg51_pikachu_docker_dual_channel_catalog_v1.json"), dataset_id="pg51_pikachu_docker_dual_channel"
    )
    assert report["row_count"] == 28
    assert report["runtime_replay"] is True
    assert report["get_post_complete"] is True
    assert report["typed_positive_count"] == 0
    assert report["complete_context_rows"] == 28
    assert report["fresh_reset_count"] == 28
    assert report["evidence_hash_count"] == 28
    assert report["utility_class"] == "negative_only_replay_evaluation"
    assert report["training_allowed"] is False
    assert report["memory_promotion_allowed"] is False
    assert "no_typed_positive" in report["blockers"]
