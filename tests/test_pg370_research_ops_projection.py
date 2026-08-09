from __future__ import annotations

import json

from app.research_ops import _pg370_candidate_projection, build_research_ops_snapshot


def test_pg370_projection_is_bounded_and_blocks_entropy_failure() -> None:
    report = {
        "schema_version": "pg370-multitask-moe-candidate-v1",
        "status": "remote_candidate_only",
        "training": {
            "device": "cuda:0",
            "seeds": [37001, 37002, 37003],
            "vocabulary_scope": "declared_ontology_manifest",
            "vocabulary_size": 877,
            "required_context_window": 621,
            "vocabulary_gaps": {"blocked": False, "unknown_token_count": 0, "unknown_slot_value_count": 0},
        },
        "candidates": [{"seed": 37001, "checkpoint": {"sha256": "a" * 64}}],
        "worst_seed": {
            "sequence_exact_min": 0.02,
            "slot_accuracy_min": 0.72,
            "ask_recall_min": 1.0,
            "repair_recall_min": 1.0,
            "positive_recall_min": 1.0,
            "negative_false_allow_max": 0,
            "entropy_relative_drop_max": 0.80,
        },
        "locks": {"datasets": {"pg362": "b" * 64}},
        "promotion": {"training_allowed": False},
        "scientific_gate": {"typed_live_replay_with_model_selected_wire": False},
        "report_sha256": "c" * 64,
    }
    result = _pg370_candidate_projection(report, report_present=True)
    assert result["artifact_status"] == "candidate_only"
    assert result["worst_seed_metrics"]["sequence_exact_min"] == 0.02
    assert result["worst_seed_metrics"]["entropy_relative_drop_max"] == 0.8
    assert result["promotion"]["vulnerability_claim_allowed"] is False
    assert result["raw_material_available"] is False
    assert "candidates" not in result and "vocabulary_tokens" not in result


def test_pg370_projection_missing_report_is_pending() -> None:
    result = _pg370_candidate_projection({}, report_present=False)
    assert result["artifact_status"] == "pending"
    assert result["promotion"]["training_allowed"] is False
    assert result["raw_material_available"] is False


def test_pg370_snapshot_projection_is_closed_and_bounded() -> None:
    snapshot = build_research_ops_snapshot()
    model = snapshot["capability"]["model"]["pg370_multitask_moe_candidate"]
    assert model["promotion_blocked"] is True
    assert model["training_eligible"] is False
    assert model["memory_promotion_allowed"] is False
    assert model["payload_catalog_promotion_allowed"] is False
    assert model["vulnerability_claim_allowed"] is False
    assert model["raw_material_available"] is False
    encoded = json.dumps(model, ensure_ascii=False)
    assert "artifacts/pg370" not in encoded
    assert "https://" not in encoded
    assert "wire=" not in encoded
