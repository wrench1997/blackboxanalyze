from __future__ import annotations

import json

from app.research_ops import _pg373_staged_candidate_projection, build_research_ops_snapshot


def _report() -> dict:
    return {
        "schema_version": "pg373-staged-pretrain-multitask-candidate-v1",
        "status": "remote_candidate_only",
        "training": {
            "device": "cuda:0",
            "seeds": [37301, 37302, 37303],
            "pretrain_epochs": 4,
            "posttrain_epochs": 4,
            "microbatch": 16,
            "baseline_kind": "train_only_next_token_pretrain",
            "required_context_window": 621,
            "vocabulary_size": 877,
            "config": {"d_model": 256, "n_layers": 4, "experts": 4, "expert_hidden": 512, "max_length": 768},
        },
        "candidates": [
            {
                "seed": 37301,
                "baseline": {"rows": 1477, "token_accuracy": 0.6, "sequence_exact": 0.0, "slot_accuracy": 0.3, "predictive_entropy": 2.8},
                "post": {"rows": 1477, "token_accuracy": 0.62, "sequence_exact": 0.001, "slot_accuracy": 0.7, "predictive_entropy": 2.3},
                "entropy_relative_drop": 0.2,
                "checkpoint": {"path": "artifacts/secret.pt", "sha256": "a" * 64},
            }
        ],
        "worst_seed": {
            "sequence_exact_min": 0.001,
            "slot_accuracy_min": 0.7,
            "ask_recall_min": 1.0,
            "repair_recall_min": 1.0,
            "positive_recall_min": 1.0,
            "negative_false_allow_max": 0,
            "entropy_relative_drop_max": 0.2,
        },
        "scientific_gate": {"trained_baseline_entropy_comparison": True, "typed_live_replay_with_model_selected_wire": False},
        "locks": {"runner_sha256": "b" * 64, "rules_sha256": "c" * 64, "datasets": {"pg362": "d" * 64}},
        "promotion": {"training_allowed": True},
        "report_sha256": "e" * 64,
    }


def test_pg373_projection_keeps_staged_baseline_and_blocks_promotion() -> None:
    projected = _pg373_staged_candidate_projection(_report(), report_present=True)
    assert projected["artifact_status"] == "candidate_only"
    assert projected["baseline_kind"] == "train_only_next_token_pretrain"
    assert projected["scientific_gate"]["trained_baseline_entropy_comparison"] is True
    assert projected["promotion"]["training_allowed"] is False
    assert projected["raw_material_available"] is False
    encoded = json.dumps(projected, ensure_ascii=False)
    assert "artifacts/secret.pt" not in encoded
    assert "candidates" not in projected


def test_pg373_missing_report_is_pending_and_closed() -> None:
    projected = _pg373_staged_candidate_projection({}, report_present=False)
    assert projected["artifact_status"] == "pending"
    assert projected["promotion_blocked"] is True
    assert projected["vulnerability_claim_allowed"] is False


def test_pg373_snapshot_projection_is_present_and_bounded() -> None:
    snapshot = build_research_ops_snapshot()
    model = snapshot["capability"]["model"]["pg373_staged_pretrain_candidate"]
    assert model["artifact_status"] == "candidate_only"
    assert model["training_eligible"] is False
    assert model["memory_promotion_allowed"] is False
    assert model["payload_catalog_promotion_allowed"] is False
    assert model["vulnerability_claim_allowed"] is False
    encoded = json.dumps(model, ensure_ascii=False)
    assert "artifacts/pg373" not in encoded
    assert "wire=" not in encoded
    assert "payload=" not in encoded
