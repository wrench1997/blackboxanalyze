"""Focused tests for the PG-381 abstract composition candidate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_pg381_abstract_composition_candidate import run_candidate_report


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "research" / "pg380_abstract_adversarial_reasoning_dataset_v1.json"


def test_cpu_smoke_trains_composition_decoder_without_raw_or_wire_flags() -> None:
    report = run_candidate_report(
        dataset_path=DATASET,
        device="cpu",
        pretrain_epochs=1,
        posttrain_epochs=1,
        microbatch=2,
        d_model=16,
        n_layers=1,
        experts=2,
        expert_hidden=32,
        slot_decoder_layers=1,
        slot_decoder_heads=2,
        max_length=128,
        row_limit=4,
        checkpoint_dir=None,
    )
    assert report["status"] == "cpu_smoke_candidate_only"
    assert report["training"]["target_tokens_read_for_evaluator"] is False
    assert report["safety"]["raw_payload_in_context"] is False
    assert report["scientific_gate"]["abstract_reasoning_only"] is True
    assert report["worst_seed"]["slot_composition_exact_min"] >= 0.0
    assert all(value is False for value in report["promotion"].values())


def test_remote_candidate_requires_explicit_weekend_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BLACKBOX_REMOTE_A800_TRAIN", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match="remote A800 gate"):
        run_candidate_report(dataset_path=DATASET, device="cuda:0", row_limit=1)


def test_plan_artifact_is_candidate_only(tmp_path: Path) -> None:
    output = tmp_path / "plan.json"
    report = {
        "schema_version": "pg381-abstract-composition-candidate-v1",
        "status": "plan_only",
        "execution": {"optimizer_started": False, "gpu_touched": False, "docker_started": False, "network_used": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    output.write_text(json.dumps(report), encoding="utf-8")
    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert parsed["status"] == "plan_only"
    assert parsed["execution"]["optimizer_started"] is False
    assert all(value is False for value in parsed["promotion"].values())
