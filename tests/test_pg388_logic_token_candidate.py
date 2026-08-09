from __future__ import annotations

from pathlib import Path

from scripts.run_pg388_logic_token_candidate import run_candidate


def test_logic_token_candidate_plan_is_closed() -> None:
    report = run_candidate(dataset_path=Path("research/pg388_logic_invariant_dataset_v1.json"), cpu_smoke=False, row_limit=64)
    assert report["status"] == "plan_only"
    assert report["gaps"]["blocked"] is False
    assert report["execution"]["optimizer_started"] is False
    assert report["training_eligible"] == 0
    assert report["promotion"]["training_allowed"] is False


def test_logic_token_candidate_cpu_smoke_keeps_wire_closed() -> None:
    report = run_candidate(dataset_path=Path("research/pg388_logic_invariant_dataset_v1.json"), cpu_smoke=True, epochs=1, row_limit=32, d_model=32, n_layers=1, experts=2, expert_hidden=64, max_length=64, microbatch=8, seeds=(38801,))
    assert report["status"] == "cpu_smoke_candidate_only"
    assert report["execution"]["optimizer_started"] is True
    assert report["execution"]["device"] == "cpu"
    assert report["execution"]["gpu_touched"] is False
    assert report["execution"]["wire_created"] is False
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert report["seeds"][0]["holdout"]["negative_false_allow"] == 0
