from __future__ import annotations

import json

from scripts.build_pg387_ctf_frontend_context_dataset import build_dataset
from scripts.run_pg387_ctf_context_token_candidate import run_candidate


def test_pg387_token_candidate_plan_is_closed_and_does_not_start_optimizer() -> None:
    report = run_candidate()
    assert report["status"] == "plan_only"
    assert report["gaps"]["blocked"] is False
    assert report["execution"]["optimizer_started"] is False
    assert report["train_only_vocabulary"]["scope"] == "train_context_only"
    assert report["promotion"]["training_allowed"] is False


def test_pg387_token_candidate_cpu_smoke_is_representation_only() -> None:
    report = run_candidate(cpu_smoke=True, row_limit=32, d_model=32, n_layers=1, experts=2, expert_hidden=64, max_length=48, microbatch=8, seeds=(38701,))
    assert report["status"] == "cpu_smoke_candidate_only"
    assert report["execution"]["optimizer_started"] is True
    assert report["execution"]["gpu_touched"] is False
    assert report["capability_training_allowed"] is False
    assert len(report["seeds"]) == 1
    assert report["seeds"][0]["holdout"]["negative_false_allow"] >= 0


def test_pg387_token_candidate_blocks_holdout_unknown_before_optimizer(tmp_path) -> None:
    dataset = build_dataset()
    holdout = next(row for row in dataset["rows"] if row["split"] == "implementation_holdout")
    holdout["context_tokens"].append("js_sink=holdout_only_unseen")
    path = tmp_path / "gap.json"
    path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
    report = run_candidate(dataset_path=path, cpu_smoke=True, row_limit=16)
    assert report["status"] == "blocked_train_only_vocab_gap"
    assert report["gaps"]["blocked"] is True
    assert report["execution"]["optimizer_started"] is False
