from __future__ import annotations

import json

import pytest

from scripts.run_pg380_abstract_reasoning_sft import DEFAULT_DATASET, run_candidate_report


def test_cpu_smoke_runs_only_abstract_reasoning_and_closes_promotion() -> None:
    report = run_candidate_report(device="cpu", epochs=1, microbatch=2, d_model=16, n_layers=1, experts=2, expert_hidden=32, max_length=128, row_limit=8)
    assert report["status"] == "cpu_smoke_candidate_only"
    assert report["data"]["target_slots"] == 13
    assert report["scientific_gate"]["abstract_reasoning_only"] is True
    assert report["scientific_gate"]["model_selected_wire_replay"] is False
    assert all(value is False for value in report["promotion"].values())


def test_remote_lane_requires_explicit_flag_and_gpu0(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BLACKBOX_REMOTE_A800_TRAIN", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="remote A800 gate"):
        run_candidate_report(dataset_path=DEFAULT_DATASET, device="cuda:0", row_limit=2, d_model=16, n_layers=1, experts=2, expert_hidden=32, max_length=128)


def test_dataset_hash_and_report_are_json_safe() -> None:
    report = run_candidate_report(device="cpu", epochs=1, microbatch=1, d_model=16, n_layers=1, experts=2, expert_hidden=32, max_length=128, row_limit=2)
    json.dumps(report, ensure_ascii=False)
    assert len(report["dataset_sha256"]) == 64
