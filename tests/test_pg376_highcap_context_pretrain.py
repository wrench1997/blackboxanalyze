from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from app.pg295_causal_moe import CausalMoEConfig
from scripts.run_pg376_highcap_context_pretrain import (
    _device_gate,
    _safe_context_rows,
    main,
    run_candidate,
)


class _GuardedRecord(dict):
    """Fail if the context-only loader attempts to inspect target material."""

    def get(self, key, default=None):  # type: ignore[override]
        if key == "target_tokens":
            raise AssertionError("PG-376 context-only runner read target_tokens")
        return super().get(key, default)


def _dataset() -> dict[str, object]:
    firewall = {"forbidden_token_count": 0, "sidecars_off_context": True}
    return {
        "status": "candidate_only",
        "representation_pretrain_candidate_allowed": True,
        "capability_training_allowed": False,
        "vocabulary": {"context_tokens": ["axis=doc", "method=get", "method=post", "shape=alpha", "shape=beta"]},
        "records": [
            _GuardedRecord(
                split="train",
                context_tokens=["axis=doc", "method=get", "shape=alpha"],
                target_tokens=["forbidden_target_train"],
                context_firewall=firewall,
                raw_payload_stored=False,
                raw_response_body_stored=False,
                oracle_answer_in_context=False,
            ),
            _GuardedRecord(
                split="train",
                context_tokens=["axis=doc", "method=post", "shape=beta"],
                target_tokens=["forbidden_target_second"],
                context_firewall=firewall,
                raw_payload_stored=False,
                raw_response_body_stored=False,
                oracle_answer_in_context=False,
            ),
            _GuardedRecord(
                split="implementation_holdout",
                context_tokens=["axis=doc", "method=get", "shape=beta"],
                target_tokens=["forbidden_target_holdout"],
                context_firewall=firewall,
                raw_payload_stored=False,
                raw_response_body_stored=False,
                oracle_answer_in_context=False,
            ),
        ],
    }


def _write_artifacts(root: Path, dataset: dict[str, object]) -> tuple[Path, Path, Path]:
    dataset_path = root / "dataset.json"
    audit_path = root / "audit.json"
    rules_path = root / "rules.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    audit_path.write_text(
        json.dumps(
            {
                "status": "passed_candidate_audit",
                "counts": {
                    "active_cross_split_exact_overlap": 0,
                    "unknown_context_tokens": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    rules_path.write_text("{}", encoding="utf-8")
    return dataset_path, audit_path, rules_path


def test_pg376_loader_is_context_only() -> None:
    rows, failures, count = _safe_context_rows(_dataset(), split="train")
    assert failures == []
    assert count == 2
    assert rows == [
        {"context_tokens": ["axis=doc", "method=get", "shape=alpha"]},
        {"context_tokens": ["axis=doc", "method=post", "shape=beta"]},
    ]
    assert "forbidden_target_train" not in json.dumps(rows)


def test_pg376_cpu_smoke_is_highcap_configurable_and_closed(tmp_path: Path) -> None:
    dataset = _dataset()
    dataset_path, audit_path, rules_path = _write_artifacts(tmp_path, dataset)
    result = run_candidate(
        dataset=dataset,
        audit=json.loads(audit_path.read_text()),
        dataset_path=dataset_path,
        audit_path=audit_path,
        rules_path=rules_path,
        device="cpu",
        seeds=(37601,),
        epochs=1,
        batch_size=1,
        config=CausalMoEConfig(d_model=8, n_heads=4, n_layers=1, experts=2, expert_hidden=16, max_length=16),
    )
    assert result["status"] == "representation_pretrain_candidate_only"
    assert result["execution"]["gpu_touched"] is False
    assert result["training"]["holdout_used_for_optimization"] is False
    assert result["gate"]["target_tokens_read"] is False
    assert result["data"]["vocabulary_scope"] == "train_context_only"
    assert all(value is False for value in result["promotion"].values())
    assert result["candidates"][0]["implementation_holdout"]["rows"] == 1


def test_pg376_unknown_holdout_token_blocks_before_optimizer(tmp_path: Path) -> None:
    dataset = _dataset()
    dataset["records"][2]["context_tokens"] = ["axis=doc", "method=get", "holdout_only"]  # type: ignore[index]
    dataset_path, audit_path, rules_path = _write_artifacts(tmp_path, dataset)
    result = run_candidate(
        dataset=dataset,
        audit=json.loads(audit_path.read_text()),
        dataset_path=dataset_path,
        audit_path=audit_path,
        rules_path=rules_path,
        device="cpu",
        seeds=(37601,),
        config=CausalMoEConfig(d_model=8, n_heads=4, n_layers=1, experts=2, expert_hidden=16, max_length=16),
    )
    assert result["status"] == "blocked_representation_contract"
    assert result["gate"]["checks"]["holdout_vocabulary_closed"] is False
    assert result["execution"]["optimizer_started"] is False


def test_pg376_weekday_cuda_is_rejected_before_remote_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLACKBOX_REMOTE_A800_TRAIN", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="weekend-only"):
        _device_gate("cuda:0", now=datetime(2026, 8, 7, 12, 0))


def test_pg376_cli_exposes_high_capacity_controls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "pg376-plan.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pg376",
            "--output",
            str(output),
            "--d-model",
            "64",
            "--layers",
            "3",
            "--experts",
            "2",
            "--hidden",
            "128",
            "--max-length",
            "512",
            "--epochs",
            "7",
            "--batch",
            "9",
        ],
    )
    assert main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "plan_only"
    assert report["training"]["config"]["d_model"] == 64
    assert report["training"]["config"]["n_layers"] == 3
    assert report["training"]["config"]["experts"] == 2
    assert report["training"]["config"]["expert_hidden"] == 128
    assert report["training"]["config"]["max_length"] == 512
    assert report["training"]["batch_size"] == 9
    assert report["execution"]["optimizer_started"] is False
    assert all(value is False for value in report["promotion"].values())
