import json
from pathlib import Path

import pytest

from scripts.run_pg373_staged_pretrain_candidate import main, run_candidate


ROOT = Path(__file__).resolve().parents[1]


def test_pg373_cpu_smoke_uses_trained_baseline(tmp_path, monkeypatch):
    output = tmp_path / "pg373.json"
    monkeypatch.setattr("sys.argv", ["run_pg373", "--cpu-smoke", "--output", str(output)])
    assert main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "cpu_smoke_candidate_only"
    assert report["training"]["baseline_kind"] == "train_only_next_token_pretrain"
    assert report["scientific_gate"]["trained_baseline_entropy_comparison"] is True
    assert all(value is False for value in report["promotion"].values())


def test_pg373_cuda_requires_explicit_remote_gate(monkeypatch):
    monkeypatch.delenv("BLACKBOX_REMOTE_A800_TRAIN", raising=False)
    with pytest.raises(RuntimeError, match="BLACKBOX_REMOTE_A800_TRAIN"):
        from scripts.run_pg373_staged_pretrain_candidate import _device_gate

        _device_gate("cuda:0")


def test_pg373_does_not_emit_raw_material(tmp_path, monkeypatch):
    output = tmp_path / "pg373.json"
    monkeypatch.setattr("sys.argv", ["run_pg373", "--cpu-smoke", "--output", str(output), "--json"])
    assert main() == 0
    encoded = output.read_text(encoding="utf-8")
    assert "raw_payload" not in encoded
    assert "response_body" not in encoded
    assert "http://" not in encoded
