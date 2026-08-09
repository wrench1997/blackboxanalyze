from pathlib import Path

import pytest

from app.experiment_ledger import ExperimentLedger


def test_experiment_ledger_hash_chain_and_workspace_scope(tmp_path: Path):
    ledger = ExperimentLedger(tmp_path / "artifacts" / "runs.jsonl", tmp_path)
    first = ledger.append({"protocol_id": "pg-test", "dataset_id": "d1", "target_instance_id": "t1", "sampling_seed": 1, "local_only": True, "evidence_hash": "a" * 64})
    second = ledger.append({"protocol_id": "pg-test", "dataset_id": "d2", "target_instance_id": "t2", "sampling_seed": 2, "local_only": True, "evidence_hash": "b" * 64})
    assert second["previous_hash"] == first["record_hash"]
    assert ledger.verify()["record_count"] == 2


def test_experiment_ledger_rejects_raw_body_and_outside_path(tmp_path: Path):
    ledger = ExperimentLedger(tmp_path / "runs.jsonl", tmp_path)
    with pytest.raises(ValueError, match="forbidden"):
        ledger.append({"protocol_id": "pg-test", "raw_body": "no"})
    with pytest.raises(ValueError, match="inside workspace"):
        ExperimentLedger(tmp_path.parent / "outside.jsonl", tmp_path)
