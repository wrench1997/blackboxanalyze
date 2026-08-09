import json
from datetime import datetime
from pathlib import Path

from scripts.run_pg336_a800_real_failure_representation_smoke import _context_rows, _gate


ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8-sig"))


def test_context_reader_drops_targets_and_preserves_seed_holdout_split():
    dataset = _load("pg336_real_failure_process_token_v1.json")
    train, train_failures = _context_rows(dataset, "train")
    holdout, holdout_failures = _context_rows(dataset, "seed_holdout")
    assert len(train) == 60
    assert len(holdout) == 120
    assert not train_failures
    assert not holdout_failures
    assert all(row["target_tokens"] == [] for row in train + holdout)
    assert all("payload=" not in " ".join(row["context_tokens"]) for row in train + holdout)


def test_gate_fails_closed_without_remote_gpu_and_explicit_flag():
    dataset = _load("pg336_real_failure_process_token_v1.json")
    audit = _load("pg336_real_failure_process_token_audit_v1.json")
    vocab = _load("pg336_real_failure_process_vocabulary_v1.json")
    train, train_failures = _context_rows(dataset, "train")
    holdout, holdout_failures = _context_rows(dataset, "seed_holdout")
    gate = _gate(
        dataset=dataset,
        audit=audit,
        vocabulary=vocab,
        env={"CUDA_VISIBLE_DEVICES": "0"},
        device={"cuda_available": False, "visible_device_count": 0, "current_device": -1, "name": ""},
        locks={key: "a" * 64 for key in ("dataset", "information_audit", "vocabulary", "rules", "script", "model")},
        train_rows=train,
        train_failures=train_failures,
        holdout_rows=holdout,
        holdout_failures=holdout_failures,
        now=datetime(2026, 8, 8, 10, 0),
    )
    assert gate["status"] == "blocked"
    assert gate["representation_training_allowed"] is False
    assert "explicit_training_flag" in gate["failures"]
    assert "single_visible_a800_gpu0" in gate["failures"]
    assert all(value is False for value in gate["promotion"].values())
