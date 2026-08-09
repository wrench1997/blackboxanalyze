from __future__ import annotations

import json
from pathlib import Path

from scripts.build_pg339_multi_shape_vocabulary import build
from scripts.run_pg339_a800_multi_shape_representation_smoke import _ablate_axis, _gate, _rows


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg339_frozen_vocab_covers_dataset_without_holdout_fitting():
    vocab = build()
    dataset = _load("pg339_multi_shape_diagnostic_dataset_v1.json")
    tokens = {str(t) for row in dataset["records"] for t in row["context_tokens"]}
    assert vocab["holdout_rows_used_for_vocabulary"] is False
    assert not (tokens - set(vocab["context_tokens"]))
    assert vocab["forbidden_tokens"] == []


def test_pg339_rows_have_train_and_shape_holdout():
    dataset = _load("pg339_multi_shape_diagnostic_dataset_v1.json")
    train, train_failures = _rows(dataset, "train")
    holdout, holdout_failures = _rows(dataset, "shape_holdout")
    assert len(train) == 9 and len(holdout) == 15
    assert train_failures == [] and holdout_failures == []


def test_pg339_axis_ablation_uses_explicit_boundaries():
    dataset = _load("pg339_multi_shape_diagnostic_dataset_v1.json")
    _, failures = _rows(dataset, "shape_holdout")
    assert failures == []
    rows, _ = _rows(dataset, "shape_holdout")
    ablated = _ablate_axis(rows[:1], "document_structure")
    assert len(ablated[0]["context_tokens"]) < len(rows[0]["context_tokens"])
    assert "axis_begin=document_structure" not in ablated[0]["context_tokens"]


def test_pg339_gate_keeps_information_and_promotion_closed():
    dataset = _load("pg339_multi_shape_diagnostic_dataset_v1.json")
    audit = _load("pg339_multi_shape_diagnostic_audit_v1.json")
    vocab = build()
    train, tf = _rows(dataset, "train")
    holdout, hf = _rows(dataset, "shape_holdout")
    gate = _gate(dataset=dataset, audit=audit, vocabulary=vocab, env={"BLACKBOX_REMOTE_A800_TRAIN": "0", "CUDA_VISIBLE_DEVICES": "0"}, device={"cuda_available": False, "visible_device_count": 0, "current_device": -1, "name": ""}, locks={k: "x" * 64 for k in ("dataset", "information_audit", "vocabulary", "rules", "script", "model")}, train_rows=train, train_failures=tf, holdout_rows=holdout, holdout_failures=hf, now=__import__("datetime").datetime(2026, 8, 8))
    assert gate["representation_training_allowed"] is False
    assert gate["information_promotion_gate_passed"] is False
    assert gate["promotion"]["vulnerability_claim_allowed"] is False
