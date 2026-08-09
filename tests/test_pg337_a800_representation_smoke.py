from __future__ import annotations

import importlib


def test_pg337_gate_requires_independent_holdout_and_explicit_lane():
    module = importlib.import_module("scripts.run_pg337_a800_cross_impl_representation_smoke")
    gate = module._gate(
        dataset={"source": {"independent_implementation_holdout": True}},
        audit={"status": "diagnostic_only"},
        vocabulary={"context_tokens": ["[BOS]", "[EOS]", "x=1"]},
        env={"BLACKBOX_REMOTE_A800_TRAIN": "0", "CUDA_VISIBLE_DEVICES": "0"},
        device={"cuda_available": False, "visible_device_count": 0, "current_device": -1, "name": ""},
        locks={key: "a" * 64 for key in ("dataset", "information_audit", "vocabulary", "rules", "script", "model")},
        train_rows=[{"context_tokens": ["[BOS]", "x=1", "[EOS]"]}],
        train_failures=[],
        holdout_rows=[{"context_tokens": ["[BOS]", "x=1", "[EOS]"]}],
        holdout_failures=[],
        now=__import__("datetime").datetime(2026, 8, 8, 10, 0),
    )
    assert gate["representation_training_allowed"] is False
    assert "explicit_training_flag" in gate["failures"]
    assert "single_visible_a800_gpu0" in gate["failures"]
