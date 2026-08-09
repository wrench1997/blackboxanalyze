from __future__ import annotations

from datetime import datetime

from scripts.run_pg342_a800_full_axis_representation_smoke import AXES, _gate, _implementation_isolation, _rows


class _GuardedRecord(dict):
    def get(self, key, default=None):  # type: ignore[override]
        if key == "target_tokens":
            raise AssertionError("context-only runner read target_tokens")
        return super().get(key, default)


def _context() -> list[str]:
    tokens: list[str] = []
    for axis in AXES:
        tokens.extend([f"axis_begin={axis}", f"{axis}_presence=observed", f"axis_end={axis}"])
    return tokens + ["chunk_boundary=begin", "chunk_boundary=end", *[f"field_bucket={index}" for index in range(12)]]


def _record(split: str, implementation_hash: str) -> _GuardedRecord:
    return _GuardedRecord(
        split=split,
        source_implementation_hash=implementation_hash,
        context_tokens=_context(),
        field_capture_manifest={axis: {"status": "observed"} for axis in AXES},
        axis_presence={axis: True for axis in AXES},
        context_firewall={"forbidden_token_count": 0, "sidecars_off_context": True},
        raw_payload_stored=False,
        raw_response_body_stored=False,
        oracle_answer_in_context=False,
    )


def test_rows_are_context_only_and_preserve_axes() -> None:
    dataset = {"records": [_record("train", "a"), _record("implementation_holdout", "b")]}
    train, train_failures = _rows(dataset, "train")
    holdout, holdout_failures = _rows(dataset, "implementation_holdout")
    assert train_failures == []
    assert holdout_failures == []
    assert len(train) == len(holdout) == 1
    assert train[0]["target_tokens"] == []
    assert _implementation_isolation(dataset, "train", "implementation_holdout")["passed"] is True


def test_gate_accepts_only_diagnostic_representation_lane() -> None:
    dataset = {"records": [_record("train", "a"), _record("implementation_holdout", "b")]}
    train, train_failures = _rows(dataset, "train")
    holdout, holdout_failures = _rows(dataset, "implementation_holdout")
    audit = {
        "status": "diagnostic_only",
        "information_gate": {"all_axes_present": True},
        "context_target_alignment": {"rate": 1.0},
        "context_firewall": {"forbidden_token_count": 0},
    }
    vocabulary = {"context_tokens": _context()}
    locks = {key: "a" * 64 for key in ("dataset", "information_audit", "vocabulary", "rules", "script", "model")}
    gate = _gate(
        dataset=dataset,
        audit=audit,
        vocabulary=vocabulary,
        env={"BLACKBOX_REMOTE_A800_TRAIN": "1", "CUDA_VISIBLE_DEVICES": "0"},
        device={"cuda_available": True, "visible_device_count": 1, "current_device": 0, "name": "NVIDIA A800-SXM4-80GB"},
        locks=locks,
        train_rows=train,
        train_failures=train_failures,
        holdout_rows=holdout,
        holdout_failures=holdout_failures,
        now=datetime(2026, 8, 8, 12, 0),
    )
    assert gate["representation_training_allowed"] is True
    assert gate["target_tokens_read"] is False
    assert gate["promotion"] == {
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }


def test_missing_axis_is_blocked_and_not_imputed() -> None:
    record = _record("train", "a")
    record["axis_presence"] = {axis: True for axis in AXES[:-1]}
    rows, failures = _rows({"records": [record]}, "train")
    assert rows == []
    assert failures == ["row_0_axis_presence"]
