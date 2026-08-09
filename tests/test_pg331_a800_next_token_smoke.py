from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts.run_pg331_a800_next_token_smoke import (
    MAX_LENGTH,
    _sha256_json,
    _sha256_file,
    evaluate_training_gate,
)


ROOT = Path(__file__).resolve().parents[1]


def _row() -> dict[str, object]:
    return {
        "training_eligible": True,
        "context_tokens": ["document_presence=observed", "request_transport_presence=observed", "method=GET"],
        "target_tokens": ["[TARGET_BOS]", "question=ask_typed", "next_action=ask_typed", "[TARGET_EOS]"],
    }


def _rules() -> dict[str, object]:
    return {
        "pg331_model_training_contract": {
            "implementation": "scripts/run_pg331_a800_next_token_smoke.py",
            "implementation_sha256": _sha256_file(ROOT / "scripts/run_pg331_a800_next_token_smoke.py"),
            "test": "tests/test_pg331_a800_next_token_smoke.py",
            "test_sha256": _sha256_file(ROOT / "tests/test_pg331_a800_next_token_smoke.py"),
            "model_implementation": "app/pg295_causal_moe.py",
            "model_implementation_sha256": _sha256_file(ROOT / "app/pg295_causal_moe.py"),
            "capacity_audit": "scripts/audit_pg331_model_capacity.py",
            "capacity_audit_sha256": _sha256_file(ROOT / "scripts/audit_pg331_model_capacity.py"),
            "capacity_audit_test": "tests/test_pg331_model_capacity.py",
            "capacity_audit_test_sha256": _sha256_file(ROOT / "tests/test_pg331_model_capacity.py"),
        }
    }


def _capacity() -> dict[str, object]:
    report = {
        "status": "passed",
        "model_vocabulary_size": 10,
        "model_vocabulary_sha256": _sha256_json(sorted(["[PAD]", "[UNK]", "[BOS]", "document_presence=observed", "request_transport_presence=observed", "method=GET", "[TARGET_BOS]", "question=ask_typed", "next_action=ask_typed", "[TARGET_EOS]"])),
        "required_context_window": 497,
        "variants": [{"capacity_pass": True, "config": {"max_length": MAX_LENGTH}}],
    }
    report["audit_sha256"] = ""
    report["audit_sha256"] = _sha256_json(report)
    return report


def test_current_real_artifacts_are_blocked_without_data_and_remote_flag() -> None:
    result = evaluate_training_gate(
        now=datetime(2026, 8, 8, 7, 30, tzinfo=ZoneInfo("Asia/Shanghai")),
        env={},
        dataset=None,
        information_audit={"status": "blocked"},
        capacity_audit={"status": "blocked"},
        vocabulary_manifest={"training_eligibility": {"allowed": False}},
        rules={},
        device={"cuda_available": False, "visible_device_count": 0, "current_device": -1, "name": "not_queried"},
    )
    assert result["status"] == "blocked"
    assert result["training_allowed"] is False
    assert "explicit_training_flag" in result["failures"]
    assert "dataset_missing_or_invalid" in result["failures"]


def test_all_gates_require_complete_abstract_rows_and_a800_gpu0() -> None:
    manifest = {
        "context_tokens": ["[BOS]", "document_presence=observed", "request_transport_presence=observed", "method=GET"],
        "target_tokens": ["[TARGET_BOS]", "question=ask_typed", "next_action=ask_typed", "[TARGET_EOS]"],
        "training_eligibility": {"allowed": True},
    }
    result = evaluate_training_gate(
        now=datetime(2026, 8, 8, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
        env={"BLACKBOX_REMOTE_A800_TRAIN": "1", "CUDA_VISIBLE_DEVICES": "0"},
        dataset={"records": [_row()]},
        information_audit={"status": "passed"},
        capacity_audit=_capacity(),
        vocabulary_manifest=manifest,
        rules=_rules(),
        device={"cuda_available": True, "visible_device_count": 1, "current_device": 0, "name": "NVIDIA A800-SXM4-80GB"},
    )
    assert result["status"] == "ready_candidate_smoke"
    assert result["training_allowed"] is True
    assert result["max_length"] == 768


def test_capacity_gate_raises_model_window_for_long_source_rows() -> None:
    capacity = _capacity()
    capacity["required_context_window"] = 4145
    capacity["variants"] = [{"capacity_pass": True, "config": {"max_length": 4145}}]
    capacity["audit_sha256"] = ""
    capacity["audit_sha256"] = _sha256_json(capacity)
    manifest = {
        "context_tokens": ["[BOS]", "document_presence=observed", "request_transport_presence=observed", "method=GET"],
        "target_tokens": ["[TARGET_BOS]", "question=ask_typed", "next_action=ask_typed", "[TARGET_EOS]"],
        "training_eligibility": {"allowed": True},
    }
    result = evaluate_training_gate(
        now=datetime(2026, 8, 8, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
        env={"BLACKBOX_REMOTE_A800_TRAIN": "1", "CUDA_VISIBLE_DEVICES": "0"},
        dataset={"records": [_row()]},
        information_audit={"status": "passed"},
        capacity_audit=capacity,
        vocabulary_manifest=manifest,
        rules=_rules(),
        device={"cuda_available": True, "visible_device_count": 1, "current_device": 0, "name": "NVIDIA A800-SXM4-80GB"},
    )
    assert result["training_allowed"] is True
    assert result["max_length"] == 4145


def test_missing_information_audit_blocks_even_with_eligible_rows() -> None:
    result = evaluate_training_gate(
        now=datetime(2026, 8, 8, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
        env={"BLACKBOX_REMOTE_A800_TRAIN": "1", "CUDA_VISIBLE_DEVICES": "0"},
        dataset={"records": [_row()]},
        information_audit={"status": "blocked"},
        capacity_audit=_capacity(),
        vocabulary_manifest={"context_tokens": ["[BOS]"], "target_tokens": ["[TARGET_BOS]"], "training_eligibility": {"allowed": True}},
        rules=_rules(),
        device={"cuda_available": True, "visible_device_count": 1, "current_device": 0, "name": "NVIDIA A800-SXM4-80GB"},
    )
    assert result["training_allowed"] is False
    assert "information_preservation_passed" in result["failures"]


def test_raw_token_leak_blocks_row_eligibility() -> None:
    row = _row()
    row["context_tokens"] = ["response_body=literal"]
    result = evaluate_training_gate(
        now=datetime(2026, 8, 8, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
        env={"BLACKBOX_REMOTE_A800_TRAIN": "1", "CUDA_VISIBLE_DEVICES": "0"},
        dataset={"records": [row]},
        information_audit={"status": "passed"},
        capacity_audit=_capacity(),
        vocabulary_manifest={"context_tokens": ["[BOS]"], "target_tokens": ["[TARGET_BOS]"], "training_eligibility": {"allowed": True}},
        rules=_rules(),
        device={"cuda_available": True, "visible_device_count": 1, "current_device": 0, "name": "NVIDIA A800-SXM4-80GB"},
    )
    assert result["training_allowed"] is False
    assert any("context_firewall" in failure for failure in result["failures"])
