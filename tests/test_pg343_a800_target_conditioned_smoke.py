from __future__ import annotations

from datetime import datetime

from scripts.run_pg343_a800_target_conditioned_smoke import _target_coverage, evaluate_gate


def _row(action: str, question: str, safe: bool, role: str = "candidate", step: str = "repair") -> dict:
    return {
        "context_tokens": ["document_presence=observed", f"belief_probe_role={role}", f"belief_process_step={step}"],
        "target_tokens": ["[TARGET_BOS]", f"question={question}", f"next_action={action}", f"safe_to_send={int(safe)}", "[TARGET_EOS]"],
        "safe_to_send": safe,
        "role_step_binding": {"source_attested": True},
    }


def test_pg343_target_coverage_tracks_ask_repair_abstain_and_positive() -> None:
    rows = [_row("repair", "ask_failure", False), _row("abstain", "ask_failure", False, "negative", "failure"), _row("send_probe", "none", True)]
    coverage = _target_coverage(rows)
    assert coverage["ask_present"] is True
    assert coverage["repair_present"] is True
    assert coverage["abstain_present"] is True
    assert coverage["positive_present"] is True


def test_pg343_gate_fails_closed_when_entropy_or_vocab_is_missing() -> None:
    rows = [_row("repair", "ask_failure", False), _row("abstain", "ask_failure", False, "negative", "failure"), _row("send_probe", "none", True)]
    gate = evaluate_gate(
        dataset={},
        audit={"status": "diagnostic_passed_not_training_eligible", "failures": [], "axis_token_sequence_entropy": {axis: {"unique_sequences": 2} for axis in ("document_structure", "navigation", "request_transport", "response_transport", "javascript_surface", "failure_feedback", "belief_and_replay")}, "counts": {"context_split_leaks": 0, "source_record_split_leaks": 0, "implementation_split_leaks": 0}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}},
        vocabulary={"context_tokens": [], "target_tokens": [], "append_only": True, "forbidden_tokens": [], "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}},
        rules={},
        env={"BLACKBOX_REMOTE_A800_TRAIN": "0", "CUDA_VISIBLE_DEVICES": "0"},
        device={"cuda_available": False, "visible_device_count": 0, "current_device": -1, "name": ""},
        locks={"dataset": "x", "audit": "x", "vocabulary": "x", "rules": "x", "script": "x", "model": "x"},
        train_rows=rows,
        train_failures=[],
        holdout_rows=rows,
        holdout_failures=[],
        now=datetime(2026, 8, 9),
    )
    assert gate["training_allowed"] is False
    assert "explicit_training_flag" in gate["failures"]
    assert "single_visible_a800_gpu0" in gate["failures"]
    assert "unknown_context_or_target_token" in gate["failures"]


def test_pg343_gate_reads_canonical_nested_axis_audit() -> None:
    rows = [_row("repair", "ask_failure", False), _row("abstain", "ask_failure", False, "negative", "failure"), _row("send_probe", "none", True)]
    axes = {axis: {"unique_sequences": 2} for axis in ("document_structure", "navigation", "request_transport", "response_transport", "javascript_surface", "failure_feedback", "belief_and_replay")}
    gate = evaluate_gate(
        dataset={},
        audit={"status": "diagnostic_passed_not_training_eligible", "failures": [], "counts": {"axis_token_sequence_entropy": axes, "context_split_leaks": 0, "source_record_split_leaks": 0, "implementation_split_leaks": 0}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}},
        vocabulary={"context_tokens": ["document_presence=observed", "belief_probe_role=candidate", "belief_process_step=repair"], "target_tokens": ["[TARGET_BOS]", "question=ask_failure", "next_action=repair", "safe_to_send=0", "[TARGET_EOS]"], "append_only": True, "forbidden_tokens": [], "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}},
        rules={}, env={"BLACKBOX_REMOTE_A800_TRAIN": "0", "CUDA_VISIBLE_DEVICES": "0"}, device={"cuda_available": False, "visible_device_count": 0, "current_device": -1, "name": ""}, locks={"dataset": "x", "audit": "x", "vocabulary": "x", "rules": "x", "script": "x", "model": "x"}, train_rows=rows, train_failures=[], holdout_rows=rows, holdout_failures=[], now=datetime(2026, 8, 9)
    )
    assert "axis_sequence_entropy" not in gate["failures"]
