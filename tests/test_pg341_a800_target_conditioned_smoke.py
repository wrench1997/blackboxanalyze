from __future__ import annotations

from datetime import datetime

from scripts.run_pg341_a800_target_conditioned_smoke import _rows, _target_coverage, evaluate_gate


def _row(split: str, question: str, action: str, safe: str = "0") -> dict:
    return {
        "view": "coarse_process",
        "split": split,
        "target_conditioned_diagnostic_eligible": True,
        "context_tokens": ["[BOS]", "surface_method=GET", "[EOS]"],
        "target_tokens": ["[TARGET_BOS]", f"question={question}", f"next_action={action}", "repair_action=observe", f"safe_to_send={safe}", "[TARGET_EOS]"],
        "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True},
    }


def test_rows_and_target_coverage_keep_ask_repair_negative_distinct() -> None:
    dataset = {"records": [_row("train", "ask_typed", "ask_typed"), _row("train", "ask_failure", "repair"), _row("train", "review_negative", "abstain"), _row("implementation_holdout", "ask_typed", "ask_typed"), _row("implementation_holdout", "ask_failure", "repair"), _row("implementation_holdout", "review_negative", "abstain")]}
    train, train_failures = _rows(dataset, "train")
    holdout, holdout_failures = _rows(dataset, "implementation_holdout")
    assert not train_failures and not holdout_failures
    coverage = _target_coverage(train)
    assert coverage["ask_present"] and coverage["repair_present"] and coverage["abstain_present"]
    assert len(holdout) == 3


def test_gate_allows_only_coarse_diagnostic_and_keeps_full_axis_closed() -> None:
    dataset = {"records": [_row("train", "ask_typed", "ask_typed"), _row("train", "ask_failure", "repair"), _row("train", "review_negative", "abstain"), _row("implementation_holdout", "ask_typed", "ask_typed"), _row("implementation_holdout", "ask_failure", "repair"), _row("implementation_holdout", "review_negative", "abstain")]}
    train, train_failures = _rows(dataset, "train")
    holdout, holdout_failures = _rows(dataset, "implementation_holdout")
    audit = {"status": "blocked_full_axis_target_gap", "coarse_process": {"diagnostic_training_allowed": True}, "full_axis": {"target_training_allowed": False}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}}
    vocabulary = {"context_tokens": ["[BOS]", "surface_method=GET", "[EOS]"], "target_tokens": ["[TARGET_BOS]", "question=ask_typed", "question=ask_failure", "question=review_negative", "next_action=ask_typed", "next_action=repair", "next_action=abstain", "repair_action=observe", "safe_to_send=0", "[TARGET_EOS]"]}
    gate = evaluate_gate(dataset=dataset, audit=audit, vocabulary=vocabulary, rules={}, env={"BLACKBOX_REMOTE_A800_TRAIN": "1", "CUDA_VISIBLE_DEVICES": "0"}, device={"cuda_available": True, "visible_device_count": 1, "current_device": 0, "name": "NVIDIA A800-SXM4-80GB"}, locks={key: "a" * 64 for key in ("dataset", "audit", "vocabulary", "rules", "script", "model")}, train_rows=train, train_failures=train_failures, holdout_rows=holdout, holdout_failures=holdout_failures, now=datetime(2026, 8, 8, 10, 0))
    assert gate["training_allowed"] is True
    assert gate["track"] == "coarse_process_only"
    assert gate["full_axis_training_allowed"] is False
    assert all(value is False for value in gate["promotion"].values())
