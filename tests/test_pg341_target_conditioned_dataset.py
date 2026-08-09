from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_pg341_target_conditioned_dataset import audit
from scripts.build_pg341_target_conditioned_dataset import build
from scripts.build_pg341_target_conditioned_vocabulary import build as build_vocab


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8-sig"))


def test_pg341_keeps_coarse_and_full_axis_views_separate_and_preserves_targets() -> None:
    dataset = build(
        _load("pg337_cross_impl_process_token_v1.json"),
        _load("pg338_information_preserving_process_token_v1.json"),
        coarse_path=ROOT / "research" / "pg337_cross_impl_process_token_v1.json",
        full_axis_path=ROOT / "research" / "pg338_information_preserving_process_token_v1.json",
    )
    assert dataset["counts"]["coarse_process"] == 183
    assert dataset["counts"]["full_axis"] == 27
    assert dataset["track_contract"]["views_must_not_be_merged_for_capability_claim"] is True
    assert all(row["training_eligible"] is False for row in dataset["records"])
    assert any(row["view"] == "coarse_process" and "question=ask_typed" in row["target_tokens"] for row in dataset["records"])
    assert any(row["view"] == "full_axis" and "question=ask_failure" in row["target_tokens"] for row in dataset["records"])
    encoded = json.dumps(dataset, ensure_ascii=False).casefold()
    for forbidden in ("payload=", "response_body=", "oracle=", "evaluator=", "route_literal="):
        assert forbidden not in encoded


def test_pg341_audit_exposes_full_axis_target_gap_without_promoting_coarse_rows() -> None:
    dataset = build(
        _load("pg337_cross_impl_process_token_v1.json"),
        _load("pg338_information_preserving_process_token_v1.json"),
        coarse_path=ROOT / "research" / "pg337_cross_impl_process_token_v1.json",
        full_axis_path=ROOT / "research" / "pg338_information_preserving_process_token_v1.json",
    )
    report = audit(dataset)
    assert report["status"] == "blocked_full_axis_target_gap"
    assert report["target_coverage"]["coarse_train_complete"] is True
    assert report["target_coverage"]["full_axis_train_complete"] is False
    assert report["coarse_process"]["diagnostic_training_allowed"] is True
    assert "full_axis_train_target_coverage_missing" in report["failures"]
    assert report["promotion"]["training_allowed"] is False


def test_pg341_vocabulary_contains_both_context_and_target_inventories() -> None:
    dataset = build(
        _load("pg337_cross_impl_process_token_v1.json"),
        _load("pg338_information_preserving_process_token_v1.json"),
        coarse_path=ROOT / "research" / "pg337_cross_impl_process_token_v1.json",
        full_axis_path=ROOT / "research" / "pg338_information_preserving_process_token_v1.json",
    )
    vocab = build_vocab(dataset)
    assert "[TARGET_BOS]" in vocab["target_tokens"]
    assert "[TARGET_BOS]" not in vocab["context_tokens"]
    assert any(token.startswith("document_presence=") for token in vocab["context_tokens"])
    assert "question=ask_failure" in vocab["target_tokens"]
    assert vocab["promotion"]["memory_promotion_allowed"] is False
