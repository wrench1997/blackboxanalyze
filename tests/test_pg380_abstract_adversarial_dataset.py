from __future__ import annotations

import json

from scripts.build_pg380_abstract_adversarial_dataset import build_dataset


def test_dataset_has_large_abstract_matrix_and_disjoint_source_splits() -> None:
    dataset = build_dataset()
    assert dataset["counts"]["records"] == 3168
    assert dataset["counts"]["unique_record_ids"] == dataset["counts"]["records"]
    assert dataset["counts"]["train"] == 2112
    assert dataset["counts"]["implementation_holdout"] == 1056
    assert dataset["audit"]["cross_split_source_hash_overlap"] == 0
    assert dataset["audit"]["training_eligible"] == 0


def test_all_thirteen_target_slots_are_abstract_and_safety_flags_are_closed() -> None:
    dataset = build_dataset()
    for row in dataset["records"]:
        tokens = row["target_tokens"]
        assert tokens[0] == "[TARGET_BOS]" and tokens[-1] == "[TARGET_EOS]"
        assert len(tokens) == 15  # BOS + 13 slots + EOS
        assert row["training_flags"]["capability_training_allowed"] is False
        assert row["promotion"]["vulnerability_claim_allowed"] is False
        assert row["abstract_observation"]["method"] in {"GET", "POST"}


def test_dataset_contains_feedback_repairs_asks_and_negative_controls() -> None:
    dataset = build_dataset()
    text = json.dumps(dataset, ensure_ascii=False).casefold()
    assert "next_action=ask" in text
    assert "repair_action=encoding" in text
    assert "repair_action=syntax" in text
    assert "negative_control=matched_triplet" in text
    assert "next_action=replay" in text


def test_no_raw_payload_wire_urls_or_evaluator_answers_are_present() -> None:
    dataset = build_dataset()
    text = json.dumps(dataset, ensure_ascii=False).casefold()
    for marker in ("http://", "https://", "javascript:", "<script", "document.cookie", "route_literal", "oracle_answer="):
        assert marker not in text
    for row in dataset["records"]:
        assert all(not str(token).casefold().startswith(("raw_", "payload=", "wire=", "response_body=")) for token in row["context_tokens"] + row["target_tokens"])
    assert dataset["audit"]["raw_marker_count"] == 0
    assert dataset["safety"]["raw_payload_in_context"] is False
