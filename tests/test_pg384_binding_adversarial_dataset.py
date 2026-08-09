"""Tests for the abstract Rule-IR binding lane."""

from __future__ import annotations

from scripts.build_pg384_binding_adversarial_dataset import build_dataset


def test_binding_matrix_has_safe_candidate_reference_lanes_and_unsafe_negative() -> None:
    dataset = build_dataset()
    assert dataset["audit"]["status"] == "passed_abstract_binding_candidate"
    assert dataset["counts"]["abstract_safe_to_send_rows"] > 0
    assert dataset["counts"]["abstract_unsafe_or_ask_rows"] > 0
    assert dataset["counts"]["candidate_reference_replay_binding_rows"] == dataset["counts"]["abstract_safe_to_send_rows"]
    for row in dataset["records"]:
        role = row["source"]["role"]
        if role == "negative":
            assert "safe_to_send=false" in row["target_tokens"]
            assert row["abstract_binding_lane"] == "ask_or_abstain"
    assert dataset["safety"]["raw_payload_in_context"] is False
    assert dataset["promotion"]["capability_training_allowed"] is False


def test_binding_lane_never_persists_concrete_wire() -> None:
    dataset = build_dataset()
    for row in dataset["records"][::311]:
        assert row["context_firewall"] == {"forbidden_token_count": 0, "sidecars_off_context": True}
        assert "wire" not in row
        assert "raw_payload" not in row
        assert all(value is False for value in row["promotion"].values())
