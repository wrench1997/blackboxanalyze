from __future__ import annotations

import pytest
import torch

from app.pg293_failure_next_action import (
    FailureNextActionModel,
    build_vocabulary,
    encode_batch,
    evaluate_model,
    normalize_record,
    sanitize_context_tokens,
)


def _row(*, confirmed: bool = False, repair: str = "none", lane: str = "hard_negative") -> dict:
    return {
        "source": "hidden-source",
        "seed": 29301,
        "context_tokens": [
            "[BOS]",
            "phase=observe",
            "method=POST",
            "family=sql",
            "lane=gold",
            "failure=candidate_no_effect",
            "repair=retry_candidate",
            "fresh_reset=1",
            "candidate_sent=1",
            "[CTX_END]",
        ],
        "labels": {
            "final_belief": "confirmed_effect" if confirmed else "oracle_gap",
            "repair_attempted": repair != "none",
            "repair_succeeded": confirmed,
        },
        "repair_action": repair,
        "lane": lane,
        "payload_grounded_eligible": confirmed,
        "source_evidence_hash": "a" * 64,
    }


def test_sanitize_context_removes_identity_and_label_shortcuts():
    tokens = sanitize_context_tokens(
        [
            "[BOS]",
            "family=sql",
            "source=secret",
            "lane=gold",
            "failure=candidate_no_effect",
            "method=GET",
            "[EOS]",
        ]
    )
    assert "family=sql" not in tokens
    assert "source=secret" not in tokens
    assert "lane=gold" not in tokens
    assert "failure=candidate_no_effect" in tokens
    assert "method=GET" in tokens


def test_literal_or_malformed_context_is_rejected():
    with pytest.raises(ValueError):
        sanitize_context_tokens(["[BOS]", "payload=<script>", "[EOS]"])


def test_normalize_record_keeps_only_abstract_target_and_hashes_evidence():
    positive = normalize_record(_row(confirmed=True, repair="recheck_oracle", lane="gold"), source_group="trajectory_a", split="route_dev")
    negative = normalize_record(_row(), source_group="trajectory_b", split="family_holdout")
    assert positive["next_action"] == "recheck_oracle"
    assert positive["safe_to_send"] is False
    assert negative["next_action"] == "abstain"
    assert negative["safe_to_send"] is False
    for record in (positive, negative):
        assert record["route_identity_stored"] is False
        assert record["family_identity_stored"] is False
        assert record["oracle_label_in_context"] is False
        assert record["raw_payload_strings_stored"] is False
        assert len(record["record_sha256"]) == 64
        assert all("family=" not in token for token in record["context_tokens"])


def test_positive_replay_target_is_abstract_and_safe_bit_is_explicit():
    positive = normalize_record(_row(confirmed=True, repair="none", lane="gold"), source_group="trajectory_a", split="route_dev")
    assert positive["next_action"] == "replay_confirmed"
    assert positive["safe_to_send"] is True
    assert "next_action=replay_confirmed" in positive["target_tokens"]
    assert "safe_to_send=1" in positive["target_tokens"]


def test_model_shapes_and_hard_negative_metric_are_bounded():
    rows = [
        normalize_record(_row(confirmed=True, lane="gold"), source_group="a", split="train"),
        normalize_record(_row(), source_group="b", split="holdout"),
    ]
    vocab = build_vocabulary(rows)
    device = torch.device("cpu")
    context, lengths, targets, actions = encode_batch(rows, vocab, device)
    model = FailureNextActionModel(vocab_size=len(vocab), hidden_dim=16)
    output = model(context, lengths, targets[:, :-1])
    assert output["token"].shape[:2] == targets[:, :-1].shape
    assert output["action"].shape == (2, 6)
    metrics = evaluate_model(model, rows, vocab, device)
    assert metrics["count"] == 2
    assert 0.0 <= metrics["token_accuracy"] <= 1.0
    assert 0 <= metrics["hard_negative_false_allow"] <= 2
