from __future__ import annotations

import json

from scripts.plan_pg388_logic_surface_candidate import plan


def test_surface_candidate_plan_fails_closed_without_training() -> None:
    report = plan()

    assert report["status"] == "blocked_surface_source_contract"
    assert report["source"]["row_count"] == 144
    assert report["source"]["split_counts"] == {
        "implementation_holdout": 72,
        "train": 72,
    }
    assert report["source"]["implementation_count"] == 2
    assert report["sequence_diversity"]["cross_split_context_overlap"] == 0
    assert report["sequence_diversity"]["cross_split_target_overlap"] == 0
    assert report["train_only_vocabulary"]["scope"] == "train_context_only"
    assert report["train_only_vocabulary"]["holdout_unknown_token_count"] > 0
    assert report["candidate_config"]["optimizer_started"] is False
    assert report["candidate_config"]["device"] == "none"
    assert report["training_eligible"] == 0
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["payload_catalog_promotion_allowed"] is False


def test_surface_candidate_plan_is_bounded_and_does_not_emit_rows_or_raw_values() -> None:
    serialized = json.dumps(plan(), ensure_ascii=False, sort_keys=True).casefold()

    for marker in (
        '"rows"',
        "context_tokens",
        "target_tokens",
        "payload=",
        "wire=",
        "response_body=",
        "http://",
        "https://",
    ):
        assert marker not in serialized


def test_surface_candidate_plan_keeps_execution_flags_false() -> None:
    report = plan()
    assert report["gate"]["gpu_touched"] is False
    assert report["gate"]["docker_started"] is False
    assert report["gate"]["network_contacted"] is False
    assert report["gate"]["checkpoint_written"] is False
