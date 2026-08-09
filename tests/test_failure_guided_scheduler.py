from __future__ import annotations

import pytest

from app.failure_guided_scheduler import failure_signature, key_feature_weights_for_signature, validate_failure_signature
from app.pg127_key_feature_policy import ASSEMBLY_FEATURE_DIM, failure_assembly_feature_vector


def test_candidate_failure_requests_other_channel_then_abstains() -> None:
    first = failure_signature(
        {"method": "GET", "role": "candidate", "candidate_signal": True, "positive": False, "positive_authority": False, "typed_available": True},
        step_count=1,
    )
    assert first["kind"] == "candidate_without_typed_effect"
    assert first["next_action"] == "probe_candidate_other_method"
    second = failure_signature(
        {"method": "POST", "role": "candidate", "candidate_signal": True, "positive": False, "positive_authority": False, "typed_available": True},
        prior_records=[{"method": "GET", "role": "candidate", "candidate_signal": True}],
        step_count=2,
    )
    assert second["next_action"] == "abstain_candidate_only"
    assert validate_failure_signature(second)["memory_promotion_allowed"] is False


def test_unknown_oracle_is_replayed_then_abstained() -> None:
    first = failure_signature(
        {"method": "GET", "role": "candidate", "candidate_signal": True, "positive": False, "positive_authority": False, "typed_available": False},
    )
    assert first["kind"] == "oracle_unavailable"
    assert first["next_action"] == "replay_other_method"
    second = failure_signature(
        {"method": "POST", "role": "candidate", "candidate_signal": True, "positive": False, "positive_authority": False, "typed_available": False},
        prior_records=[{"method": "GET", "role": "candidate", "candidate_signal": True}],
    )
    assert second["next_action"] == "abstain_unknown_oracle"


def test_positive_is_terminal_only_with_typed_authority() -> None:
    result = failure_signature(
        {"method": "POST", "role": "candidate", "candidate_signal": True, "positive": True, "positive_authority": True, "typed_available": True},
        prior_records=[{"method": "GET", "role": "candidate", "candidate_signal": True}],
    )
    assert result["kind"] == "typed_positive"
    assert result["next_action"] == "stop_confirmed_positive"


def test_first_typed_positive_uses_one_canonical_cross_channel_action() -> None:
    result = failure_signature(
        {"method": "GET", "role": "candidate", "candidate_signal": True, "positive": True, "positive_authority": True, "typed_available": True},
    )
    assert result["kind"] == "typed_positive"
    assert result["next_action"] == "probe_candidate_other_method"


def test_failure_focus_weights_are_normalized_and_gate_driven() -> None:
    result = failure_signature(
        {
            "method": "GET",
            "role": "candidate",
            "candidate_signal": True,
            "positive": False,
            "positive_authority": False,
            "typed_available": True,
            "probe_round": 1,
            "max_probe_rounds": 3,
        },
    )
    checked = validate_failure_signature(result)
    weights = checked["key_feature_weights"]
    assert abs(sum(weights.values()) - 1.0) <= 1e-5
    assert checked["key_features_ranked"][0] == "failed_gate"
    assert weights["probe_budget"] > weights["failure_kind"]
    assert key_feature_weights_for_signature(checked) == weights


def test_failure_assembly_changes_when_focus_is_removed_but_capacity_is_fixed() -> None:
    signature = failure_signature(
        {
            "method": "POST",
            "role": "candidate",
            "candidate_signal": True,
            "positive": False,
            "positive_authority": False,
            "typed_available": True,
            "probe_round": 1,
            "max_probe_rounds": 2,
        },
    )
    weighted = failure_assembly_feature_vector(signature, mode="weighted")
    uniform = failure_assembly_feature_vector(signature, mode="uniform")
    zero = failure_assembly_feature_vector(signature, mode="zero")
    assert len(weighted) == len(uniform) == len(zero) == ASSEMBLY_FEATURE_DIM
    assert weighted != uniform
    assert any(value != 0.0 for value in weighted)
    assert all(value == 0.0 for value in zero)


def test_failure_transition_records_and_requires_changed_action() -> None:
    result = failure_signature(
        {
            "method": "GET",
            "role": "candidate",
            "candidate_signal": True,
            "positive": False,
            "positive_authority": False,
            "typed_available": True,
            "previous_action": "probe_candidate_same_method",
        }
    )
    assert result["action_changed"] is True
    assert result["repair_transition_required"] is True
    assert validate_failure_signature(result)["repair_transition_valid"] is True

    with pytest.raises(ValueError, match="must change"):
        failure_signature(
            {
                "method": "GET",
                "role": "candidate",
                "candidate_signal": True,
                "positive": False,
                "positive_authority": False,
                "typed_available": True,
                "previous_action": "probe_candidate_other_method",
            }
        )

    with pytest.raises(ValueError, match="not allow-listed"):
        failure_signature(
            {
                "method": "GET",
                "role": "candidate",
                "candidate_signal": True,
                "positive": False,
                "positive_authority": False,
                "typed_available": True,
                "previous_action": "raw-request-body-not-an-action",
            }
        )
