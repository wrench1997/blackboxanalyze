from __future__ import annotations

from app.failure_guided_scheduler import failure_signature
from app.pg129_atomic_token_policy import ATOMIC_GROUPS, ATOMIC_TOKEN_FEATURE_DIM, atomic_trajectory_matrix, is_failure_token


def test_only_failure_tokens_get_focus_and_forward_tokens_reset_to_baseline() -> None:
    failure = failure_signature({"method": "GET", "role": "candidate", "candidate_signal": True, "positive": False, "positive_authority": False, "typed_available": True, "probe_round": 1, "max_probe_rounds": 2})
    forward = failure_signature({"method": "POST", "role": "candidate", "candidate_signal": True, "positive": True, "positive_authority": True, "typed_available": True}, prior_records=[{"method": "GET", "role": "candidate", "candidate_signal": True}])
    matrix = atomic_trajectory_matrix([failure, forward])
    failure_focus = [matrix[index][12] for index in range(len(ATOMIC_GROUPS))]
    forward_focus = [matrix[index][12] for index in range(len(ATOMIC_GROUPS), 2 * len(ATOMIC_GROUPS))]
    assert is_failure_token(failure) is True
    assert is_failure_token(forward) is False
    assert len(set(round(value, 6) for value in failure_focus)) > 1
    assert all(abs(value - 1.0 / len(ATOMIC_GROUPS)) <= 1e-6 for value in forward_focus)
    assert len(matrix) == 36
    assert len(matrix[0]) == ATOMIC_TOKEN_FEATURE_DIM
