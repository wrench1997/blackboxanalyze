import math

import pytest

from app.maze_distance import DEFAULT_GATES, compute_maze_distance


def _posterior(effect: float, unknown: float) -> dict[str, float]:
    return {"effect": effect, "input_only": (1.0 - effect - unknown) / 2.0, "no_effect": (1.0 - effect - unknown) / 2.0, "unknown": unknown}


def test_distance_is_vector_valued_and_fail_closed_for_missing_gates():
    distance = compute_maze_distance(gates={"authorized_scope": True}, posterior=_posterior(0.2, 0.4), false_accept_risk=0.3, action_cost=2, action_budget=8)
    assert distance["remaining_gate_count"] == len(DEFAULT_GATES) - 1
    assert distance["axes"]["gate"] == round((len(DEFAULT_GATES) - 1) / len(DEFAULT_GATES), 6)
    assert set(distance["remaining_gates"]) == set(DEFAULT_GATES) - {"authorized_scope"}
    assert distance["display_only"] is True
    assert distance["model_input_allowed"] is False
    assert distance["confirmation_allowed"] is False


def test_distance_moves_closer_when_a_gate_is_completed_but_keeps_uncertainty_separate():
    gates = {name: True for name in DEFAULT_GATES}
    before = compute_maze_distance(gates={**gates, "typed_effect": False}, posterior=_posterior(0.25, 0.25), false_accept_risk=0.2, action_cost=2, action_budget=8)
    after = compute_maze_distance(gates=gates, posterior=_posterior(0.25, 0.25), false_accept_risk=0.2, action_cost=2, action_budget=8)
    assert after["remaining_gate_count"] < before["remaining_gate_count"]
    assert after["axes"]["uncertainty"] == before["axes"]["uncertainty"]
    assert after["display_distance"] < before["display_distance"]


def test_distance_never_turns_confidence_into_confirmation():
    distance = compute_maze_distance(gates={name: True for name in DEFAULT_GATES}, posterior={"effect": 1.0, "input_only": 0.0, "no_effect": 0.0, "unknown": 0.0}, false_accept_risk=0.0, action_cost=0, action_budget=1)
    assert math.isclose(distance["display_distance"], 0.0)
    assert distance["confirmation_allowed"] is False


def test_distance_rejects_invalid_action_budget():
    with pytest.raises(ValueError):
        compute_maze_distance(gates={}, posterior=_posterior(0.2, 0.4), false_accept_risk=0.2, action_cost=1, action_budget=0)
