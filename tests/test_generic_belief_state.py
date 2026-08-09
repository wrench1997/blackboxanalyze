from app.generic_belief_state import (
    GENERIC_STATES,
    GenericBeliefState,
    likelihood_from_projection,
    schedule_next_action,
)


def test_generic_belief_updates_and_deduplicates_evidence_without_family_names():
    state = GenericBeliefState()
    first = state.observe("get:p0", {"effect": 0.8, "input_only": 0.05, "no_effect": 0.05, "unknown": 0.1}, evidence_hash="evidence-00000001")
    before = dict(state.posterior)
    duplicate = state.observe("post:p0", {"effect": 0.8, "input_only": 0.05, "no_effect": 0.05, "unknown": 0.1}, evidence_hash="evidence-00000001")
    assert first["accepted"] is True
    assert duplicate["duplicate_evidence"] is True
    assert state.posterior == before
    assert set(state.posterior) == set(GENERIC_STATES)
    assert all(name not in state.snapshot() for name in ("xss", "sql_injection", "authentication"))


def test_input_only_anomaly_requires_negative_replay_and_never_confirms():
    output = {
        "decision": "abstain",
        "reason": "input_changed_without_surface_effect",
        "composition": {"observed_atoms": ["probe_binding_valid", "candidate_without_surface_delta"]},
        "promotion_eligible": False,
    }
    likelihood = likelihood_from_projection(output)
    assert max(likelihood, key=likelihood.get) == "input_only"
    assert schedule_next_action(output, observed_methods={"GET"}, max_steps=2, step_count=1) == "repeat_matched_negative_other_method"
    assert schedule_next_action(output, observed_methods={"GET", "POST"}, max_steps=2, step_count=2) == "await_typed_oracle_then_abstain"


def test_effect_requires_both_methods_then_hands_off_to_typed_oracle():
    output = {
        "decision": "confirm_candidate",
        "composition": {"observed_atoms": ["effect_present", "probe_binding_valid", "supported_active_slot:p0"]},
        "promotion_eligible": False,
    }
    assert schedule_next_action(output, observed_methods={"GET"}, max_steps=2, step_count=1) == "replay_other_method"
    assert schedule_next_action(output, observed_methods={"GET", "POST"}, max_steps=2, step_count=2) == "await_typed_oracle"

