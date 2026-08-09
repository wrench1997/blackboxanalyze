from app.belief_state import MultiStepBelief, belief_entropy, jensen_shannon_divergence


def test_belief_update_reduces_entropy_on_discriminating_signal():
    state = MultiStepBelief()
    before = belief_entropy(state.posterior)
    step = state.observe("/metrics", {"observability": 0.9, "xss": 0.01})
    assert step["information_gain"] > 0
    assert abs(jensen_shannon_divergence(step["prior"], step["likelihood"]) - step["information_gain"]) < 1e-5
    assert abs(sum(state.posterior.values()) - 1.0) < 1e-6


def test_belief_probe_choice_uses_probabilities_not_family_labels():
    state = MultiStepBelief()
    rows = [
        {"path": "/flat", "model_score": 0.9, "surface_discriminator": {"probabilities": {"observability": 0.5, "xss": 0.5}}},
        {"path": "/sharp", "model_score": 0.2, "surface_discriminator": {"probabilities": {"observability": 0.98, "xss": 0.02}}},
    ]
    chosen = state.choose_next_probe(rows)
    assert chosen["path"] == "/sharp"


def test_belief_does_not_count_the_same_evidence_hash_twice():
    state = MultiStepBelief()
    probabilities = {"xss": 0.99, "sql_injection": 0.01}
    first = state.observe("/same", probabilities, evidence_hash="same-evidence")
    after_first = dict(state.posterior)
    second = state.observe("/same", probabilities, evidence_hash="same-evidence")
    assert first["accepted"] is True
    assert second["accepted"] is False
    assert second["duplicate_evidence"] is True
    assert second["information_gain"] == 0.0
    assert state.posterior == after_first
    assert state.snapshot()["unique_evidence_count"] == 1


def test_belief_still_accumulates_independent_evidence_hashes():
    state = MultiStepBelief()
    probabilities = {"xss": 0.99, "sql_injection": 0.01}
    state.observe("/first", probabilities, evidence_hash="evidence-1")
    after_first = dict(state.posterior)
    state.observe("/second", probabilities, evidence_hash="evidence-2")
    assert state.posterior != after_first
    assert state.snapshot()["unique_evidence_count"] == 2
