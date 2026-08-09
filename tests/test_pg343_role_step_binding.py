from __future__ import annotations

import pytest

from app.pg331_web_tokenizer import tokenize_web_observation
from app.pg343_role_step_binding import bind_context_tokens, bind_observation, binding_present


def _observation() -> dict:
    return {
        "belief_and_replay": {
            "observation_presence": "present",
            "observation_delta_axis": "none",
            "belief_prior_bucket": "low",
            "belief_posterior_bucket": "low",
            "belief_delta_axis": "none",
            "history_action": "observe",
            "typed_available": "present",
            "evidence_present": "present",
            "negative_control": "present",
            "fresh_reset": "present",
            "replay_ready": "present",
            "reference_present": "present",
            "candidate_present": "present",
            "step_budget": "few",
            "probe_count": 1,
            "evidence_hash_present": "present",
        }
    }


def test_pg343_observation_binding_emits_abstract_role_and_step_tokens() -> None:
    bound = bind_observation(_observation(), role="candidate", step="failure")
    result = tokenize_web_observation(bound)
    tokens = result["context_tokens"]
    assert "belief_probe_role=candidate" in tokens
    assert "belief_process_step=failure" in tokens
    assert not any(item.get("kind") == "raw_fields_omitted" for item in result["loss_report"]["losses"])


def test_pg343_context_binding_is_idempotent_and_conflicts_fail_closed() -> None:
    tokens = bind_context_tokens(["document_presence=observed"], role="negative", step="repair")
    assert binding_present(tokens) is True
    assert bind_context_tokens(tokens, role="negative", step="repair") == tokens
    with pytest.raises(ValueError):
        bind_context_tokens(tokens, role="candidate", step="repair")
    with pytest.raises(ValueError):
        bind_context_tokens(tokens, role="negative", step="baseline")


def test_pg343_never_infers_role_from_target_or_accepts_raw_literals() -> None:
    with pytest.raises(ValueError):
        bind_context_tokens(["document_presence=observed", "probe_variant_ref=candidate"], role="candidate", step="failure")
    with pytest.raises(ValueError):
        bind_context_tokens(["document_presence=observed", "raw_hint=anything"], role="candidate", step="failure")
