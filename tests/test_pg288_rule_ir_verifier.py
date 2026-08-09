import torch

from app.pg285_payload_grounding import PayloadGroundingDecoder, build_vocabs, encode_rows
from app.pg288_rule_ir_verifier import apply_context_safety_gate, constrained_greedy_decode, evaluate_decoded_plans, verify_plan_tokens


def _plan(**overrides: str) -> list[str]:
    values = {
        "plan": "candidate_probe",
        "method": "GET",
        "probe_class": "sql",
        "channel": "query",
        "encoding": "url_percent",
        "wire": "query_param",
        "field_slot": "observed_or_runtime_canary",
        "repair_delta": "none",
        "family_agnostic": "1",
        "final_action": "candidate_probe",
        "safe_to_send": "1",
    }
    values.update(overrides)
    return ["[TARGET_BOS]"] + [f"{key}={value}" for key, value in values.items()] + ["[TARGET_EOS]"]


def test_valid_abstract_candidate_is_renderable_but_not_authorized():
    result = verify_plan_tokens(_plan())
    assert result["valid_structure"] is True
    assert result["safe_consistent"] is True
    assert result["renderable"] is True
    assert result["eligible_for_send"] is False
    assert result["errors"] == []


def test_abstain_must_not_carry_a_wire():
    result = verify_plan_tokens(
        _plan(plan="abstain", final_action="abstain", safe_to_send="0", wire="none", field_slot="none", encoding="unknown")
    )
    assert result["valid_structure"] is True
    assert result["renderable"] is False


def test_channel_wire_mismatch_is_rejected():
    result = verify_plan_tokens(_plan(channel="form", wire="query_param"))
    assert result["valid_structure"] is False
    assert "channel_wire_mismatch" in result["errors"]


def test_repair_requires_a_delta_and_is_not_safe():
    result = verify_plan_tokens(_plan(plan="repair_alternate", final_action="repair_alternate", safe_to_send="0", repair_delta="none"))
    assert result["valid_structure"] is False
    assert "repair_action_requires_delta" in result["errors"]


def test_literal_probe_is_never_renderable():
    result = verify_plan_tokens(_plan(field_slot="<script>alert-1</script>"))
    assert result["valid_structure"] is False
    assert result["literal_probe_token"] is True
    assert result["renderable"] is False


def test_replay_confirmation_requires_external_typed_oracle():
    result = verify_plan_tokens(_plan(plan="replay_confirmed", final_action="replay_confirmed"))
    assert result["valid_structure"] is True
    assert "replay_confirmation_requires_typed_oracle" in result["errors"]
    assert result["eligible_for_send"] is False


def test_hard_negative_false_allow_is_separate_from_action_accuracy():
    rows = [{"target": {"next_action": "abstain", "safe_to_send": False}, "target_tokens": _plan(plan="abstain", final_action="abstain", safe_to_send="0", wire="none", field_slot="none", encoding="unknown")}]
    predicted = [_plan()]
    metrics = evaluate_decoded_plans(rows, predicted, hard_negative=True)
    assert metrics["action_accuracy"] == 0.0
    assert metrics["hard_negative_false_allow"] == 1


def test_constrained_decoder_emits_a_slot_complete_skeleton():
    row = {"context_tokens": ["[BOS]", "method=GET"], "target_tokens": _plan()}
    context_vocab, target_vocab = build_vocabs([row])
    model = PayloadGroundingDecoder(len(context_vocab), len(target_vocab), embed_dim=8, hidden_dim=12)
    context_values, context_lengths, _, _ = encode_rows([row], context_vocab, target_vocab)
    decoded = constrained_greedy_decode(model, context_values, context_lengths, target_vocab, max_tokens=16)
    verification = verify_plan_tokens(decoded[0])
    assert verification["valid_structure"] is True
    assert verification["literal_probe_token"] is False


def test_context_safety_gate_abstains_without_reading_target_label():
    rows = [{"context_tokens": ["typed_available=0", "feedback=unresolved", "candidate_sent=0", "replay_consistent=0", "method=POST"]}]
    guarded, changed = apply_context_safety_gate(rows, [_plan()])
    assert changed == 1
    result = verify_plan_tokens(guarded[0])
    assert result["valid_structure"] is True
    assert result["fields"]["final_action"] == "abstain"
    assert result["fields"]["safe_to_send"] == "0"


def test_context_safety_gate_preserves_resolved_candidate():
    rows = [{"context_tokens": ["typed_available=1", "feedback=typed_effect", "candidate_sent=1", "replay_consistent=1", "method=POST"]}]
    guarded, changed = apply_context_safety_gate(rows, [_plan()])
    assert changed == 0
    assert guarded[0] == _plan()
