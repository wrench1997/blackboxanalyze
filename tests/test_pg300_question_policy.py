from app.pg300_question_policy import audit_question_records, canonical_question_context, question_for_observation, question_record


def test_question_policy_prioritizes_unavailable_observation_without_answer_leak():
    tokens = ["surface_status=302", "evidence_present=unknown", "typed_available=unknown", "feedback_state=unknown", "replay_ready=unknown"]
    assert question_for_observation(tokens) == "ask_typed_availability"
    context = canonical_question_context(tokens)
    assert context[:5] == ["[BOS]", "typed_available=unknown", "feedback_state=unknown", "replay_ready=unknown", "evidence_present=unknown"]
    assert "result_verified" not in " ".join(context)


def test_question_record_target_is_short_causal_sequence():
    row = question_record({"record_id": "r1", "split": "train", "training_eligible": True, "context_tokens": ["typed_available=1", "feedback_state=observable_progress", "replay_ready=unknown", "evidence_present=1"]})
    assert row["target_tokens"] == ["[TARGET_BOS]", "question=ask_replay_readiness", "[TARGET_EOS]"]
    assert row["safe_to_send"] is False


def test_question_audit_rejects_forbidden_answer_fields():
    row = question_record({"record_id": "r1", "split": "train", "training_eligible": True, "context_tokens": ["typed_available=unknown", "feedback_state=unknown", "replay_ready=unknown", "evidence_present=unknown"]})
    row["context_tokens"].append("family=sqli")
    audit = audit_question_records([row])
    assert audit["status"] == "failed"
    assert audit["checks"]["forbidden_answer_leaks_absent"] is False
