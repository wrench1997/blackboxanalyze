from app.pg349_constrained_rule_ir_decoder import audit_rule_ir_output, constrain_rule_ir, decode_rule_ir


def _context(**overrides):
    values = {
        "belief_typed_available": "present",
        "belief_fresh_reset": "present",
        "belief_replay_ready": "present",
        "belief_evidence_present": "present",
        "belief_reference_present": "present",
        "belief_candidate_present": "present",
        "belief_negative_control": "clean",
        "document_presence": "observed",
        "request_transport_presence": "observed",
        "response_transport_presence": "observed",
        "javascript_presence": "observed",
        "failure_failure_class": "none",
    }
    values.update(overrides)
    return [f"{key}={value}" for key, value in values.items()]


def _proposal(**overrides):
    result = {
        "question": "none",
        "next_action": "send_probe",
        "repair_action": "none",
        "transport_ref": "get_query",
        "field_role_ref": "query_text",
        "encoding_ref": "url_percent",
        "probe_variant_ref": "source_attested_candidate",
        "payload_shape_ref": "query_marker",
        "oracle_ref": "typed_effect",
        "safe_to_send": "1",
    }
    result.update(overrides)
    return result


def test_missing_observation_forces_typed_ask_and_no_send():
    result = constrain_rule_ir(_context(belief_replay_ready="unknown"), _proposal())
    assert result["safe_to_send"] is False
    assert result["target"]["question"] == "ask_typed"
    assert result["target"]["next_action"] == "ask_typed"
    assert "replay_ready" in result["missing_fields"]


def test_failure_forces_one_variable_repair():
    result = constrain_rule_ir(_context(failure_failure_class="parse_error"), _proposal())
    assert result["safe_to_send"] is False
    assert result["target"]["question"] == "ask_failure"
    assert result["target"]["next_action"] == "repair"
    assert result["target"]["repair_action"] == "one_variable"


def test_complete_evidence_allows_only_abstract_bound_candidate():
    result = constrain_rule_ir(_context(), _proposal())
    assert result["safe_to_send"] is True
    assert result["target"]["payload_shape_ref"] == "query_marker"
    assert result["raw_payload_in_output"] is False
    assert audit_rule_ir_output(_context(), _proposal())["status"] == "passed"


def test_invalid_or_raw_proposal_is_fail_closed():
    proposal = _proposal(payload="literal-should-never-pass", oracle_ref="unknown")
    result = constrain_rule_ir(_context(), proposal)
    assert result["safe_to_send"] is False
    assert result["target"]["question"] == "ask_typed"
    assert result["target"]["safe_to_send"] == "0"
    assert result["raw_payload_in_output"] is False


def test_decode_is_bounded_target_shape():
    tokens = decode_rule_ir(_context(belief_evidence_present="unknown"), _proposal())
    assert tokens[0] == "[TARGET_BOS]"
    assert tokens[-1] == "[TARGET_EOS]"
    assert len(tokens) == 12
    assert all("<" not in token and "http" not in token.lower() for token in tokens)

