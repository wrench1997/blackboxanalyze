import torch

from app.rule_ir_decoder import (
    DECODER_FAMILIES,
    FEATURE_DIM,
    RuleIRDecoder,
    abstract_rule_ir,
    abstract_rule_ir_canonical,
    trace_feature_vector,
    validate_abstract_rule_ir,
)


def test_abstract_templates_are_grammar_checked_and_canonical():
    for family in DECODER_FAMILIES:
        rule = abstract_rule_ir(family)
        validate_abstract_rule_ir(rule)
        assert abstract_rule_ir_canonical(family).startswith("{")


def test_trace_projection_ignores_oracle_only_fields():
    visible = [{"input": {"payload": "&amp;lt;b&amp;gt;"}, "context": {}, "state": {}, "history": [], "output": True}]
    leaked = [{**visible[0], "intended_output": False, "is_counterexample": True, "family": "xss", "record_id": "secret"}]
    assert trace_feature_vector(visible) == trace_feature_vector(leaked)
    assert len(trace_feature_vector(visible)) == FEATURE_DIM


def test_decoder_emits_valid_rule_ir_with_zero_abstention_threshold():
    model = RuleIRDecoder()
    decoded = model.decode(torch.zeros(2, FEATURE_DIM), abstain_threshold=0.0)
    assert len(decoded) == 2
    for row in decoded:
        assert row["candidate_family"] in DECODER_FAMILIES
        validate_abstract_rule_ir(row["rule_ir"])
