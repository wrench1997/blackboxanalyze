from copy import deepcopy

import pytest

from app.rule_ir_binding import bind_rule_ir_slots, shadow_evidence, validate_binding


def test_shadow_binding_preserves_unseen_trusted_origin_as_unbound():
    rule = {
        "op": "origin_eq",
        "left": {"op": "policy_slot", "name": "candidate_url"},
        "right": {"op": "policy_slot", "name": "trusted_origin"},
    }
    evidence = shadow_evidence(
        {"method": "GET", "path": "/redirect?to=https%3A%2F%2Fexample.com"},
        {"status_code": 302, "headers": {"location": "https://example.com"}, "body_length": 0, "body_shape": "unknown"},
    )
    bound = bind_rule_ir_slots(rule, evidence)
    assert bound["status"] == "partially_bound"
    assert bound["bindings"]["candidate_url"]["status"] == "bound"
    assert bound["bindings"]["trusted_origin"]["status"] == "unbound"
    assert bound["executable"] is False
    assert validate_binding(bound)["valid"] is True


def test_shadow_binding_marks_metrics_as_public_artifact_evidence():
    rule = {"op": "not", "arg": {"op": "policy_slot", "name": "sensitive_operational_artifact_is_public"}}
    evidence = shadow_evidence(
        {"method": "GET", "path": "/metrics"},
        {"status_code": 200, "headers": {"content-type": "text/plain; version=0.0.4"}, "body_length": 1200, "body_shape": "prometheus"},
    )
    bound = bind_rule_ir_slots(rule, evidence)
    assert bound["bindings"]["sensitive_operational_artifact_is_public"]["value"] is True
    assert len(bound["evidence_hash"]) == 64


def test_transport_error_does_not_bind_authentication_from_status_zero():
    rule = {"op": "policy_slot", "name": "subject_authenticated"}
    evidence = shadow_evidence(
        {"method": "GET", "path": "/session"},
        {
            "status_code": 0,
            "headers": {},
            "body_length": 0,
            "body_shape": "unknown",
            "transport_error": "timeout",
        },
    )
    bound = bind_rule_ir_slots(rule, evidence)
    assert bound["bindings"]["subject_authenticated"]["status"] == "unbound"
    assert bound["bindings"]["subject_authenticated"]["value"] is None
    assert validate_binding(bound)["valid"] is True


def test_validate_binding_rejects_mutated_derived_slot():
    rule = {"op": "policy_slot", "name": "subject_authenticated"}
    evidence = shadow_evidence(
        {"method": "GET", "path": "/session"},
        {"status_code": 200, "headers": {}, "body_length": 0, "body_shape": "json"},
    )
    bound = bind_rule_ir_slots(rule, evidence)
    tampered = deepcopy(bound)
    tampered["bindings"]["subject_authenticated"]["value"] = False
    with pytest.raises(ValueError, match="binding derivation mismatch"):
        validate_binding(tampered)
