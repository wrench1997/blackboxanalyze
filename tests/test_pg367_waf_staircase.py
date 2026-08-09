from __future__ import annotations

import pytest

from app.pg367_waf_staircase import POLICIES, build_failure_transition, evaluate_waf_probe


def _probe(**updates: object) -> dict[str, object]:
    result = {"role": "candidate", "method": "GET", "field_role": "query_term", "syntax_category": "marker", "encoding_chain": "identity"}
    result.update(updates)
    return result


def test_allow_policy_confirms_only_abstract_effect() -> None:
    result = evaluate_waf_probe(POLICIES[0], _probe())
    assert result["typed_effect_confirmed"] is True
    assert result["raw_payload_stored"] is False
    assert result["external_network"] is False


def test_filter_failure_is_typed_as_process_observation() -> None:
    result = evaluate_waf_probe(POLICIES[1], _probe())
    assert result["typed_effect_confirmed"] is False
    assert result["failure_signature"] == "encoded_delimiter"
    assert result["repair_axis"] == "encoding"


def test_negative_never_becomes_positive() -> None:
    result = evaluate_waf_probe(POLICIES[0], _probe(role="negative"))
    assert result["typed_effect_confirmed"] is False
    assert result["negative_control_clean"] is True


def test_failure_repair_changes_abstract_action() -> None:
    before = evaluate_waf_probe(POLICIES[1], _probe())
    after_probe = _probe(encoding_chain="url_percent", syntax_category="delimiter_boundary")
    transition = build_failure_transition(POLICIES[1], before, after_probe)
    assert transition["action_changed"] is True
    assert transition["changed_axis"] == "encoding"


@pytest.mark.parametrize("key", ["payload", "raw_payload", "wire", "response_body"])
def test_raw_fields_are_rejected(key: str) -> None:
    with pytest.raises(ValueError):
        evaluate_waf_probe(POLICIES[0], _probe(**{key: "forbidden"}))
